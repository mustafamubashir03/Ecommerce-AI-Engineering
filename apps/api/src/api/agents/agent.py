"""Both graph nodes: the intent gate, and the agent that answers.

`intent_router_node` runs first and decides whether the question is about the
catalogue at all. `agent_node` is the shopping assistant itself, a ReAct agent
that owns the retrieval tool and writes the answer.

They are in one file because they are the graph. `graph.py` wires them together;
this is what it wires.
"""

import re
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from api.agents.llm import get_chat_model, get_instructor_client
from api.agents.prompts import prompt
from api.agents.retrieval import hydrate_used_context
from api.agents.text import join_tool_output, last_message_text, optional
from api.agents.tools import retrieve_data_tool
from api.core.settings import get_settings

CITATIONS = re.compile(r"\[([^\[\]\n]{2,64})\]")
PRODUCT_ID = re.compile(r"^[A-Z0-9]{8,14}$")


# --- node 1: is this question about the catalogue? --------------------------


class IntentRouterResponse(BaseModel):
    """This node's own answer: relevant, or not, and why."""

    question_relevancy: bool = Field(
        description="True when the question can be answered with products in stock."
    )
    answer: str = Field(
        description="Empty when relevant, otherwise why the question is out of scope."
    )


@traceable(name="intent_router", run_type="llm")
def intent_router_node(state) -> dict:
    response = get_instructor_client().chat.completions.create(
        model=get_settings().llm.model,
        messages=[
            {"role": "system", "content": prompt("intent_router")},
            {"role": "user", "content": optional(state.get("initial_query"))},
        ],
        response_model=IntentRouterResponse,
    )
    return {
        "question_relevancy": response.question_relevancy,
        "answer": response.answer,
    }


def intent_router_condition_edges(state) -> str:
    """Relevant goes to the agent, anything else ends the turn."""
    return "agent_node" if state.get("question_relevancy") else "end"


# --- node 2: the shopping assistant ----------------------------------------


def _cited_ids(text: str) -> list:
    """Product ids quoted in the answer, in order, without duplicates.

    The model writes them as `[B0ABC12345]`, and may wrap them in markdown
    emphasis, so the quoting characters are stripped before the id is checked.
    """
    found: list[str] = []
    for quoted in CITATIONS.findall(text or ""):
        cleaned = quoted.strip().strip("*_` ").replace(" ", "")
        if PRODUCT_ID.match(cleaned) and cleaned not in found:
            found.append(cleaned)
    return found


@lru_cache(maxsize=1)
def get_agent():
    """The compiled agent, built on first use so a bad key cannot stop boot."""
    return create_agent(
        model=get_chat_model(),
        tools=[retrieve_data_tool],
        system_prompt=prompt("shopping_agent"),
        name="shopping_assistant",
    )


@traceable(name="agent_node", run_type="llm")
def agent_node(state) -> dict:
    history = state.get("messages", [])
    result = get_agent().invoke(
        {"messages": [*history, HumanMessage(content=state["initial_query"])]},
        config={"recursion_limit": get_settings().agent.max_iterations},
    )

    messages = result["messages"]
    answer = last_message_text(messages)

    cited = _cited_ids(answer)
    if not cited:
        cited = _cited_ids(join_tool_output(messages))

    return {
        "answer": answer,
        "messages": messages[len(history) :],
        "used_context": hydrate_used_context([(product_id, "") for product_id in cited]),
    }
