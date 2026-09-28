"""The main agent: a ReAct agent that owns the retrieval tool.

`create_agent` binds the tools to the model for us, so the model decides on its
own how often to retrieve and with which query.
"""

import re
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langsmith import traceable

from api.agents.llm import get_chat_model
from api.agents.prompts import prompt
from api.agents.retrieval_generation import hydrate_used_context
from api.agents.text import join_tool_output, last_message_text
from api.agents.tools import retrieve_data_tool
from api.core.settings import get_settings

CITATIONS = re.compile(r"\[([^\[\]\n]{2,64})\]")
# A catalogue id: the shape Qdrant stores in parent_asin. Models like to wrap
# these in markdown, e.g. [**B091GHFVTK**], so emphasis is stripped before the
# id is looked up, otherwise every citation misses and no product is returned.
PRODUCT_ID = re.compile(r"^[A-Z0-9]{8,14}$")


def _cited_ids(text: str) -> list[str]:
    """Product ids quoted in the answer, in order, without duplicates."""
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
    """One agent turn. Token streaming is handled by the graph itself."""
    history = state.get("messages", [])
    result = get_agent().invoke(
        {"messages": [*history, HumanMessage(content=state["initial_query"])]},
        config={"recursion_limit": get_settings().agent.max_iterations},
    )

    messages = result["messages"]
    answer = last_message_text(messages)

    # Products the agent quoted, or everything it retrieved if it quoted none.
    cited = _cited_ids(answer)
    if not cited:
        cited = _cited_ids(join_tool_output(messages))

    return {
        "answer": answer,
        "messages": messages[len(history) :],
        "used_context": hydrate_used_context([(product_id, "") for product_id in cited]),
    }
