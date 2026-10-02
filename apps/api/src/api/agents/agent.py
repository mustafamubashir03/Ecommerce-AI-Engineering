import json
import logging
import re
from functools import lru_cache
from typing import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, ValidationError

from api.agents.llm import get_chat_model, get_instructor_client
from api.agents.prompts import prompt
from api.agents.retrieval import hydrate_used_context
from api.agents.text import join_tool_output, last_message_text, optional
from api.agents.tools import retrieve_data_tool, retrieve_reviews_data
from api.core.settings import get_settings
from api.core.tracing import current_trace_id, trace_usage, traced

logger = logging.getLogger(__name__)

CITATIONS = re.compile(r"\[([^\[\]\n]{2,64})\]")
PRODUCT_ID = re.compile(r"^[A-Z0-9]{8,14}$")


class IntentRouterResponse(BaseModel):
    """This node's own answer: relevant, or not, and why."""

    question_relevancy: bool = Field(
        description=(
            "True when the customer is asking about the products this store sells, "
            "false when they are asking about anything else."
        )
    )
    answer: str = Field(
        description=(
            "Empty when relevant. Otherwise, why the question is out of scope."
        )
    )


def _parse_intent(raw: str) -> IntentRouterResponse:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return IntentRouterResponse(**json.loads(text[start : end + 1]))
        except (TypeError, ValueError) as error:
            logger.warning("intent router answer was not readable: %s", error)
    return IntentRouterResponse(question_relevancy=True, answer="")


def _intent_parse_failures() -> tuple:
    from instructor.core.exceptions import InstructorError, ResponseParsingError

    return (
        InstructorError,
        ResponseParsingError,
        TypeError,
        AttributeError,
        KeyError,
        IndexError,
    )


@traced(name="intent_router", run_type="llm")
def intent_router_node(state) -> dict:
    question = optional(state.get("initial_query"))
    messages = [
        {"role": "system", "content": prompt("intent_router")},
        {"role": "user", "content": question},
    ]

    try:
        response = get_instructor_client().chat.completions.create(
            model=get_settings().llm.model,
            messages=messages,
            response_model=IntentRouterResponse,
        )
        return {
            "question_relevancy": response.question_relevancy,
            "answer": response.answer,
        }
    except _intent_parse_failures() as error:
        logger.warning("instructor could not read the intent (%s), reading the text instead", error)

    parsed = _parse_intent(get_chat_model().invoke(messages).content)
    return {"question_relevancy": parsed.question_relevancy, "answer": parsed.answer}


def intent_router_condition_edges(state) -> str:
    """Relevant goes to the agent, anything else ends the turn."""
    return "agent_node" if state.get("question_relevancy") else "end"


def _cited_ids(text: str) -> list:
    found: list[str] = []
    for quoted in CITATIONS.findall(text or ""):
        cleaned = quoted.strip().strip("*_` ").replace(" ", "")
        if PRODUCT_ID.match(cleaned) and cleaned not in found:
            found.append(cleaned)
    return found


def _compact_history(messages: list) -> list:
    last_question = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_question = index
            break

    if last_question is None:
        return list(messages)

    earlier_questions: list = []
    for message in messages[:last_question]:
        if not isinstance(message, HumanMessage):
            continue
        if earlier_questions and earlier_questions[-1].content == message.content:
            continue
        earlier_questions.append(message)

    current_turn = messages[last_question:]
    if (
        earlier_questions
        and isinstance(current_turn[0], HumanMessage)
        and earlier_questions[-1].content == current_turn[0].content
    ):
        earlier_questions.pop()

    return [*earlier_questions, *current_turn]


def agent_middleware() -> list:
    from langchain.agents.middleware import (
        ContextEditingMiddleware,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )
    from langchain.agents.middleware.context_editing import ClearToolUsesEdit

    return [
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=4, exit_behavior="continue"),
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=3500, keep=2)],
        ),
    ]


@lru_cache(maxsize=1)
def get_agent():
    """The compiled agent, built on first use so a bad key cannot stop boot."""
    return create_agent(
        model=get_chat_model(),
        tools=[retrieve_data_tool, retrieve_reviews_data],
        system_prompt=prompt("shopping_agent"),
        middleware=agent_middleware(),
        name="shopping_assistant",
    )


def _turn_usage(messages: list) -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    measured = False
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
                measured = True
    return totals if measured else {}


def _stream_writer():

    def discard(chunk) -> None:
        return None

    try:
        return get_stream_writer()
    except RuntimeError:
        return discard


def _answer_text(token, metadata) -> str:
    if not isinstance(token, (AIMessage, AIMessageChunk)):
        return ""
    if isinstance(metadata, dict) and metadata.get("langgraph_node") == "intent_router_node":
        return ""
    return getattr(token, "text", "") or ""


def _as_mode_chunk(item) -> tuple[str, object]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0], item[1]
    return "", item


def _stream_agent_turn(history: list, question: str) -> Iterator[tuple[str, object]]:
    for item in get_agent().stream(
        {"messages": [*history, HumanMessage(content=question)]},
        config={"recursion_limit": get_settings().agent.max_iterations},
        stream_mode=["messages", "values"],
    ):
        yield _as_mode_chunk(item)


@traced(name="agent_node", run_type="llm")
def agent_node(state) -> dict:
    question = state["initial_query"]
    prompt = _compact_history([*state.get("messages", []), HumanMessage(content=question)])
    turn_history = prompt[:-1]

    write = _stream_writer()

    final: dict = {}

    for mode, chunk in _stream_agent_turn(turn_history, question):
        if mode == "messages":
            token, metadata = chunk
            text = _answer_text(token, metadata)
            if text:
                write({"token": text})
        elif mode == "values" and isinstance(chunk, dict) and chunk.get("messages"):
            final = chunk

    messages = final.get("messages", [])
    answer = last_message_text(messages)

    trace_usage(**_turn_usage(messages[len(turn_history) :]))

    cited = _cited_ids(answer)
    if not cited:
        cited = _cited_ids(join_tool_output(messages))

    return {
        "answer": answer,
        "messages": messages[len(turn_history) :],
        "used_context": hydrate_used_context([(product_id, "") for product_id in cited]),
        "trace_id": current_trace_id(),
    }
