import uuid
from typing import Annotated, Any, Dict, Iterator, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from api.agents.agent import (
    agent_node,
    intent_router_condition_edges,
    intent_router_node,
)
from api.agents.checkpointer import get_checkpointer

EMPTY_ANSWER = "Please ask a question about the products in stock."


class State(TypedDict, total=False):
    initial_query: str
    question_relevancy: bool
    answer: str
    messages: Annotated[List[AnyMessage], add_messages]
    used_context: List[Dict[str, Any]]


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_edge(START, "intent_router_node")
    workflow.add_conditional_edges(
        "intent_router_node",
        intent_router_condition_edges,
        {"agent_node": "agent_node", "end": END},
    )
    workflow.add_edge("agent_node", END)
    return workflow.compile(checkpointer=get_checkpointer())


graph = build_graph()


def _config(thread_id: str | None) -> dict:
    return {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}


def _result(state: dict, thread_id: str) -> dict:
    return {
        "answer": state.get("answer", ""),
        "used_context": state.get("used_context", []),
        "question_relevancy": state.get("question_relevancy", False),
        "thread_id": thread_id,
    }


def run_agent(query: str, thread_id: str = None) -> dict:
    """Blocking single turn. Reuse the same thread_id to keep the history."""
    if not query or not query.strip():
        return _result({"answer": EMPTY_ANSWER}, thread_id or "")

    config = _config(thread_id)
    state = graph.invoke({"initial_query": query}, config=config)
    return _result(state, config["configurable"]["thread_id"])


def stream_agent(query: str, thread_id: str = None) -> Iterator[dict]:
    """Yield SSE ready events while the agent is still working.

    `messages` gives the tokens as the model produces them, `values` gives the
    finished state once the agent loop ends.
    """
    if not query or not query.strip():
        yield {"type": "result", "payload": _result({"answer": EMPTY_ANSWER}, thread_id or "")}
        return

    config = _config(thread_id)
    result_sent = False
    for mode, chunk in graph.stream(
        {"initial_query": query},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            token, _metadata = chunk
            if getattr(token, "type", None) != "ai":
                continue
            text = token.text
            if text:
                yield {"type": "token", "text": text}
        elif mode == "values" and chunk.get("answer") and not result_sent:
            result_sent = True
            yield {
                "type": "result",
                "payload": _result(chunk, config["configurable"]["thread_id"]),
            }
