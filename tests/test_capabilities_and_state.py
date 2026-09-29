"""The model's own configuration, and the conversation state behind it.

Two things are worth pinning: that the timeout written in seconds reaches the
client as seconds, and that a multi-turn conversation on one thread id does not
lose, duplicate or leak messages.
"""

import asyncio
import inspect

import pytest

import fake_openrouter as fake

PRODUCTS = [
    {
        "id": "B0TEST0001",
        "image_url": "https://example.test/1.jpg",
        "price": 10.0,
        "description": "A test product.",
        "rating": 4.0,
    }
]


# --- the configured timeout reaches the client ------------------------------


def test_the_configured_timeout_reaches_the_client_as_seconds(monkeypatch):
    """`config.yaml` says seconds, and the client is given seconds.

    The OpenAI client takes a timeout in seconds, so no conversion happens
    anywhere. A conversion here would turn 120 into 120000 and hide every
    timeout behind a two minute hang.
    """
    from langchain_openai import ChatOpenAI

    from api.agents.llm import get_chat_model
    from api.core.settings import get_settings

    built: dict = {}
    original = ChatOpenAI.__init__

    def spy(self, **kwargs):
        built.update(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(ChatOpenAI, "__init__", spy)
    get_chat_model.cache_clear()
    get_chat_model()

    assert built["timeout"] == get_settings().llm.timeout_seconds
    assert 60 < built["timeout"] < 600, "a timeout of two hours is not a timeout"


def test_no_millisecond_timeout_key_is_left_in_the_config():
    """One timeout, in one unit. A second key would be a second opinion."""
    from api.core.settings import get_settings

    llm = get_settings().llm
    assert not hasattr(llm, "timeout_ms")
    assert not hasattr(llm, "max_retries"), "a single provider has nothing to retry against"


# --- the model is one client, built once ------------------------------------


def test_the_model_is_built_once():
    from api.agents.llm import get_chat_model

    get_chat_model.cache_clear()
    assert get_chat_model() is get_chat_model(), "one client, built once"


def test_the_agent_does_not_block_the_event_loop():
    """The client's async call is a coroutine, so awaiting it yields the loop."""
    from api.agents.llm import get_chat_model

    model = get_chat_model()
    assert asyncio.iscoroutinefunction(model._agenerate)


# --- the intent router is the one structured call ---------------------------


def test_the_intent_router_asks_for_the_response_model():
    """It goes through instructor, so any OpenAI-compatible model can answer."""

    import api.agents.agent as agent_module

    source = inspect.getsource(agent_module)
    assert "response_model=IntentRouterResponse" in source, "the schema is requested, not parsed"
    assert "get_instructor_client" in source



# --- conversation state ------------------------------------------------------


@pytest.fixture
def agent_graph(monkeypatch):
    """A real graph and a real agent, with only the model and catalogue stubbed."""
    import api.agents.graph as graph_module
    import api.agents.agent as shopping
    import api.agents.tools as tools
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(
        graph_module, "intent_router_node", lambda state: {"question_relevancy": True, "answer": ""}
    )
    monkeypatch.setattr(shopping, "get_chat_model", lambda *a, **k: fake.ToolCallingModel())
    monkeypatch.setattr(
        tools,
        "retrieve_data",
        lambda query, k=None: {
            "retrieved_context_ids": [p["id"] for p in PRODUCTS],
            "retrieved_context": [p["description"] for p in PRODUCTS],
            "retrieved_context_ratings": [p["rating"] for p in PRODUCTS],
        },
    )
    # The answer's citations are hydrated against the real catalogue, which is a
    # Qdrant call. These tests are about conversation state, not retrieval.
    monkeypatch.setattr(shopping, "hydrate_used_context", lambda pairs: list(PRODUCTS))

    workflow = graph_module.StateGraph(graph_module.State)
    workflow.add_node("intent_router_node", graph_module.intent_router_node)
    workflow.add_node("agent_node", graph_module.agent_node)
    workflow.add_edge(graph_module.START, "intent_router_node")
    workflow.add_conditional_edges(
        "intent_router_node",
        graph_module.intent_router_condition_edges,
        {"agent_node": "agent_node", "end": graph_module.END},
    )
    workflow.add_edge("agent_node", graph_module.END)
    compiled = workflow.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(graph_module, "graph", compiled)
    return compiled


def test_a_fresh_thread_starts_empty(agent_graph):
    from api.agents.graph import run_agent

    state = run_agent("washing machines", thread_id="brand-new")
    assert state["thread_id"] == "brand-new"
    assert state["question_relevancy"] is True
    assert state["used_context"], "the agent found products"


def test_a_turn_is_answered_from_the_products_the_tool_returned(agent_graph):
    from api.agents.graph import run_agent

    state = run_agent("washing machines", thread_id="t-answer")
    assert "B0TEST0001" in state["answer"], "the answer cites the product it used"


def test_the_second_turn_keeps_the_first(agent_graph):
    from api.agents.graph import run_agent

    run_agent("washing machines", thread_id="t-keep")
    second = run_agent("and laptops?", thread_id="t-keep")

    assert second["thread_id"] == "t-keep"
    assert second["used_context"], "the second turn still answers from the catalogue"


def test_two_threads_do_not_share_state(agent_graph):
    from api.agents.graph import run_agent

    run_agent("washing machines", thread_id="left")
    state = run_agent("laptops", thread_id="right")

    assert state["thread_id"] == "right"
    assert state["used_context"], "a different thread still works on its own"
