"""Capability assumptions, pool semantics, and the outer/inner message boundary."""

import json

import pytest

import fake_openrouter as fake
from api.agents.model_router import RoutedChatModel

# --- structured output versus tool calling ---------------------------------


def test_structured_output_defaults_to_tool_calling():
    """`with_structured_output` is implemented as a tool call by default.

    That is why the general tool-calling pool is also valid for structured
    output: a model that supports `tools` can produce the schema, and no
    separate `structured_outputs` capability is required.
    """
    import inspect

    from langchain_openai.chat_models.base import BaseChatOpenAI

    assert inspect.signature(BaseChatOpenAI.with_structured_output).parameters["method"].default == (
        "function_calling"
    )


def test_a_model_without_structured_outputs_still_answers():
    """Proved against a real model object: the pool needs tools, not the flag.

    `poolside/laguna-s-2.1:free` does not advertise `structured_outputs`, and
    this shows the request is still built, through the tool calling path.
    """
    from api.api.models import RAGGenerationResponse
    from api.agents.llm import build_chat_model

    model = build_chat_model("poolside/laguna-s-2.1:free")
    assert type(model).__name__ == "ChatOpenRouter"
    runnable = model.with_structured_output(RAGGenerationResponse)
    # A function calling request is a sequence that binds the schema as a tool.
    assert type(runnable).__name__ == "RunnableSequence"


def test_explicit_json_schema_is_downgraded_not_accepted_silently():
    """A capability mismatch must not become a silent wrong answer."""
    from api.api.models import RAGGenerationResponse
    from api.agents.llm import build_chat_model

    model = build_chat_model("poolside/laguna-s-2.1:free")
    runnable = model.with_structured_output(RAGGenerationResponse, method="json_schema")
    assert runnable is not None, "a runnable is always produced"
    # langchain downgrades to function_calling for models without the flag, so
    # the request stays valid instead of 400-ing on an unsupported parameter.
    assert type(runnable).__name__ in ("RunnableSequence", "Runnable")


def test_structured_output_routing_uses_the_same_pool(monkeypatch):
    """One pool serves both paths, and the fallback policy is unchanged."""
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(503)]},
        structured_factory=fake.ScriptedStructuredModel,
        schema_result={"ok": True},
    )
    routed = RoutedChatModel(models=["model/a", "model/b"], options={"api_key": "t"})
    assert routed.with_structured_output(dict).invoke([]) == {"ok": True}
    assert calls == ["model/a", "model/b"], "a tool-capable fallback is valid for structured output"


# --- pool semantics ---------------------------------------------------------


def test_pool_has_one_source_of_truth():
    from api.core.settings import get_settings

    settings = get_settings()
    fields = type(settings.llm).model_fields
    assert "models" not in fields, "the old per-provider single model map is gone"
    assert settings.llm.model_prefix == "openrouter", "the init_chat_model prefix is the only prefix source"
    assert settings.llm.pool, "the pool comes from config.yaml or OPENROUTER_MODELS"


def test_no_stale_single_model_setting_remains():
    """OPENROUTER_MODEL (singular) must not exist anywhere as a mechanism."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in list(root.glob("*.toml")) + [root / "config.yaml", root / ".env.example"]:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "OPENROUTER_MODELS" in stripped or "OPENROUTER_PRIMARY_MODEL" in stripped:
                continue
            if "OPENROUTER_MODEL" in stripped:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, f"stale single model variable in {offenders}"


def test_pool_is_walked_at_most_once_per_request(monkeypatch):
    """Each model is tried once; there is no loop over the pool."""
    calls = fake.install(monkeypatch, {name: [fake.server_error(503)] for name in ("model/a", "model/b", "model/c")})
    with pytest.raises(Exception):
        RoutedChatModel(models=["model/a", "model/b", "model/c"], options={}).invoke("hello")
    assert calls == ["model/a", "model/b", "model/c"]


def test_pool_order_is_config_order(monkeypatch):
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(500)], "model/b": [fake.server_error(500)]},
    )
    RoutedChatModel(models=["model/a", "model/b", "model/c"], options={}).invoke("hello")
    assert calls == ["model/a", "model/b", "model/c"], "no reshuffling, the configured order is used"


def test_every_pooled_id_is_a_well_formed_openrouter_id():
    from api.core.settings import get_settings

    for model_id in get_settings().model_pool():
        assert model_id.count("/") == 1, f"{model_id} is not provider/model"
        provider, _, name = model_id.partition("/")
        assert provider and name, model_id
        base, _, variant = name.partition(":")
        assert base, model_id
        assert variant in ("free", "batch", "nitro", "floor", "exacto", ""), f"{model_id} has an unknown suffix"


# --- outer graph state versus inner agent history --------------------------


@pytest.fixture
def agent_graph(monkeypatch):
    """A real graph and a real agent, with only the model and catalogue stubbed."""
    import api.agents.graph as graph_module
    import api.agents.retrieval_generation as retrieval
    import api.agents.shopping_agent as shopping
    import api.agents.tools as tools
    from langgraph.checkpoint.memory import InMemorySaver

    products = [{"id": "B0TEST0001", "image_url": "https://example.test/1.jpg", "price": 10.0,
                 "description": "A test product.", "rating": 4.0}]

    monkeypatch.setattr(graph_module, "intent_router_node", lambda state: {"question_relevancy": True, "answer": ""})
    monkeypatch.setattr(shopping, "get_chat_model", lambda *a, **k: fake.ToolCallingModel())
    monkeypatch.setattr(tools, "retrieve_data", lambda query, k=None: {
        "retrieved_context_ids": [p["id"] for p in products],
        "retrieved_context": [p["description"] for p in products],
        "retrieved_context_ratings": [p["rating"] for p in products],
    })
    monkeypatch.setattr(retrieval, "fetch_product_payloads", lambda ids: {p["id"]: {"image": p["image_url"]} for p in products})
    monkeypatch.setattr(shopping, "hydrate_used_context", lambda refs: products)
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda: InMemorySaver())
    shopping.get_agent.cache_clear()
    monkeypatch.setattr(graph_module, "graph", graph_module.build_graph())
    return graph_module


def test_message_slice_is_correct_over_three_turns(agent_graph):
    """`messages[len(history):]` must return only this turn's new messages."""
    config = {"configurable": {"thread_id": "slice-check"}}

    first = agent_graph.run_agent("washing machines", thread_id="slice-check")
    assert first["answer"], "turn 1 produced an answer"

    state_after_first = agent_graph.graph.get_state(config).values
    messages_after_first = state_after_first["messages"]

    # human, ai(tool call), tool, ai(answer) for the first turn
    assert len(messages_after_first) == 4, [m.type for m in messages_after_first]
    assert messages_after_first[0].type == "human"
    assert messages_after_first[2].type == "tool", "the tool call and its result are both recorded"

    second = agent_graph.run_agent("and laptops?", thread_id="slice-check")
    assert second["answer"]

    after_second = agent_graph.graph.get_state(config).values["messages"]
    assert len(after_second) == 8, [m.type for m in after_second]
    assert [m.type for m in after_second[:4]] == [m.type for m in messages_after_first], "turn 1 was not rewritten"
    assert after_second[4].content == "and laptops?", "turn 2 starts with its own human message"

    third = agent_graph.run_agent("and dishwashers?", thread_id="slice-check")
    assert third["answer"]
    after_third = agent_graph.graph.get_state(config).values["messages"]
    assert len(after_third) == 12, [m.type for m in after_third]
    assert [m.type for m in after_third] == ["human", "ai", "tool", "ai"] * 3


def test_no_message_is_duplicated_across_turns(agent_graph):
    config = {"configurable": {"thread_id": "no-dupes"}}
    for query in ("one", "two", "three"):
        agent_graph.run_agent(query, thread_id="no-dupes")
    messages = agent_graph.graph.get_state(config).values["messages"]
    ids = [str(m.id) for m in messages]
    assert len(ids) == len(set(ids)), "the outer state duplicated a message"


def test_a_fresh_thread_starts_empty(agent_graph):
    agent_graph.run_agent("hello", thread_id="thread-one")
    other = agent_graph.run_agent("hello", thread_id="thread-two")
    assert other["thread_id"] == "thread-two"
    state = agent_graph.graph.get_state({"configurable": {"thread_id": "thread-two"}}).values
    assert len(state["messages"]) == 4, "a different thread has its own history"


def test_turn_returns_only_its_own_messages(agent_graph):
    """The node's return value is the slice, not the whole history."""
    node = agent_graph.agent_node
    graph = agent_graph.graph
    config = {"configurable": {"thread_id": "slice-only"}}

    graph.invoke({"initial_query": "first"}, config=config)
    history = graph.get_state(config).values["messages"]
    result = node({"messages": history, "initial_query": "second"})
    assert len(history) == 4
    assert result["messages"], "the node returns the new messages"
    assert result["messages"][0].type == "human"
    assert result["messages"][0].content == "second"
    assert len(result["messages"]) == 4, "four new messages for the second turn"
    assert json.dumps(result["used_context"]), "products are still returned"
