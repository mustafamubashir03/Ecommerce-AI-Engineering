"""The existing API contract must survive the routing work.

These tests mock the model, never the retrieval contract: the graph, the tool,
the hydration and the HTTP/SSE shapes all run for real, so a change to the
routing layer cannot quietly alter what the client receives.
"""

import json

import pytest
from fastapi.testclient import TestClient

import fake_openrouter as fake

PRODUCTS = [
    {
        "id": "B0TEST0001",
        "image_url": "https://example.test/one.jpg",
        "price": 19.9,
        "description": "Silent Guys Anti Vibration Pads for Washing Machine.",
        "rating": 4.2,
    },
    {
        "id": "B0TEST0002",
        "image_url": "https://example.test/two.jpg",
        "price": None,
        "description": "Portable Washer with no recorded price.",
        "rating": 4.6,
    },
]

EMPTY_ANSWER = "Please ask a question about the products in stock."


@pytest.fixture
def client(monkeypatch):
    """The real app, with only the model and the catalogue replaced."""
    import api.agents.graph as graph_module
    import api.agents.retrieval as retrieval
    import api.agents.agent as shopping
    import api.agents.tools as tools
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(graph_module, "intent_router_node", lambda state: {"question_relevancy": True, "answer": ""})
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
    monkeypatch.setattr(
        retrieval, "fetch_product_payloads", lambda ids: {p["id"]: {"image": p["image_url"]} for p in PRODUCTS}
    )
    monkeypatch.setattr(shopping, "hydrate_used_context", lambda refs: [p for p in PRODUCTS])
    # An in memory saver keeps these tests hermetic and repeatable.
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda: InMemorySaver())
    shopping.get_agent.cache_clear()
    monkeypatch.setattr(graph_module, "graph", graph_module.build_graph())

    from api.app import app

    return TestClient(app, raise_server_exceptions=False)


# --- blocking endpoint ------------------------------------------------------


def test_agent_response_contract(client):
    body = client.post("/agent/", json={"query": "which washing machines do you have?"}).json()
    assert set(body) == {"request_id", "answer", "question_relevancy", "used_context", "thread_id"}
    assert isinstance(body["request_id"], str) and len(body["request_id"]) > 8
    assert body["answer"]
    assert body["question_relevancy"] is True
    assert body["used_context"] == PRODUCTS, "product context is passed through unchanged"
    assert body["thread_id"], "a thread id is generated when the client sends none"


def test_used_context_item_contract(client):
    item = client.post("/agent/", json={"query": "washing machines"}).json()["used_context"][0]
    assert set(item) == {"id", "image_url", "price", "description", "rating"}
    assert item["price"] is None or isinstance(item["price"], (int, float))
    assert item["rating"] is None or isinstance(item["rating"], (int, float))


def test_existing_thread_id_is_echoed_back(client):
    first = client.post("/agent/", json={"query": "washing machines", "thread_id": "keep-me"}).json()
    second = client.post("/agent/", json={"query": "and laptops?", "thread_id": "keep-me"}).json()
    assert first["thread_id"] == "keep-me"
    assert second["thread_id"] == "keep-me", "the client's thread id is preserved across turns"


def test_request_id_differs_per_request(client):
    one = client.post("/agent/", json={"query": "a"}).json()["request_id"]
    two = client.post("/agent/", json={"query": "b"}).json()["request_id"]
    assert one != two


def test_empty_query_is_answered_without_a_model_call(client):
    body = client.post("/agent/", json={"query": "   "}).json()
    assert body["answer"]
    assert body["used_context"] == []


# --- streaming endpoint -----------------------------------------------------


def test_stream_event_contract(client):
    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines", "thread_id": "s1"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert [event["type"] for event in events][-1] == "done", "a stream always terminates"
    result = next(event for event in events if event["type"] == "result")
    assert set(result["payload"]) == {
        "request_id",
        "answer",
        "question_relevancy",
        "used_context",
        "thread_id",
    }
    assert result["payload"]["used_context"] == PRODUCTS
    assert result["payload"]["thread_id"] == "s1"
    assert result["payload"]["question_relevancy"] is True


def test_blocking_and_streaming_expose_the_same_contract(client):
    """One logical contract: a streamed answer carries the same fields."""
    blocking = client.post("/agent/", json={"query": "washing machines", "thread_id": "same"}).json()
    events = []
    with client.stream(
        "POST", "/agent/stream", json={"query": "washing machines", "thread_id": "same"}
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    streamed = next(event for event in events if event["type"] == "result")["payload"]

    assert set(blocking) == set(streamed)
    assert streamed["used_context"] == blocking["used_context"]
    assert streamed["question_relevancy"] == blocking["question_relevancy"]
    assert streamed["thread_id"] == blocking["thread_id"] == "same"
    # The request id identifies this request, so the two differ.
    assert blocking["request_id"] != streamed["request_id"]


def test_stream_carries_tokens(client):
    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert any(event["type"] == "token" for event in events), "the answer arrives as tokens"


def test_stream_does_not_leak_the_question_or_the_tool_output(client):
    """Only the assistant's own writing belongs in the chat bubble.

    The stream also carries the human message and the raw tool result, and the
    client appends every token it receives to the assistant message, so any
    token that is not the assistant's text is printed into the conversation.
    """
    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    streamed = "".join(event["text"] for event in events if event["type"] == "token")
    result = next(event for event in events if event["type"] == "result")["payload"]

    assert streamed.strip() == result["answer"].strip(), (
        "the tokens must add up to exactly the answer, nothing else"
    )
    assert "washing machines" not in streamed.lower(), "the question was echoed back into the chat"
    for item in PRODUCTS:
        assert item["description"] not in streamed, (
            "a retrieved description reached the chat, so the raw tool output leaked: " + item["id"]
        )
    assert len(streamed) < 1000, f"a 2 product answer must not stream {len(streamed)} characters"


def test_off_topic_questions_finish_cleanly(client, monkeypatch):
    """An off-topic question must not crash the turn.

    The intent router ends the graph for a question that is not about the
    catalogue, and that branch is easy to get wrong: returning langgraph's END
    instead of the map's own key raises KeyError('__end__'), which reaches the
    client as a 500 rather than as a polite "ask about the catalogue".
    """
    import api.agents.graph as graph_module

    monkeypatch.setattr(
        graph_module, "intent_router_node", lambda state: {"question_relevancy": False, "answer": EMPTY_ANSWER}
    )
    graph_module.graph = graph_module.build_graph()

    response = client.post("/agent/", json={"query": "who won the 1998 world cup?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == EMPTY_ANSWER
    assert body["question_relevancy"] is False
    assert body["used_context"] == [], "an off-topic question retrieves nothing"


def test_off_topic_questions_also_stream_cleanly(client, monkeypatch):
    import api.agents.graph as graph_module

    monkeypatch.setattr(
        graph_module, "intent_router_node", lambda state: {"question_relevancy": False, "answer": EMPTY_ANSWER}
    )
    graph_module.graph = graph_module.build_graph()

    events = []
    with client.stream("POST", "/agent/stream", json={"query": "who won the 1998 world cup?"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    result = next(event for event in events if event["type"] == "result")["payload"]
    assert result["answer"] == EMPTY_ANSWER
    assert result["used_context"] == []
    assert events[-1]["type"] == "done", "the stream still closes cleanly"


def test_get_chat_model_returns_a_usable_model():
    """The agents' entry point has to actually build a model.

    `get_chat_model` is the one call the graph makes, and it is normally mocked
    in tests, so a broken import inside it stays invisible until a real request
    arrives. This calls it for real. The key is read from the environment, and
    the client is never asked to answer, so nothing reaches the network.
    """
    from api.agents.llm import get_chat_model, provider_label
    from langchain_openai import ChatOpenAI

    model = get_chat_model()
    assert isinstance(model, ChatOpenAI), f"got {type(model).__name__}"
    assert model.model_name, "the configured model reached the client"
    assert model.openai_api_base, "the configured base_url reached the client"
    assert model.request_timeout, "the configured timeout reached the client"
    assert provider_label()


def test_a_failure_after_the_result_does_not_also_report_an_error(client, monkeypatch):
    """A finished answer must not arrive next to an error banner.

    The graph can deliver the answer and its products and then fail on
    something later. The turn is complete from the client's point of view, so
    the stream closes cleanly instead of sending an error the UI would render
    as a failure under a complete answer.
    """
    import api.api.endpoints as endpoints

    def result_then_failure(query, thread_id=None):
        yield {"type": "token", "text": "an answer"}
        yield {"type": "result", "payload": {"answer": "an answer", "used_context": PRODUCTS}}
        raise fake.payload_too_large_error()

    monkeypatch.setattr(endpoints, "stream_agent", result_then_failure)

    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    kinds = [event["type"] for event in events]
    assert kinds == ["token", "result", "done"], kinds
    assert not any(event["type"] == "error" for event in events), (
        "an error after a complete result would render as a failure banner"
    )


def test_a_failure_before_the_result_is_still_reported(client, monkeypatch):
    """The opposite case: nothing was delivered, so the client must be told."""
    import api.api.endpoints as endpoints

    def fails_immediately(query, thread_id=None):
        raise fake.server_error(503)

    monkeypatch.setattr(endpoints, "stream_agent", fails_immediately)

    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert [event["type"] for event in events] == ["error", "done"], events
    assert events[0]["status"] == 503


def test_exactly_one_result_event_per_turn(client):
    """The client stores a result by running its handler, so a duplicate result
    is a duplicated state update, and a stream contract that varies in length is
    harder to reason about than one that does not."""
    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert [event["type"] for event in events].count("result") == 1, events
    assert events[-1]["type"] == "done"


def test_exactly_one_result_for_an_off_topic_turn(client, monkeypatch):
    import api.agents.graph as graph_module

    monkeypatch.setattr(
        graph_module, "intent_router_node", lambda state: {"question_relevancy": False, "answer": EMPTY_ANSWER}
    )
    graph_module.graph = graph_module.build_graph()

    events = []
    with client.stream("POST", "/agent/stream", json={"query": "who won the 1998 world cup?"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    assert [event["type"] for event in events].count("result") == 1, events
    assert events[-1]["type"] == "done"


def test_citations_survive_markdown_emphasis():
    """Models wrap catalogue ids in emphasis, e.g. [**B0TEST0001**].

    The id has to be recovered from that, otherwise the answer renders no
    product cards at all, because no id survives to be looked up.
    """
    from api.agents.agent import _cited_ids

    assert _cited_ids("Here you go [**B0TEST0001**] and [B0TEST0002]") == [
        "B0TEST0001",
        "B0TEST0002",
    ]
    assert _cited_ids("plain [B0TEST0001]") == ["B0TEST0001"]
    assert _cited_ids("underscored [_B0TEST0001_]") == ["B0TEST0001"]
    assert _cited_ids("backticked `[B0TEST0001]`") == ["B0TEST0001"]
    assert _cited_ids("repeated [B0TEST0001] twice [B0TEST0001]") == ["B0TEST0001"]
    assert _cited_ids("no citations at all") == []
    assert _cited_ids("") == []


def test_only_real_catalogue_ids_are_cited():
    """Bracketed text that is not an id must not be treated as a product."""
    from api.agents.agent import _cited_ids

    assert _cited_ids("no results [1] [two words] [tool_use_failed]") == []


def test_blocking_answers_cite_the_ids_the_products_came_from(client, monkeypatch):
    """The ids in the answer must be the ids the client receives.

    This is the link the chat's citation buttons follow, so a mismatch here is
    what makes a citation unclickable. The agent is cached, so the cache is
    cleared after the model is replaced, otherwise the scripted model is never
    the one that answers and the test passes without testing anything.
    """
    import api.agents.agent as shopping

    # cite=False, otherwise the fake appends its own plain [id] citations to the
    # scripted answer and the emphasis is never the only thing under test.
    answered = "These are ours [**B0TEST0001**], [_B0TEST0002_]."
    monkeypatch.setattr(
        shopping, "get_chat_model", lambda *a, **k: fake.ToolCallingModel(answer=answered, cite=False)
    )
    shopping.get_agent.cache_clear()

    body = client.post("/agent/", json={"query": "washing machines"}).json()
    returned = [item["id"] for item in body["used_context"]]
    cited = shopping._cited_ids(body["answer"])

    assert "[**B0TEST0001**]" in body["answer"], "the scripted answer really is emphasised"
    assert cited == ["B0TEST0001", "B0TEST0002"], f"emphasis was not recovered: {cited}"
    assert set(cited) <= set(returned), f"cited but not returned: {sorted(set(cited) - set(returned))}"


def test_stream_error_event_carries_the_real_status(client, monkeypatch):
    """The provider's status reaches the client; its own words do not.

    The message is written for the caller, so the quota marker the provider sent
    is no longer in it. `tests/test_error_sanitization.py` covers that in full.
    """
    import api.api.endpoints as endpoints

    def failing(*args, **kwargs):
        raise fake.daily_cap_error()

    # The endpoint imported the function by name, so patch it where it is used.
    monkeypatch.setattr(endpoints, "stream_agent", failing)
    events = []
    with client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    error = next(event for event in events if event["type"] == "error")
    assert error["status"] == 429, "the provider's own status is reported"
    assert "free-models-per-day" not in error["message"], "the raw body is not sent to the client"
    assert "rate limiting" in error["message"].lower(), "the caller is told what happened"
    assert events[-1]["type"] == "done", "the stream still closes cleanly"


# --- the provider setup -----------------------------------------------------


def test_one_model_one_base_url_one_key():
    """The whole provider setup is three values, and nothing else.

    Anything more than that is a second opinion about which provider to talk to,
    which is how the routing layer grew in the first place.
    """
    from api.core.settings import get_settings

    llm = get_settings().llm
    assert llm.model, "a model is configured"
    assert llm.base_url, "a base url is configured"
    assert llm.api_key_env, "the env var holding the key is named"

    for gone in ("pool", "primary_model", "fallback_order", "active", "key_envs", "base_urls"):
        assert not hasattr(llm, gone), f"llm.{gone} is a leftover from the routing layer"


def test_the_configured_model_is_free_and_tool_capable():
    """The agent needs tool calling, so the model has to support it.

    Cross checks the configured id against OpenRouter's own catalog. This is a
    catalog read, not a model request, so it needs no quota; it skips when the
    catalog cannot be reached.
    """
    import json
    import urllib.request

    from api.core.settings import get_settings

    llm = get_settings().llm
    if "openrouter" not in llm.base_url:
        pytest.skip("not an OpenRouter configuration, so the catalog says nothing useful")

    try:
        with urllib.request.urlopen(llm.base_url.rstrip("/") + "/models", timeout=30) as response:
            catalog = json.loads(response.read())
    except Exception as error:  # offline or blocked: the check simply cannot run
        pytest.skip(f"could not read the OpenRouter catalog: {type(error).__name__}")

    entries = {entry["id"]: entry for entry in catalog.get("data", [])}
    assert entries, "the catalog returned no models"

    entry = entries.get(llm.model)
    assert entry is not None, f"{llm.model} is not in the OpenRouter catalog"
    pricing = entry.get("pricing") or {}
    assert str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0", (
        f"{llm.model} is not free"
    )
    assert entry["architecture"]["output_modalities"] == ["text"], f"{llm.model} is not a text chat model"
    assert "tools" in (entry.get("supported_parameters") or []), f"{llm.model} cannot call tools"


def test_no_direct_provider_sdk_in_the_request_path():
    """The agents reach a model only through the OpenRouter client."""
    import inspect

    import api.agents.llm as llm

    source = inspect.getsource(llm)
    assert "openrouter.ai/api/v1" not in source, "the base url belongs in config.yaml, not in code"
    for other in ("api.openai.com", "generativelanguage", "api.anthropic.com", "api.cohere"):
        assert other not in source
