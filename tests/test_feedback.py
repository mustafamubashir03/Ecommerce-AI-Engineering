import pytest

TRACE = "01a0f1c2-f417-7071-b203-1b05192f811c"


class RecordingClient:
    """Stands in for the LangSmith client and remembers what it was asked to write."""

    def __init__(self):
        self.calls = []

    def create_feedback(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": "fb-1"}


@pytest.fixture
def recorder(monkeypatch):
    """The feedback endpoint, wired to a client that records instead of writing."""
    import api.api.endpoints as endpoints
    from fastapi.testclient import TestClient

    from api.app import app

    client = RecordingClient()
    monkeypatch.setattr(endpoints, "get_trace_client", lambda: client)
    with TestClient(app) as test_client:
        yield test_client, client


def post(test_client, **body):
    payload = {"trace_id": TRACE, **body}
    return test_client.post("/feedback/", json=payload)


def test_a_vote_is_written_as_thumbs(recorder):
    test_client, client = recorder

    response = post(test_client, feedback_score=1)

    assert response.status_code == 200
    assert [call["key"] for call in client.calls] == ["thumbs"]
    assert client.calls[0]["value"] == 1
    assert client.calls[0]["trace_id"] == TRACE


def test_a_down_vote_is_written_too(recorder):
    test_client, client = recorder

    post(test_client, feedback_score=-1)

    assert client.calls[0]["key"] == "thumbs"
    assert client.calls[0]["value"] == -1, "a negative vote is still a vote"


def test_a_comment_is_written_as_comment(recorder):
    test_client, client = recorder

    post(test_client, feedback_text="the third one is not a washing machine")

    assert [call["key"] for call in client.calls] == ["comment"]
    assert client.calls[0]["value"] == "the third one is not a washing machine"


def test_a_vote_and_a_comment_are_two_keys(recorder):
    """They are answered differently, so they are not stored together."""
    test_client, client = recorder

    post(test_client, feedback_score=1, feedback_text="helpful but wordy")

    assert [call["key"] for call in client.calls] == ["thumbs", "comment"]


def test_a_score_of_zero_is_still_recorded(recorder):
    """A plain `if score` would drop this and silently record nothing."""
    test_client, client = recorder

    post(test_client, feedback_score=0)

    assert [call["key"] for call in client.calls] == ["thumbs"]
    assert client.calls[0]["value"] == 0


def test_no_vote_writes_no_thumbs(recorder):
    test_client, client = recorder

    post(test_client, feedback_text="just a comment")

    assert not [call for call in client.calls if call["key"] == "thumbs"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_a_blank_comment_is_not_written(recorder, text):
    """Whitespace is not a comment, and an empty one is not worth a write."""
    test_client, client = recorder

    post(test_client, feedback_score=1, feedback_text=text)

    assert not [call for call in client.calls if call["key"] == "comment"]


def test_a_surrounding_comment_is_trimmed(recorder):
    test_client, client = recorder

    post(test_client, feedback_text="  useful  ")

    assert client.calls[0]["value"] == "useful"


def test_feedback_is_tagged_as_coming_from_the_api_by_default(recorder):
    """LangSmith only defines 'api' and 'model'; anything else is refused."""
    test_client, client = recorder

    post(test_client, feedback_score=1)

    assert client.calls[0]["feedback_source_type"] == "api"


def test_a_model_source_can_be_declared(recorder):
    test_client, client = recorder

    post(test_client, feedback_score=1, feedback_source_type="model")

    assert client.calls[0]["feedback_source_type"] == "model"


def test_an_unknown_source_type_is_refused(recorder):
    """Better a 422 than a write that LangSmith would refuse at the last moment."""
    test_client, client = recorder

    response = post(test_client, feedback_score=1, feedback_source_type="user")

    assert response.status_code == 422
    assert client.calls == []


def test_the_conversation_is_recorded_with_the_feedback(recorder):
    """So a feedback row can be traced back to the exchange it is about."""
    test_client, client = recorder

    post(test_client, feedback_score=1, thread_id="t-42")

    assert client.calls[0]["source_info"] == {"thread_id": "t-42"}


def test_no_source_info_without_a_thread(recorder):
    """An empty conversation id is worse than none at all."""
    test_client, client = recorder

    post(test_client, feedback_score=1, thread_id=None)

    assert client.calls[0]["source_info"] is None


def test_the_response_is_a_request_id_and_a_status(recorder):
    test_client, _ = recorder

    body = post(test_client, feedback_score=1).json()

    assert set(body) == {"request_id", "status"}
    assert body["status"] == "recorded"
    assert len(body["request_id"]) > 8


def test_both_kinds_still_answer_with_the_same_contract(recorder):
    test_client, _ = recorder

    assert post(test_client, feedback_score=1, feedback_text="ok").json()["status"] == "recorded"
    assert post(test_client, feedback_text="only words").json()["status"] == "recorded"


def test_feedback_without_a_trace_is_refused(recorder):
    """There is nothing to attach it to, and pretending otherwise loses the vote."""
    test_client, client = recorder

    response = test_client.post("/feedback/", json={"trace_id": "", "feedback_score": 1})

    assert response.status_code == 400
    assert client.calls == []
    assert "trace" in response.json()["detail"]["error"].lower()


def test_feedback_with_nothing_in_it_is_refused(recorder):
    test_client, client = recorder

    response = post(test_client, feedback_text="   ")

    assert response.status_code == 400
    assert client.calls == []


def test_feedback_is_refused_when_tracing_is_not_configured(monkeypatch):
    """A 503 says the vote was not saved, which is the truth."""
    import api.api.endpoints as endpoints
    from fastapi.testclient import TestClient

    from api.app import app

    monkeypatch.setattr(endpoints, "get_trace_client", lambda: None)
    with TestClient(app) as test_client:
        response = post(test_client, feedback_score=1)

    assert response.status_code == 503
    assert "not recorded" not in response.json()["detail"]["error"]
    assert "configured" in response.json()["detail"]["error"]


def test_a_refused_write_is_reported_rather_than_swallowed(monkeypatch, caplog):
    """The most important case: a failed write must not look like a success."""
    import api.api.endpoints as endpoints
    from fastapi.testclient import TestClient

    from api.app import app

    class Refusing:
        def create_feedback(self, **kwargs):
            raise RuntimeError("tenant exceeded usage limits")

    monkeypatch.setattr(endpoints, "get_trace_client", lambda: Refusing())
    with TestClient(app) as test_client:
        response = post(test_client, feedback_score=1)

    assert response.status_code == 502
    assert "could not be recorded" in response.json()["detail"]["error"]


def test_a_vote_saved_before_a_failed_comment_is_not_rolled_back(monkeypatch):
    """The vote did land, so it is reported as saved and the failure is still raised."""
    import api.api.endpoints as endpoints
    from fastapi.testclient import TestClient

    from api.app import app

    written = []

    class HalfRefusing:
        def create_feedback(self, **kwargs):
            if kwargs["key"] == "comment":
                raise RuntimeError("quota")
            written.append(kwargs)

    monkeypatch.setattr(endpoints, "get_trace_client", lambda: HalfRefusing())
    with TestClient(app) as test_client:
        response = post(test_client, feedback_score=1, feedback_text="nice")

    assert response.status_code == 502
    assert written and written[0]["key"] == "thumbs", "the vote that landed is still recorded"


PRODUCTS = [
    {
        "id": "B0TEST0001",
        "image_url": "https://example.test/one.jpg",
        "price": 19.9,
        "description": "Portable washer.",
        "rating": 4.2,
    },
]


@pytest.fixture
def answering_client(monkeypatch):
    import api.agents.agent as shopping
    import api.agents.graph as graph_module
    import api.agents.retrieval as retrieval
    import api.agents.tools as tools
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver

    import fake_openrouter as fake

    monkeypatch.setattr(
        graph_module,
        "intent_router_node",
        lambda state: {"question_relevancy": True, "answer": ""},
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
    monkeypatch.setattr(
        retrieval, "fetch_product_payloads", lambda ids: {p["id"]: {"image": p["image_url"]} for p in PRODUCTS}
    )
    monkeypatch.setattr(shopping, "hydrate_used_context", lambda refs: list(PRODUCTS))
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda: InMemorySaver())
    shopping.get_agent.cache_clear()
    monkeypatch.setattr(graph_module, "graph", graph_module.build_graph())

    from api.app import app

    return TestClient(app, raise_server_exceptions=False), shopping


def test_the_trace_id_is_returned_with_the_answer(monkeypatch, answering_client):
    """A vote cannot be filed without the id of the trace that produced the answer."""
    test_client, shopping = answering_client

    monkeypatch.setattr(shopping, "current_trace_id", lambda: TRACE)

    response = test_client.post("/agent/", json={"query": "washing machines"})

    assert response.status_code == 200, response.text
    assert response.json()["trace_id"] == TRACE


def test_the_trace_id_is_returned_on_the_stream_too(monkeypatch, answering_client):
    """The streaming client votes on the same answer, so it needs the same id."""
    import json as json_module

    test_client, shopping = answering_client

    monkeypatch.setattr(shopping, "current_trace_id", lambda: TRACE)

    with test_client.stream("POST", "/agent/stream", json={"query": "washing machines"}) as response:
        events = [
            json_module.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    result = next(event for event in events if event["type"] == "result")
    assert result["payload"]["trace_id"] == TRACE


def test_no_trace_id_is_an_empty_string_not_a_missing_field(monkeypatch, answering_client):
    """Tracing can be off, and the client should not have to handle a missing key."""
    test_client, shopping = answering_client

    monkeypatch.setattr(shopping, "current_trace_id", lambda: "")

    body = test_client.post("/agent/", json={"query": "washing machines"}).json()

    assert body["trace_id"] == ""


def test_an_off_topic_turn_still_returns_the_field(monkeypatch):
    """That turn never runs the agent, but the response shape must not change."""
    import api.agents.graph as graph_module
    from fastapi.testclient import TestClient

    from api.app import app

    monkeypatch.setattr(
        graph_module,
        "intent_router_node",
        lambda state: {"question_relevancy": False, "answer": "Please ask about the products."},
    )
    monkeypatch.setattr(graph_module, "graph", graph_module.build_graph())
    test_client = TestClient(app, raise_server_exceptions=False)

    response = test_client.post("/agent/", json={"query": "who won the 1998 world cup"})

    assert response.status_code == 200, response.text
    assert response.json()["trace_id"] == ""


def test_the_trace_id_comes_from_the_root_of_the_trace(monkeypatch):
    """A span's own id is not the trace's, and feedback needs the trace."""
    from langsmith import tracing_context

    from api.core import tracing as tracing_module

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()

    seen = {}

    def body():
        seen["id"] = tracing_module.current_trace_id()

    try:

        @tracing_module.traced(name="outer_span")
        def outer():
            body()

        with tracing_context(enabled=True):
            outer()
    finally:
        tracing_module.get_trace_client.cache_clear()

    assert seen["id"], "a trace id must be available inside a span"
    assert seen["id"].count("-") == 4, f"not a uuid: {seen['id']}"


def test_no_trace_id_outside_a_span():
    """Callers must not have to check whether tracing is on before asking."""
    from api.core.tracing import current_trace_id

    assert current_trace_id() == ""
