import inspect

import pytest
from instructor.core.exceptions import ResponseParsingError

from api.agents import agent as agent_module
from api.agents import tools as tools_module
from api.agents.tools import retrieve_data_tool, retrieve_reviews_data


class _Point:
    def __init__(self, asin: str, text: str):
        self.payload = {"parent_asin": asin, "preprocessed_data": text}
        self.score = 0.5


class _Recorder:
    """Stands in for the shared Qdrant client and remembers every call."""

    def __init__(self, points=None, error=None):
        self.calls: list[dict] = []
        self._points = points or []
        self._error = error

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return type("Result", (), {"points": self._points})()


@pytest.fixture
def qdrant(monkeypatch):
    """Replace the shared client, and prove the tool reuses it."""

    def install(points=None, error=None):
        recorder = _Recorder(points, error)
        monkeypatch.setattr(tools_module, "qdrant_client", recorder)
        return recorder

    return install


@pytest.fixture(autouse=True)
def stub_embedding(monkeypatch):
    """No Cohere call from these tests."""
    monkeypatch.setattr(
        tools_module, "generate_embedding", lambda text: [0.0] * 8, raising=True
    )


def call(item_list, query="is it reliable?", k=None):
    args = {"query": query, "item_list": item_list}
    if k is not None:
        args["k"] = k
    return retrieve_reviews_data.invoke(args)


def test_a_valid_item_list_is_retrieved(qdrant):
    recorder = qdrant([_Point("B0AAA00001", "quiet and compact")])
    out = call(["B0AAA00001"])
    assert recorder.calls, "the tool must actually query"
    assert "quiet and compact" in out
    assert recorder.calls[0]["collection_name"] == tools_module.REVIEWS_COLLECTION


def test_several_asins_become_one_match_any(qdrant):
    from qdrant_client.models import MatchAny

    recorder = qdrant([])
    call(["B0AAA00001", "B0BBB00002", "B0CCC00003"])
    condition = recorder.calls[0]["prefetch"][0].filter.must[0]
    assert isinstance(condition.match, MatchAny)
    assert list(condition.match.any) == ["B0AAA00001", "B0BBB00002", "B0CCC00003"]


def test_the_parent_asin_filter_is_on_both_prefetches(qdrant):
    """One filter per branch, or a product could leak in through the sparse side."""
    recorder = qdrant([])
    call(["B0AAA00001"])
    prefetch = recorder.calls[0]["prefetch"]

    assert len(prefetch) == 2, "one dense branch and one sparse branch"
    for branch in prefetch:
        condition = branch.filter.must[0]
        assert condition.key == "parent_asin"
        assert list(condition.match.any) == ["B0AAA00001"]


def test_the_dense_and_sparse_branches_use_the_named_vectors(qdrant):
    recorder = qdrant([])
    call(["B0AAA00001"])
    dense, sparse = recorder.calls[0]["prefetch"]
    assert dense.using == "text-embedding-3-small"
    assert sparse.using == "bm25"
    assert recorder.calls[0]["query"].fusion == "rrf"


def test_k_defaults_to_five(qdrant):
    recorder = qdrant([_Point(f"B0AAA0000{i}", "t") for i in range(9)])
    call(["B0AAA00001"])
    assert recorder.calls[0]["limit"] == 5


def test_k_is_capped_so_one_turn_stays_affordable(qdrant):
    recorder = qdrant([])
    call(["B0AAA00001"], k=500)
    assert recorder.calls[0]["limit"] == 10


def test_k_is_honoured_below_the_cap(qdrant):
    recorder = qdrant([])
    call(["B0AAA00001"], k=3)
    assert recorder.calls[0]["limit"] == 3


def test_an_empty_item_list_asks_for_ids_instead_of_querying(qdrant):
    recorder = qdrant()
    out = call([])
    assert not recorder.calls, "no ids means no query"
    assert "product ids" in out.lower()
    assert "error" not in out.lower()


def test_an_unknown_asin_says_so_without_raising(qdrant):
    qdrant([])
    out = call(["B0NOPE0000"])
    assert "no buyer reviews" in out.lower()


def test_a_qdrant_failure_falls_back_to_the_products(qdrant):
    qdrant(error=RuntimeError("connection refused"))
    out = call(["B0AAA00001"])
    assert "answer from the products alone" in out.lower()
    assert "unavailable" in out.lower()


def test_each_review_keeps_its_own_asin(qdrant):
    qdrant([_Point("B0AAA00001", "first review"), _Point("B0BBB00002", "second review")])
    out = call(["B0AAA00001", "B0BBB00002"], k=2)
    assert "[B0AAA00001] customer review: first review" in out
    assert "[B0BBB00002] customer review: second review" in out


def test_a_point_without_a_payload_does_not_crash_the_tool(qdrant):
    bare = type("P", (), {"payload": None, "score": 0.1})()
    qdrant([bare])
    out = call(["B0AAA00001"])
    assert "None" in out or "customer review" in out


def test_the_tool_uses_the_shared_module_level_client(qdrant):
    """The double is installed on the module, so using it proves no new client."""
    recorder = qdrant([_Point("B0AAA00001", "ok")])
    call(["B0AAA00001"])
    assert len(recorder.calls) == 1, "the call went through the shared client"


def test_no_qdrant_client_is_constructed_in_the_tools_module():
    source = inspect.getsource(tools_module)
    assert "QdrantClient(" not in source, "the tools must reuse retrieval's client"


def test_retrieval_owns_the_only_runtime_client():
    from api.agents import retrieval

    source = inspect.getsource(retrieval)
    assert "qdrant_client = QdrantClient(" in source
    assert source.count("QdrantClient(") == 1, "exactly one client, built once at import"


def test_the_etl_may_have_its_own_clients():
    """It is a separate batch process, so it is allowed its own connection."""
    from api.etl import build_reviews_collection

    source = inspect.getsource(build_reviews_collection)
    assert "QdrantClient(" in source


def test_both_tools_are_bound_to_the_agent():
    agent = agent_module.get_agent()
    assert retrieve_data_tool.name == "retrieve_data_tool"
    assert retrieve_reviews_data.name == "retrieve_reviews_data"
    assert "tools" in agent.nodes and "model" in agent.nodes


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.seen: list[dict] = []

    def create(self, **kwargs):
        self.seen.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeInstructor:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.fixture
def router(monkeypatch):
    """Drive the intent node with a controllable instructor and chat model."""

    def install(instructor_response=None, instructor_error=None, text_reply=None):
        completions = _FakeCompletions(instructor_response, instructor_error)
        monkeypatch.setattr(
            agent_module, "get_instructor_client", lambda: _FakeInstructor(completions)
        )

        class _Msg:
            def __init__(self, content):
                self.content = content

        class _Model:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                return _Msg(text_reply)

        model = _Model()
        monkeypatch.setattr(agent_module, "get_chat_model", lambda: model)
        return completions, model

    return install


def test_a_tool_calling_model_is_read_by_instructor(router):
    completions, model = router(instructor_response=agent_module.IntentRouterResponse(
        question_relevancy=True, answer=""
    ))
    out = agent_module.intent_router_node({"initial_query": "washing machines?"})
    assert out == {"question_relevancy": True, "answer": ""}
    assert len(completions.seen) == 1
    assert model.calls == 0, "a tool-calling model must not cost a second call"


def test_a_plain_json_model_falls_back_and_costs_one_more_call(router):
    completions, model = router(
        instructor_error=ResponseParsingError("no tool call in the response"),
        text_reply='{"question_relevancy": true, "answer": ""}',
    )
    out = agent_module.intent_router_node({"initial_query": "washing machines?"})
    assert out == {"question_relevancy": True, "answer": ""}
    assert len(completions.seen) == 1
    assert model.calls == 1, "the fallback reads the same model's text"


def test_surrounding_prose_does_not_hide_the_json(router):
    router(
        instructor_error=ResponseParsingError("no tool call"),
        text_reply='Sure, here you go: {"question_relevancy": false, "answer": "no"} Hope that helps.',
    )
    out = agent_module.intent_router_node({"initial_query": "weather?"})
    assert out == {"question_relevancy": False, "answer": "no"}


def test_unreadable_output_is_treated_as_relevant(router):
    router(instructor_error=ResponseParsingError("no tool call"), text_reply="I am not JSON")
    out = agent_module.intent_router_node({"initial_query": "anything?"})
    assert out["question_relevancy"] is True, "let the agent decide rather than refuse"


def test_malformed_json_is_treated_as_relevant(router):
    router(instructor_error=ResponseParsingError("no tool call"), text_reply='{"question_relevancy": ')
    out = agent_module.intent_router_node({"initial_query": "anything?"})
    assert out["question_relevancy"] is True


def test_wrong_field_types_are_treated_as_relevant(router):
    router(
        instructor_error=ResponseParsingError("no tool call"),
        text_reply='{"question_relevancy": "maybe", "answer": 12}',
    )
    out = agent_module.intent_router_node({"initial_query": "anything?"})
    assert out["question_relevancy"] is True, "a schema violation must not answer the question"


def test_a_missing_api_key_is_not_hidden_as_relevant(router, monkeypatch):
    """A configuration failure must surface, not become 'relevant'."""
    completions = _FakeCompletions(error=RuntimeError("No LLM API key found."))
    monkeypatch.setattr(
        agent_module, "get_instructor_client", lambda: _FakeInstructor(completions)
    )
    with pytest.raises(RuntimeError, match="API key"):
        agent_module.intent_router_node({"initial_query": "anything?"})


def test_a_programming_error_is_not_hidden_as_relevant(router, monkeypatch):
    completions = _FakeCompletions(error=NameError("undefined_name"))
    monkeypatch.setattr(
        agent_module, "get_instructor_client", lambda: _FakeInstructor(completions)
    )
    with pytest.raises(NameError):
        agent_module.intent_router_node({"initial_query": "anything?"})


def test_keyboard_interrupt_is_never_swallowed(router, monkeypatch):
    completions = _FakeCompletions(error=KeyboardInterrupt())
    monkeypatch.setattr(
        agent_module, "get_instructor_client", lambda: _FakeInstructor(completions)
    )
    with pytest.raises(KeyboardInterrupt):
        agent_module.intent_router_node({"initial_query": "anything?"})


@pytest.mark.parametrize(
    "raw, relevant, answer",
    [
        ('{"question_relevancy": true, "answer": ""}', True, ""),
        ('{"question_relevancy": false, "answer": "off topic"}', False, "off topic"),
        ('pre {"question_relevancy": false, "answer": "no"} post', False, "no"),
        ("no json at all", True, ""),
        ("", True, ""),
        ('{"question_relevancy": ', True, ""),
        ('{"question_relevancy": "yes", "answer": null}', True, ""),
        ('{"answer": "missing the flag"}', True, ""),
    ],
)
def test_the_parser_reads_what_it_can(raw, relevant, answer):
    parsed = agent_module._parse_intent(raw)
    assert parsed.question_relevancy is relevant
    assert parsed.answer == answer


def test_the_router_prompt_mentions_buyer_experience():
    """Otherwise 'any complaints' is routed away and the tool is unreachable."""
    from api.agents.prompts import prompt

    text = prompt("intent_router").lower()
    assert "reviews" in text or "complaints" in text
    assert "retrieve_reviews_data" in prompt("shopping_agent")
