import logging
import threading

import pytest

from api.agents import llm as llm_module
from api.core import tracing as tracing_module
from api.core.settings import get_settings

URL = "https://api.groq.com/openai/v1/chat/completions"


def rate_limited(body: dict):
    import httpx
    from openai import RateLimitError

    response = httpx.Response(429, json=body, request=httpx.Request("POST", URL))
    return RateLimitError("Error code: 429", response=response, body=body)


PROVIDER_LIMIT_BODY = {
    "error": {
        "message": "Rate limit reached for model `x`",
        "metadata": {
            "provider_name": "Poolside",
            "limit_source": "upstream_provider_shared_pool",
        },
    }
}
ACCOUNT_QUOTA_BODY = {
    "error": {
        "message": "Rate limit exceeded: free-models-per-day",
        "metadata": {"limit_source": "openrouter_free_tier_daily"},
    }
}


def test_the_configured_client_retry_count_is_zero():
    assert get_settings().llm.max_retries == 0


def test_the_chat_model_is_built_with_no_client_retries():
    model = llm_module.get_chat_model()
    assert model.max_retries == 0, "ChatOpenAI must not retry on its own"
    assert model.client._client.max_retries == 0


def test_the_instructor_client_is_built_with_no_client_retries(monkeypatch):
    """The intent router uses a raw OpenAI client, whose own default is 2."""
    import instructor
    import openai

    seen = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        instructor, "from_openai", lambda client, mode=None: {"max_retries": seen.get("max_retries")}
    )
    llm_module.get_instructor_client.cache_clear()
    try:
        assert llm_module.get_instructor_client()["max_retries"] == 0
    finally:
        llm_module.get_instructor_client.cache_clear()


def test_a_provider_429_reaches_the_caller_unretried(monkeypatch):
    from openai import RateLimitError

    attempts = {"n": 0}
    error = rate_limited(PROVIDER_LIMIT_BODY)

    def refuse(*args, **kwargs):
        attempts["n"] += 1
        raise error

    model = llm_module.get_chat_model()
    monkeypatch.setattr(model, "_generate", refuse)
    monkeypatch.setattr(model, "_stream", lambda *a, **k: refuse())

    with pytest.raises(RateLimitError) as raised:
        model.invoke("hello")

    assert attempts["n"] == 1, "the refusal reached us on the first response"
    assert raised.value is error


def test_the_real_status_reaches_the_classifier_unretried():
    """429 stays 429 all the way in, with the body intact."""
    from api.agents.errors import body_of, status_of

    error = rate_limited(PROVIDER_LIMIT_BODY)
    assert status_of(error) == 429
    assert "upstream_provider_shared_pool" in body_of(error), (
        "the body is what tells a provider limit from an account quota"
    )


def test_a_provider_limit_body_differs_from_an_account_quota_body():
    from api.agents.errors import body_of

    provider = body_of(rate_limited(PROVIDER_LIMIT_BODY))
    account = body_of(rate_limited(ACCOUNT_QUOTA_BODY))

    assert "upstream_provider_shared_pool" in provider
    assert "upstream_provider_shared_pool" not in account
    assert "openrouter_free_tier_daily" in account
    assert provider != account


def test_a_429_is_sanitised_but_still_reported_as_429():
    """The client's status reaches the endpoint, its body does not."""
    import json

    from api.api.endpoints import _as_http_error

    http_error = _as_http_error(rate_limited(PROVIDER_LIMIT_BODY))
    assert http_error.status_code == 429, "the provider's status is passed through"

    body = json.dumps(http_error.detail)
    assert "upstream_provider_shared_pool" not in body
    assert "openrouter" not in body.lower() or "provider" in body.lower()


def test_tracing_is_off_when_it_is_not_switched_on(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    tracing_module.get_trace_client.cache_clear()
    try:
        assert tracing_module.get_trace_client() is None
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_tracing_is_off_without_a_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    tracing_module.get_trace_client.cache_clear()
    try:
        assert tracing_module.get_trace_client() is None
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_the_client_is_built_once_and_shared(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()
    try:
        first = tracing_module.get_trace_client()
        assert first is tracing_module.get_trace_client()
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_the_client_does_not_retry_a_refused_upload(monkeypatch):
    """A 429 will not succeed on a second attempt, so it is not retried."""
    from langsmith.utils import LangSmithRetry

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()
    try:
        client = tracing_module.get_trace_client()
        policy = getattr(client, "retry_config", None)
        assert isinstance(policy, LangSmithRetry)
        assert not policy.status, f"429 would be retried: {policy.status}"
        assert not policy.status_forcelist, "no status should be retried"
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_both_tracing_paths_use_the_configured_client(monkeypatch):
    import langsmith.run_trees as run_trees
    from langchain_core.tracers.langchain import LangChainTracer

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    monkeypatch.setattr(run_trees, "_CLIENT", None)
    tracing_module.get_trace_client.cache_clear()
    try:
        client = tracing_module.get_trace_client()
        assert run_trees.get_cached_client() is client, "the cached default was not seeded"
        assert LangChainTracer(project_name="p").client is client, (
            "the environment-driven tracer built its own client"
        )
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_a_refused_quota_stops_being_logged_as_a_warning(caplog):
    import logging

    filt = tracing_module._DemoteQuotaNoise()

    record = logging.LogRecord(
        name="langsmith.client",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=(
            "Failed to send compressed multipart ingest: "
            "langsmith.utils.LangSmithRateLimitError: Rate limit exceeded for "
            "https://api.smith.langchain.com/runs/multipart, "
            "'{\"error\":\"Too many requests: tenant exceeded usage limits: "
            "Monthly unique traces usage limit exceeded\"}'"
        ),
        args=(),
        exc_info=None,
    )
    assert filt.filter(record) is True, "the record must be kept, only demoted"
    assert record.levelno == logging.DEBUG


@pytest.mark.parametrize(
    "message",
    [
        "Failed to send compressed multipart ingest: ConnectionError",
        "Failed to send compressed multipart ingest: ReadTimeout",
        "Failed to send compressed multipart ingest: 500 Server Error",
    ],
)
def test_a_genuine_transport_failure_is_still_a_warning(message):
    """Only the quota is quiet. Losing a connection or timing out still matters."""
    import logging

    filt = tracing_module._DemoteQuotaNoise()
    record = logging.LogRecord(
        name="langsmith.client",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    assert filt.filter(record) is True
    assert record.levelno == logging.WARNING


def test_the_filter_is_installed_once_on_the_sdk_logger():
    import logging

    sdk_logger = logging.getLogger("langsmith.client")
    before = [f for f in sdk_logger.filters if isinstance(f, tracing_module._DemoteQuotaNoise)]
    tracing_module._quiet_exhausted_quota()
    tracing_module._quiet_exhausted_quota()
    after = [f for f in sdk_logger.filters if isinstance(f, tracing_module._DemoteQuotaNoise)]
    assert len(after) == len(before), "installing repeatedly must not stack filters"
    assert len(after) >= 1, "the filter has to be installed at least once"


def test_the_seeded_client_reaches_a_worker_thread(monkeypatch):
    """The SSE endpoint runs the agent on a thread; tracing must be quiet there too."""
    import langsmith.run_trees as run_trees

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()
    seen = {}

    def probe():
        seen["client"] = run_trees.get_cached_client()

    try:
        client = tracing_module.get_trace_client()
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        assert seen.get("client") is client
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_an_ingest_failure_is_logged_at_debug_not_warning(caplog):
    """A refused quota must not fill the log the way a warning did."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="api.core.tracing"):
        result = tracing_module._report_ingest_failure(
            RuntimeError("429 Monthly unique traces usage limit exceeded")
        )

    assert result is None, "the callback must not re-raise"
    records = [r for r in caplog.records if "LangSmith" in r.getMessage()]
    assert records, "the failure is still recorded, so it is not invisible"
    assert all(r.levelno == logging.DEBUG for r in records), (
        f"ingest failures must not be warnings: {[r.levelname for r in records]}"
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_span_returns_its_result_even_when_the_upload_fails(monkeypatch):
    from langsmith import Client
    from langsmith.utils import LangSmithRetry

    client = Client(
        api_key="lsv2_fake",
        auto_batch_tracing=False,
        retry_config=LangSmithRetry(status=0),
        tracing_error_callback=tracing_module._report_ingest_failure,
    )

    def refuse(*args, **kwargs):
        raise OSError("ingest refused: quota exceeded")

    monkeypatch.setattr(client.session, "request", refuse)

    monkeypatch.setattr(tracing_module, "get_trace_client", lambda: client)

    @tracing_module.traced(name="local_probe")
    def work(value):
        return value * 2

    assert work(21) == 42, "the work completed even though the upload failed"


def test_a_span_works_on_a_worker_thread(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()
    try:
        client = tracing_module.get_trace_client()
        seen = {}

        @tracing_module.traced(name="thread_probe")
        def work():
            seen["ran"] = True
            return "done"

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

        assert seen.get("ran") is True
        assert client is not None
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_real_application_errors_are_not_swallowed_by_tracing(monkeypatch):
    """A failure inside the work must still propagate, loudly."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()

    @tracing_module.traced(name="failing_probe")
    def work():
        raise ValueError("the work itself failed")

    try:
        with pytest.raises(ValueError, match="the work itself failed"):
            work()
    finally:
        tracing_module.get_trace_client.cache_clear()


def test_every_span_is_still_there_and_uses_the_configured_tracer():
    """The instrumentation is unchanged; only the client differs."""
    import inspect

    from api.agents import agent, rag, retrieval

    for module in (agent, rag, retrieval):
        source = inspect.getsource(module)
        assert "@traced(" in source, f"{module.__name__} lost its spans"
        assert "from langsmith import traceable" not in source, (
            f"{module.__name__} still uses an unconfigured tracer"
        )


def _run_tree_of(func, *args):
    from langsmith import tracing_context
    from langsmith.run_trees import RunTree

    captured = []

    def capture(run_tree):
        captured.append(run_tree)

    with tracing_context(enabled=True):
        func(*args, langsmith_extra={"on_end": capture})

    assert captured, "the span never reached on_end, so nothing was measured"
    assert isinstance(captured[0], RunTree)
    return captured[0]


def test_token_counts_reach_the_span_metadata(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()

    @tracing_module.traced(name="priced_span", run_type="llm")
    def work():
        tracing_module.trace_usage(input_tokens=118, output_tokens=42, total_tokens=160)

    try:
        run_tree = _run_tree_of(work)
    finally:
        tracing_module.get_trace_client.cache_clear()

    metadata = (run_tree.extra or {}).get("metadata") or {}
    assert metadata.get("input_tokens") == 118
    assert metadata.get("output_tokens") == 42
    assert metadata.get("total_tokens") == 160


def test_recording_usage_outside_a_span_is_harmless():
    """Calling code must not need to know whether tracing is switched on."""
    tracing_module.trace_usage(input_tokens=5, output_tokens=5, total_tokens=10)


def test_a_span_with_no_usage_gets_no_usage_fields(monkeypatch):
    """Nothing measured must not be reported as zero."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()

    @tracing_module.traced(name="unpriced_span")
    def work():
        return "done"

    try:
        run_tree = _run_tree_of(work)
    finally:
        tracing_module.get_trace_client.cache_clear()

    metadata = (run_tree.extra or {}).get("metadata") or {}
    assert "input_tokens" not in metadata
    assert "output_tokens" not in metadata
    assert "total_tokens" not in metadata


def test_each_call_keeps_its_own_counts(monkeypatch):
    """Two turns running one after another must not share a token count."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake")
    tracing_module.get_trace_client.cache_clear()

    @tracing_module.traced(name="counting_span")
    def work(total):
        tracing_module.trace_usage(input_tokens=total, output_tokens=0, total_tokens=total)

    try:
        first = _run_tree_of(work, 7)
        second = _run_tree_of(work, 99)
    finally:
        tracing_module.get_trace_client.cache_clear()

    assert ((first.extra or {}).get("metadata") or {}).get("input_tokens") == 7
    assert ((second.extra or {}).get("metadata") or {}).get("input_tokens") == 99


def test_the_node_still_looks_like_a_node_to_langgraph():
    import inspect

    from api.agents.agent import agent_node

    assert agent_node.__name__ == "agent_node"
    assert list(inspect.signature(agent_node).parameters) == ["state"]


def test_a_turn_is_costed_across_every_call_it_made():
    from langchain_core.messages import AIMessage

    from api.agents.agent import _turn_usage

    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_data", "args": {}, "id": "1"}],
            usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
        ),
        AIMessage(
            content="Here are the machines [B0ABC12345].",
            usage_metadata={"input_tokens": 1200, "output_tokens": 180, "total_tokens": 1380},
        ),
    ]

    assert _turn_usage(messages) == {
        "input_tokens": 2100,
        "output_tokens": 220,
        "total_tokens": 2320,
    }


def test_a_model_that_reports_no_usage_is_not_reported_as_free():
    """No usage must mean no numbers, not a confident zero."""
    from langchain_core.messages import AIMessage, HumanMessage

    from api.agents.agent import _turn_usage

    assert _turn_usage([AIMessage(content="no usage here")]) == {}
    assert _turn_usage([HumanMessage(content="a question")]) == {}
    assert _turn_usage([]) == {}


def test_a_partial_usage_report_is_still_summed():
    from api.agents.agent import _turn_usage

    class Reported:
        def __init__(self, usage_metadata):
            self.usage_metadata = usage_metadata

    messages = [
        Reported({"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}),
        Reported({"input_tokens": 50, "output_tokens": None}),
        Reported({"input_tokens": 1.5, "output_tokens": "7", "total_tokens": True}),
    ]

    totals = _turn_usage(messages)
    assert totals["input_tokens"] == 150
    assert totals["output_tokens"] == 10, "a missing value must not become a zero"
    assert totals["total_tokens"] == 110


def test_only_this_turns_messages_are_costed():
    """Conversation history must not be billed to the turn that followed it."""
    from langchain_core.messages import AIMessage, HumanMessage

    from api.agents.agent import _turn_usage

    history = [
        HumanMessage(content="earlier question"),
        AIMessage(
            content="earlier answer",
            usage_metadata={"input_tokens": 5000, "output_tokens": 500, "total_tokens": 5500},
        ),
    ]
    this_turn = [
        AIMessage(
            content="new answer",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )
    ]

    assert _turn_usage(this_turn)["input_tokens"] == 100
    assert _turn_usage([*history, *this_turn])["input_tokens"] == 5100, (
        "the slice the node passes must be the one that excludes history"
    )
