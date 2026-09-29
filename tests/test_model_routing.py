"""Routing across the OpenRouter model pool.

Every test mocks OpenRouter, so none of them depend on a free model being
available. The rule under test: rotate only for availability failures, and stop
for everything else.
"""

import pytest

import fake_openrouter as fake
from api.agents.model_router import RoutedChatModel
from api.agents.errors import describe, is_account_quota, is_fallback_worthy

POOL = ["model/a", "model/b", "model/c"]


def routed(pool=POOL) -> RoutedChatModel:
    return RoutedChatModel(models=list(pool), options={"api_key": "test"})


def text_of(message) -> str:
    """`.invoke` hands back the AIMessage, not the ChatResult."""
    return message.content


# --- the primary model is the one that answers -----------------------------


def test_primary_model_success(monkeypatch):
    calls = fake.install(monkeypatch, {"model/a": []})
    result = routed().invoke("which washing machines do you have?")
    assert text_of(result) == "answer"
    assert calls == ["model/a"], "the pool must not be walked when the first model works"


# --- availability failures rotate ------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(fake.provider_limit_error, id="429 provider rate limit"),
        pytest.param(lambda: fake.server_error(503), id="503 provider down"),
        pytest.param(lambda: fake.server_error(500), id="500"),
        pytest.param(fake.timeout_error, id="timeout"),
        pytest.param(ConnectionError, id="network failure"),
    ],
)
def test_availability_failure_falls_back(monkeypatch, failure):
    """A 429 that belongs to a provider, a 5xx, a timeout and a dropped
    connection are all things the next model might survive."""
    calls = fake.install(monkeypatch, {"model/a": [failure()]})
    result = routed().invoke("hello")
    assert text_of(result) == "answer"
    assert calls == ["model/a", "model/b"]


def test_multiple_failures_walk_the_pool(monkeypatch):
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(503)], "model/b": [fake.timeout_error()]},
    )
    result = routed().invoke("hello")
    assert text_of(result) == "answer"
    assert calls == ["model/a", "model/b", "model/c"], "each failure moves one step on"


def test_all_models_failing_surfaces_the_last_real_error(monkeypatch):
    calls = fake.install(
        monkeypatch,
        {name: [fake.server_error(503)] for name in POOL},
    )
    with pytest.raises(Exception) as caught:
        routed().invoke("hello")
    assert calls == POOL, "the whole pool is tried once, in order"
    assert "503" in str(describe(caught.value)), "the real status is reported"


# --- failures that must not rotate ----------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(fake.bad_request_error, id="400 our own bad request"),
        pytest.param(fake.auth_error, id="401 invalid key"),
        pytest.param(lambda: fake.server_error(404), id="404 model gone"),
        pytest.param(lambda: fake.server_error(422), id="422 unprocessable"),
    ],
)
def test_non_fallback_failures_stop_immediately(monkeypatch, failure):
    calls = fake.install(monkeypatch, {"model/a": [failure()]})
    with pytest.raises(Exception):
        routed().invoke("hello")
    assert calls == ["model/a"], "rotating would hide the real cause, not fix it"


# --- account quota is not a routing problem --------------------------------


def test_account_quota_stops_after_one_model(monkeypatch):
    """The daily cap applies to the key, so no other model can answer."""
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    with pytest.raises(Exception) as caught:
        routed().invoke("hello")
    assert calls == ["model/a"], "quota exhaustion must not walk the pool"
    message = str(describe(caught.value))
    assert "free-models-per-day" in message, "the provider's own wording is surfaced"
    assert "quota exhausted" not in message.lower(), "no wording invented by us"


def test_account_quota_is_not_fallback_worthy():
    assert is_account_quota(fake.daily_cap_error())
    assert not is_fallback_worthy(fake.daily_cap_error())


def test_payment_required_is_account_quota():
    assert is_account_quota(fake.payment_error())
    assert not is_fallback_worthy(fake.payment_error())


def test_provider_rate_limit_is_fallback_worthy():
    assert not is_account_quota(fake.provider_limit_error())
    assert is_fallback_worthy(fake.provider_limit_error())


def test_unmarked_429_is_treated_as_model_specific():
    """A 429 with no account marker is the model's own limit, so it rotates."""
    error = fake.FakeOpenRouterError(429, {"error": {"message": "Rate limit exceeded"}})
    assert not is_account_quota(error)
    assert is_fallback_worthy(error)


def test_quota_error_trips_the_cooldown(monkeypatch):
    """After a quota refusal, later turns cost no request at all."""
    from api.agents import errors as provider_errors

    fake.install(monkeypatch, {})
    provider_errors.trip(fake.daily_cap_error(), 60)
    held = provider_errors.hold_error()
    assert held is not None
    assert "free-models-per-day" in str(held), "the provider's own wording is replayed"


# --- streaming -------------------------------------------------------------


def test_the_router_exposes_real_generators_for_streaming():
    """`_stream` and `_astream` must be generator functions, not plain methods.

    A method that *returns* a generator instead of being one is invisible to a
    type checker and to any test that only awaits or iterates the result, and it
    breaks the framework's own streaming path: LangChain streams by driving
    these two, so anything that is not a generator silently produces nothing.
    """
    import inspect

    assert inspect.isgeneratorfunction(RoutedChatModel._stream), (
        "_stream must be a generator function"
    )
    assert inspect.isasyncgenfunction(RoutedChatModel._astream), (
        "_astream must be an async generator function"
    )


def test_streaming_reaches_the_client_through_both_paths(monkeypatch):
    """The streamed text has to arrive, not just the right calls be made."""
    import asyncio

    fake.install(monkeypatch, {}, stream_text="one two three")

    sync_text = "".join(chunk.content for chunk in routed(["model/a"]).stream("hello"))

    async def main() -> str:
        return "".join(
            [chunk.content async for chunk in routed(["model/a"]).astream("hello")]
        )

    assert sync_text.split() == ["one", "two", "three"]
    assert asyncio.run(main()).split() == ["one", "two", "three"]


def test_stream_falls_back_when_nothing_was_emitted(monkeypatch):
    """Model A fails before its first token, so model B streams instead."""
    calls = fake.install(monkeypatch, {"model/a": [fake.server_error(503)]})
    streamed = "".join(chunk.content for chunk in routed().stream("hello"))
    assert streamed.strip() == "answer"
    assert calls == ["model/a", "model/b"]


def test_stream_does_not_splice_two_models(monkeypatch):
    """Once tokens are out they belong to that model: never append another."""
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(503)]},
        tokens_before_failure=2,
        stream_text="one two three four",
    )
    emitted = []
    with pytest.raises(Exception):
        for chunk in routed().stream("hello"):
            emitted.append(chunk.content)
    assert calls == ["model/a"], "no second model after partial output"
    assert "".join(emitted) == "one two ", "the client keeps a truncated stream, not a spliced one"


def test_stream_failure_after_partial_output_propagates(monkeypatch):
    fake.install(monkeypatch, {"model/a": [fake.timeout_error()]}, tokens_before_failure=2)
    with pytest.raises(Exception):
        list(routed().stream("hello"))


def test_stream_succeeds_on_the_primary(monkeypatch):
    calls = fake.install(monkeypatch, {"model/a": []})
    streamed = "".join(chunk.content for chunk in routed().stream("hello"))
    assert streamed.strip() == "answer"
    assert calls == ["model/a"]


# --- the pool itself -------------------------------------------------------


def test_no_configured_model_is_a_clear_error():
    with pytest.raises(RuntimeError, match="OPENROUTER_MODELS"):
        RoutedChatModel(models=[], options={}).invoke("hello")


# --- the two shapes the agents actually use --------------------------------


def test_structured_output_rotates(monkeypatch):
    """The intent router calls with_structured_output, which must rotate too."""
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(503)]},
        structured_factory=fake.ScriptedStructuredModel,
        schema_result={"question_relevancy": True},
    )
    result = routed().with_structured_output(dict).invoke([{"role": "user", "content": "hi"}])
    assert result == {"question_relevancy": True}
    assert calls == ["model/a", "model/b"]


def test_structured_output_stops_on_account_quota(monkeypatch):
    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.daily_cap_error()]},
        structured_factory=fake.ScriptedStructuredModel,
        schema_result={},
    )
    with pytest.raises(Exception):
        routed().with_structured_output(dict).invoke([{"role": "user", "content": "hi"}])
    assert calls == ["model/a"]


def test_the_tool_calling_agent_rotates(monkeypatch):
    """create_agent binds tools to the routed model; the loop must still work.

    The failure lands on the second model call, after the tool has run, which is
    the case a shopping assistant actually hits when a model drops mid-turn.
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    calls = fake.install(
        monkeypatch,
        {"model/a": [fake.server_error(503), fake.server_error(503)]},
    )

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "a result"

    agent = create_agent(model=routed(), tools=[lookup])
    result = agent.invoke({"messages": [HumanMessage(content="hello")]})
    texts = [m.content for m in result["messages"] if m.type == "ai"]
    assert "answer" in texts, "the turn completed on a later model"
    assert calls.models[:2] == ["model/a", "model/b"]


def test_bind_tools_is_carried_into_every_attempt(monkeypatch):
    """A bound model must not silently lose its tools when it rotates."""
    from langchain_core.tools import tool

    seen: list[bool] = []
    fake.install(monkeypatch, {"model/a": [fake.server_error(503)]})

    original = fake.ScriptedModel.bind_tools

    def spy(self, tools, **kwargs):
        seen.append(bool(tools))
        return original(self, tools, **kwargs)

    monkeypatch.setattr(fake.ScriptedModel, "bind_tools", spy)

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "x"

    routed().bind_tools([lookup]).invoke("hello")
    assert len(seen) == 2, "tools are bound on every model the router builds"
    assert all(seen)
