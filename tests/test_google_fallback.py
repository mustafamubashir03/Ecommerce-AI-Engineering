"""The external provider fallback.

OpenRouter keeps owning the pool. Google is a different provider, asked once
after the pool cannot answer, and never asked to answer a request the pool was
told not to retry.
"""

import asyncio

import pytest

import fake_openrouter as fake
from api.agents.model_router import ProviderPlan, RoutedChatModel
from api.agents.prompts import IntentRouterResponse
from api.agents.errors import describe, trip

POOL = ["model/a", "model/b"]


class FakeGoogle:
    """A second provider. Records the calls so the policy can be asserted."""

    def __init__(self, answer="google answer", fail_with=None, tokens=0):
        self.answer = answer
        self.fail_with = fail_with
        self.tokens = tokens
        self.calls = 0

    def _record(self):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        self._record()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        self._record()
        for index in range(self.tokens or 1):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"g{index} "))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        self._record()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        self._record()
        for index in range(self.tokens or 1):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"g{index} "))


def routed(google=None, pool=POOL) -> RoutedChatModel:
    fallbacks = (
        [] if google is None else [ProviderPlan(name="google", models=["*"], build=lambda _id: google)]
    )
    return RoutedChatModel(models=list(pool), options={"api_key": "t"}, fallbacks=fallbacks)


# --- configuration ----------------------------------------------------------


def test_google_block_is_configured():
    from api.core.settings import get_settings

    google = get_settings().llm.google
    assert google.enabled, "the fallback is configured"
    assert google.api_key_env == "GOOGLE_API_KEY", "GOOGLE_API_KEY stays the credential"
    assert google.model, "a model id is configured, not guessed in code"
    assert google.model.count(".") >= 1 and "/" not in google.model


def test_google_model_id_and_key_are_not_hardcoded_in_python():
    """Both come from configuration, so switching either is not a code change."""
    import inspect

    from api.agents.providers import google

    source = inspect.getsource(google)
    assert "gemini-" not in source, "the model id belongs in config.yaml"
    assert "AIza" not in source, "no key material in source"
    # The credential is read through the env var name config.yaml names.
    assert "os.getenv(google.api_key_env)" in source
    assert "os.getenv(\"GOOGLE_API_KEY\")" not in source


def test_google_model_is_built_from_the_configured_key(monkeypatch):
    from api.agents.providers import google
    from api.core.settings import get_settings

    seen: dict = {}

    class Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(google, "AgentOwnedToolLoop", Fake)
    monkeypatch.setenv(get_settings().llm.google.api_key_env, "test-key")

    model = google.build()
    assert model is not None
    assert seen["model"] == get_settings().llm.google.model
    assert seen["google_api_key"] == "test-key"
    assert seen["timeout"] == get_settings().llm.google.timeout_seconds
    assert seen["timeout"] == 90.0, "seconds, which is what this integration expects"


def test_google_is_optional_when_the_key_is_absent(monkeypatch):
    import api.agents.llm as llm
    from api.core.settings import get_settings

    monkeypatch.delenv(get_settings().llm.google.api_key_env, raising=False)
    assert llm.build_google_model() is None, "a missing key disables the fallback, it does not break boot"


def test_google_is_optional_when_disabled(monkeypatch):
    import api.agents.llm as llm
    from api.core.settings import get_settings

    monkeypatch.setattr(get_settings().llm.google, "enabled", False)
    assert llm.build_google_model() is None


# --- when Google is used ----------------------------------------------------


def test_openrouter_success_never_calls_google(monkeypatch):
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {"model/a": []})
    result = routed(google).invoke("hello")
    assert result.content == "answer"
    assert calls == ["model/a"]
    assert google.calls == 0, "the fallback must stay untouched on the happy path"


def test_account_quota_goes_straight_to_google(monkeypatch):
    """The main case: quota is known, so no OpenRouter model is retried."""
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})

    result = routed(google).invoke("hello")
    assert result.content == "google answer"
    assert google.calls == 1
    assert calls == ["model/a"], "the walk stops at the quota, it does not try model/b"


def test_a_held_quota_sends_the_next_request_straight_to_google(monkeypatch):
    """The cooldown must not be a reason to keep asking OpenRouter."""
    trip(fake.daily_cap_error(), 300)
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {"model/a": []})

    result = routed(google).invoke("hello")
    assert result.content == "google answer"
    assert google.calls == 1
    assert calls == [], "zero OpenRouter requests while the cooldown is active"


def test_transient_failure_uses_the_pool_before_google(monkeypatch):
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {"model/a": [fake.server_error(503)]})
    result = routed(google).invoke("hello")
    assert result.content == "answer", "OpenRouter's own fallback ran first"
    assert calls == ["model/a", "model/b"]
    assert google.calls == 0, "an OpenRouter recovery must not disturb the other provider"


def test_google_answers_only_after_the_pool_is_exhausted(monkeypatch):
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {name: [fake.server_error(503)] for name in POOL})
    result = routed(google).invoke("hello")
    assert result.content == "google answer"
    assert calls == POOL, "the whole pool is tried first"
    assert google.calls == 1


@pytest.mark.parametrize("failure", [fake.bad_request_error, fake.auth_error, lambda: fake.server_error(422)])
def test_application_errors_are_not_handed_to_google(monkeypatch, failure):
    """A malformed request or a bad key fails the same way on Google."""
    google = FakeGoogle()
    fake.install(monkeypatch, {"model/a": [failure()]})
    with pytest.raises(Exception):
        routed(google).invoke("hello")
    assert google.calls == 0, "an application error is surfaced, not retried elsewhere"


def test_a_google_failure_is_the_final_error(monkeypatch):
    google = FakeGoogle(fail_with=fake.server_error(503))
    fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    with pytest.raises(Exception) as caught:
        routed(google).invoke("hello")
    assert google.calls == 1, "Google is asked once, never retried"
    assert "503" in str(describe(caught.value))


def test_no_recursion_google_never_reaches_openrouter_again(monkeypatch):
    """A Google failure must not send the request back to the pool."""
    google = FakeGoogle(fail_with=fake.server_error(503))
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    with pytest.raises(Exception):
        routed(google).invoke("hello")
    assert calls == ["model/a"], "OpenRouter was not asked a second time"


# --- streaming --------------------------------------------------------------


def test_google_streams_when_openrouter_produced_nothing(monkeypatch):
    google = FakeGoogle(tokens=2)
    calls = fake.install(monkeypatch, {name: [fake.server_error(503)] for name in POOL})
    streamed = "".join(chunk.content for chunk in routed(google).stream("hello"))
    assert streamed == "g0 g1 "
    assert calls == POOL
    assert google.calls == 1


def test_no_splice_when_openrouter_emitted_a_token(monkeypatch):
    """An OpenRouter prefix and a Google answer must never share a stream."""
    google = FakeGoogle(tokens=2)
    fake.install(monkeypatch, {"model/a": [fake.server_error(503)]}, tokens_before_failure=1, stream_text="openrouter partial")

    emitted = []
    with pytest.raises(Exception):
        for chunk in routed(google).stream("hello"):
            emitted.append(chunk.content)

    assert google.calls == 0, "Google must not be started after output began"
    # The first provider's first word, then nothing. A truncated stream the
    # client can retry, never OpenRouter's prefix plus Google's answer.
    assert "".join(emitted).strip() == "openrouter"


def test_google_is_not_streamed_after_an_application_error(monkeypatch):
    google = FakeGoogle(tokens=2)
    fake.install(monkeypatch, {"model/a": [fake.bad_request_error()]})
    with pytest.raises(Exception):
        list(routed(google).stream("hello"))
    assert google.calls == 0


# --- async ------------------------------------------------------------------


def test_async_path_reaches_google_without_a_sync_call(monkeypatch):
    google = FakeGoogle()
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    result = asyncio.run(routed(google).ainvoke([{"role": "user", "content": "hi"}]))
    assert result.content == "google answer"
    assert google.calls == 1
    assert calls == ["model/a"]


def test_async_stream_keeps_the_no_splice_rule(monkeypatch):
    google = FakeGoogle(tokens=2)
    fake.install(monkeypatch, {"model/a": [fake.server_error(503)]}, tokens_before_failure=1, stream_text="partial")
    emitted = []

    async def main():
        async for chunk in routed(google).astream([{"role": "user", "content": "hi"}]):
            emitted.append(chunk.content)

    with pytest.raises(Exception):
        asyncio.run(main())
    assert google.calls == 0
    assert "".join(emitted).strip() == "partial"


# --- tools ------------------------------------------------------------------


class FakeStructuredGoogle(FakeGoogle):
    """A fallback provider that also implements the structured output path."""

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class _Structured:
            def invoke(self, input, config=None, **kw):
                outer.calls += 1
                if outer.fail_with is not None:
                    raise outer.fail_with
                return IntentRouterResponse(question_relevancy=True, answer="from google")

            async def ainvoke(self, input, config=None, **kw):
                outer.calls += 1
                if outer.fail_with is not None:
                    raise outer.fail_with
                return IntentRouterResponse(question_relevancy=True, answer="from google")

        return _Structured()


def test_structured_output_reaches_the_fallback_provider(monkeypatch):
    """The intent router's call must be able to use the fallback too.

    The runnable is handed a factory, not a model, so this guards the wiring
    that a live request exposed. The schema is the real one the router uses.
    """
    from api.agents.prompts import IntentRouterResponse

    google = FakeStructuredGoogle()
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    result = routed(google).with_structured_output(IntentRouterResponse).invoke(
        [{"role": "user", "content": "hi"}]
    )
    assert result.question_relevancy is True
    assert google.calls == 1
    assert calls == ["model/a"]


def test_structured_output_async_reaches_the_fallback_provider(monkeypatch):
    from api.agents.prompts import IntentRouterResponse

    google = FakeStructuredGoogle()
    fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    result = asyncio.run(
        routed(google).with_structured_output(IntentRouterResponse).ainvoke([{"role": "user", "content": "hi"}])
    )
    assert result.question_relevancy is True
    assert google.calls == 1


def test_structured_output_happy_path_never_calls_google(monkeypatch):
    google = FakeStructuredGoogle()
    calls = fake.install(
        monkeypatch,
        {"model/a": []},
        structured_factory=fake.ScriptedStructuredModel,
        schema_result=IntentRouterResponse(question_relevancy=True, answer=""),
    )
    result = routed(google).with_structured_output(IntentRouterResponse).invoke([])
    assert result.question_relevancy is True
    assert calls == ["model/a"]
    assert google.calls == 0


def test_tools_are_bound_on_the_fallback_provider(monkeypatch):
    """The agent's tools must reach Google, not just the OpenRouter models."""
    google = FakeGoogle()
    bound = routed(google).bind_tools(["retrieve_data_tool"])
    assert [p.name for p in bound.fallbacks] == ["google"]
    # The binding is applied when the delegate is produced, and the fake records it.
    fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    result = bound.invoke("hello")
    assert result.content == "google answer"
    assert google.calls == 1


def test_the_real_tool_calling_interface_is_what_google_implements():
    """Google must satisfy the same interface the OpenRouter models do."""
    from langchain_core.language_models import BaseChatModel

    from api.agents.llm import build_google_model

    google = build_google_model()
    assert google is not None, "the fallback is built from GOOGLE_API_KEY"
    assert isinstance(google, BaseChatModel), "same interface as every other model here"
    for method in ("_generate", "_stream", "_agenerate", "_astream", "bind_tools", "with_structured_output"):
        assert hasattr(google, method), f"the agent needs {method}"


def test_automatic_function_calling_is_disabled_on_every_path(monkeypatch):
    """Google must not run the tool itself, or the graph loses the products.

    Left enabled, the SDK loops internally and returns only the final text, so
    no ToolMessage reaches `create_agent` and the citations are lost. Every call
    path has to carry the opt out.
    """
    import api.agents.llm as llm
    from api.agents.tools import retrieve_data_tool
    from langchain_core.messages import HumanMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_google_genai import ChatGoogleGenerativeAI

    google = llm.build_google_model()
    assert google is not None
    seen: list = []

    def capture(name):
        def stub(*args, **kwargs):
            seen.append((name, kwargs.get("automatic_function_calling")))
            raise RuntimeError("no network in this test")

        return stub

    # Patch the vendor class the subclass delegates to, so what is captured is
    # exactly what the subclass passes on.
    for method in ("_generate", "_agenerate", "_stream", "_astream"):
        monkeypatch.setattr(ChatGoogleGenerativeAI, method, capture(method), raising=True)

    async def drain(async_iterator):
        """Drive an async generator to exhaustion.

        `_stream` and `_astream` are generators, so their bodies do not run
        until they are iterated. Draining them is what actually reaches the
        parent, so that is what this test has to do.
        """
        async for _ in async_iterator:
            pass

    tools = [convert_to_openai_tool(retrieve_data_tool)]
    with pytest.raises(Exception):
        google._generate([HumanMessage(content="hi")], tools=tools)
    with pytest.raises(Exception):
        list(google._stream([HumanMessage(content="hi")], tools=tools))
    with pytest.raises(Exception):
        asyncio.run(google._agenerate([HumanMessage(content="hi")], tools=tools))
    with pytest.raises(Exception):
        asyncio.run(drain(google._astream([HumanMessage(content="hi")], tools=tools)))

    assert len(seen) == 4, seen
    for name, flag in seen:
        assert flag == {"disable": True}, f"{name} would let Google run the tool: {flag}"


def test_the_streaming_overrides_keep_the_shape_of_the_methods_they_override():
    """An override has to be the same kind of function as the parent.

    The parent `_astream` is an async generator, so `await` on it raises
    TypeError: an async generator is not awaitable. Delegating with `async for`
    keeps the same shape; awaiting it silently changes the contract.
    """
    import inspect

    from langchain_google_genai import ChatGoogleGenerativeAI

    from api.agents.providers.google import AgentOwnedToolLoop

    def kind(function):
        if inspect.isasyncgenfunction(function):
            return "async generator"
        if inspect.iscoroutinefunction(function):
            return "coroutine"
        if inspect.isgeneratorfunction(function):
            return "generator"
        return "plain function"

    for method in ("_generate", "_agenerate", "_stream", "_astream"):
        assert kind(getattr(AgentOwnedToolLoop, method)) == kind(
            getattr(ChatGoogleGenerativeAI, method)
        ), f"{method} is not the same kind of function as the method it overrides"


def test_google_still_sends_the_tools():
    """Disabling the loop must not remove the tool from the request."""
    import api.agents.llm as llm
    from api.agents.tools import retrieve_data_tool
    from langchain_core.messages import HumanMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool

    google = llm.build_google_model()
    request = google._prepare_request(
        [HumanMessage(content="hi")],
        tools=[convert_to_openai_tool(retrieve_data_tool)],
        automatic_function_calling={"disable": True},
    )
    assert request["config"].tools, "the agent's tool must still reach the model"
    assert request["config"].automatic_function_calling.disable is True
