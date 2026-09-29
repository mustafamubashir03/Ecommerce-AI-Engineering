"""Groq as a first class provider, behind the same model boundary.

Groq speaks the OpenAI protocol, so the project points the OpenAI client it
already depends on at Groq's base url. Nothing else about the RAG pipeline
changes: same retrieval, same tool, same response contract.
"""

import asyncio

import pytest

import fake_openrouter as fake
from api.agents.model_router import PRIMARY, ProviderPlan, RoutedChatModel
from api.agents.prompts import IntentRouterResponse
from api.agents.errors import trip

GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


class FakeProvider:
    """Stands in for one provider, so the routing policy can be asserted."""

    def __init__(self, name: str, models, answer="answer", fail_with=None, tokens=0, strict=True):
        self.name = name
        self.models = list(models)
        self.answer = answer
        self.fail_with = fail_with
        self.tokens = tokens
        self.calls: list[str] = []
        self.strict = strict

    def build(self, model_id: str):
        self.calls.append(model_id)
        return _FakeModel(self, model_id)

    def bind_tools(self, tools, **kwargs):
        return self


class _FakeModel:
    def __init__(self, provider: FakeProvider, model_id: str):
        self.provider = provider
        self.model_id = model_id
        self.answer = provider.answer

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        provider = self.provider

        class _Structured:
            def invoke(self, input, config=None, **kw):
                if provider.fail_with is not None:
                    raise provider.fail_with
                return IntentRouterResponse(question_relevancy=True, answer="")

            async def ainvoke(self, input, config=None, **kw):
                if provider.fail_with is not None:
                    raise provider.fail_with
                return IntentRouterResponse(question_relevancy=True, answer="")

        return _Structured()

    def _raise_if_asked(self):
        if self.provider.fail_with is not None:
            raise self.provider.fail_with

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        self._raise_if_asked()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        self._raise_if_asked()
        for index in range(self.provider.tokens or 1):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"{self.model_id} "))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


def groq_plan(provider: FakeProvider, **kwargs) -> ProviderPlan:
    return ProviderPlan(
        name="groq",
        models=provider.models,
        build=provider.build,
        strict_structured_output=kwargs.get("strict", GROQ_MODELS),
    )


def routed(fallbacks=(), pool=("model/a",)) -> RoutedChatModel:
    return RoutedChatModel(models=list(pool), options={"api_key": "t"}, fallbacks=list(fallbacks))


# --- tool binding reaches the request ---------------------------------------


def test_bound_tools_reach_the_delegate_call():
    """A binding is unwrapped, because `_generate` on one drops the tools.

    `bind_tools` returns a RunnableBinding that only applies its tools on
    `invoke`. Calling `_generate` on it reaches the wrapped model with no tools
    at all, which silently turns the agent into a chatbot. This guards that the
    unwrapped model and the converted tool schemas are both passed on.
    """
    from langchain_core.tools import tool

    from api.agents.model_router import RoutedChatModel

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "x"

    seen: dict = {}

    class Recorder:
        def bind_tools(self, tools, **kwargs):
            seen["bind_kwargs"] = kwargs
            return type(
                "Binding",
                (),
                {"bound": self, "kwargs": {"tools": [{"function": {"name": "lookup"}}], **kwargs}},
            )()

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen["call_kwargs"] = kwargs
            raise RuntimeError("stop here")

    model = RoutedChatModel(models=["a"], options={}, fallbacks=[]).bind_tools([lookup])
    inner, call_kwargs = model._delegate_parts(Recorder())
    assert isinstance(inner, Recorder), "the raw model is used, not the binding"
    assert call_kwargs["tools"] == [{"function": {"name": "lookup"}}]

    # And end to end: the tools are present in the kwargs the step actually calls
    # the model with, which is where they were being lost.
    routed = RoutedChatModel(
        models=[], options={}, bound_tools=[lookup], fallbacks=[ProviderPlan(name="p", models=["m"], build=lambda _m: Recorder())]
    )
    with pytest.raises(RuntimeError):
        routed._generate([{"role": "user", "content": "hi"}])
    assert seen["call_kwargs"]["tools"] == [{"function": {"name": "lookup"}}], (
        "the tools must be in the per step call"
    )


def test_no_tools_bound_means_no_tool_kwargs():
    from api.agents.model_router import RoutedChatModel

    plain = RoutedChatModel(models=["a"], options={}, fallbacks=[])

    class Plain:
        pass

    inner, kwargs = plain._delegate_parts(Plain())
    assert isinstance(inner, Plain)
    assert kwargs == {}


# --- configuration ----------------------------------------------------------


def test_groq_base_url_is_the_documented_one():
    from api.core.settings import get_settings

    assert get_settings().llm.groq.base_url == "https://api.groq.com/openai/v1"
    assert get_settings().llm.groq.api_key_env == "GROQ_API_KEY"


def test_groq_models_come_from_configuration():
    from api.core.settings import get_settings

    groq = get_settings().llm.groq
    assert groq.model_pool()[0] == groq.primary_model, "the primary is tried first"
    assert set(groq.model_pool()) <= set(GROQ_MODELS), "only models the live catalog offered"
    assert len(groq.model_pool()) >= 2, "more than one model, so a fallback exists"


def test_groq_models_are_not_hardcoded_in_python():
    import inspect

    import api.agents.llm as llm

    source = inspect.getsource(llm)
    assert "gpt-oss" not in source and "qwen3" not in source, "model ids belong in config.yaml"
    assert "gsk_" not in source, "no key material in source"


def test_strict_capable_models_match_the_documented_set():
    from api.core.settings import get_settings

    declared = set(get_settings().llm.groq.strict_structured_output)
    assert declared == set(GROQ_MODELS), (
        "Groq documents strict JSON Schema for exactly these three models"
    )


def test_a_missing_key_disables_groq_without_breaking(monkeypatch):
    import api.agents.llm as llm
    from api.core.settings import get_settings

    monkeypatch.delenv(get_settings().llm.groq.api_key_env, raising=False)
    assert llm.build_groq_model(GROQ_MODELS[0]) is None


def test_disabled_groq_is_skipped(monkeypatch):
    import api.agents.llm as llm
    from api.core.settings import get_settings

    monkeypatch.setattr(get_settings().llm.groq, "enabled", False)
    assert llm.build_groq_model(GROQ_MODELS[0]) is None


def test_groq_is_built_through_the_openai_compatible_client(monkeypatch):
    """The same OpenAI client, pointed at Groq. No Groq SDK."""
    import api.agents.llm as llm
    from api.core.settings import get_settings
    from langchain_openai import ChatOpenAI

    seen: dict = {}

    def capture(self, *args, **kwargs):
        seen.update(kwargs)
        seen["model"] = kwargs.get("model")

    monkeypatch.setattr(ChatOpenAI, "__init__", capture)
    monkeypatch.setenv(get_settings().llm.groq.api_key_env, "test-key")

    llm.build_groq_model("openai/gpt-oss-120b")
    assert seen["base_url"] == "https://api.groq.com/openai/v1"
    assert seen["api_key"] == "test-key"
    assert seen["model"] == "openai/gpt-oss-120b"
    assert seen["timeout"] == get_settings().llm.groq.timeout_seconds


def test_fallback_order_is_configured_not_scattered():
    from api.core.settings import get_settings

    order = get_settings().llm.fallback_order
    assert "groq" in order, "groq is a configured provider"
    assert len(order) == len(set(order)), "no duplicates"


# --- ordering ---------------------------------------------------------------


def test_openrouter_success_never_calls_groq(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS)
    calls = fake.install(monkeypatch, {"model/a": []})
    result = routed([groq_plan(provider)]).invoke("hello")
    assert result.content == "answer"
    assert calls == ["model/a"]
    assert provider.calls == [], "the fallback provider stays untouched"


def test_groq_answers_when_openrouter_quota_is_exhausted(monkeypatch):
    """The main case: a known account quota costs zero OpenRouter requests."""
    provider = FakeProvider("groq", GROQ_MODELS)
    calls = fake.install(monkeypatch, {model: [fake.daily_cap_error()] for model in ("model/a", "model/b")})
    result = routed([groq_plan(provider)]).invoke("hello")
    assert result.content == "answer"
    assert provider.calls == [GROQ_MODELS[0]], "the first Groq model, once"
    assert calls == ["model/a"], "the OpenRouter pool is skipped, not walked"


def test_openrouter_pool_is_exhausted_before_groq(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS)
    calls = fake.install(monkeypatch, {"model/a": [fake.server_error(503)], "model/b": [fake.server_error(503)]})
    result = routed([groq_plan(provider)], pool=("model/a", "model/b")).invoke("hello")
    assert result.content == "answer"
    assert calls == ["model/a", "model/b"]
    assert provider.calls == [GROQ_MODELS[0]]


def test_groq_falls_back_to_its_own_next_model(monkeypatch):
    first = FakeProvider("a", GROQ_MODELS[:1], fail_with=fake.server_error(503))
    second = FakeProvider("b", GROQ_MODELS[1:])
    plan_a = ProviderPlan(name="groq-1", models=GROQ_MODELS[:1], build=first.build)
    plan_b = ProviderPlan(name="groq-2", models=GROQ_MODELS[1:], build=second.build)
    fake.install(monkeypatch, {model: [fake.daily_cap_error()] for model in ("model/a", "model/b")})
    result = routed([plan_a, plan_b], pool=("model/a", "model/b")).invoke("hello")
    assert result.content == "answer"
    assert first.calls == [GROQ_MODELS[0]] and second.calls == [GROQ_MODELS[1]]


def test_an_application_error_is_not_handed_to_groq(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS)
    fake.install(monkeypatch, {"model/a": [fake.bad_request_error()]})
    with pytest.raises(Exception):
        routed([groq_plan(provider)]).invoke("hello")
    assert provider.calls == [], "a bad request fails the same way on Groq"


def test_no_recursion_back_to_openrouter(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS, fail_with=fake.server_error(503))
    calls = fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    with pytest.raises(Exception):
        routed([groq_plan(provider)]).invoke("hello")
    assert calls == ["model/a"], "OpenRouter was not asked a second time"
    assert provider.calls == GROQ_MODELS, "each Groq model once, then stop"


def test_a_cooldown_skips_openrouter_entirely(monkeypatch):
    trip(fake.daily_cap_error(), 300)
    provider = FakeProvider("groq", GROQ_MODELS)
    calls = fake.install(monkeypatch, {"model/a": []})
    result = routed([groq_plan(provider)]).invoke("hello")
    assert result.content == "answer"
    assert provider.calls == [GROQ_MODELS[0]]
    assert calls == [], "zero OpenRouter requests while the cooldown is active"


def test_steps_are_ordered_primary_then_providers():
    provider = FakeProvider("groq", GROQ_MODELS)
    model = routed([groq_plan(provider)], pool=("model/a", "model/b"))
    order = [(step.provider, step.model_id) for step in model._steps()]
    assert order == [
        (PRIMARY, "model/a"),
        (PRIMARY, "model/b"),
        ("groq", GROQ_MODELS[0]),
        ("groq", GROQ_MODELS[1]),
        ("groq", GROQ_MODELS[2]),
    ]


# --- structured outputs -----------------------------------------------------


def test_a_strict_request_only_reaches_strict_capable_models():
    """Groq documents strict JSON Schema for three models, so only those get it."""
    provider = FakeProvider("groq", GROQ_MODELS + ["some/other-model"])
    model = routed([groq_plan(provider, strict=GROQ_MODELS)])
    strict = [s.model_id for s in model._steps(strict_only=True) if s.provider == "groq"]
    assert strict == GROQ_MODELS, "a model without strict support is never asked for it"
    loose = [s.model_id for s in model._steps(strict_only=False) if s.provider == "groq"]
    assert "some/other-model" in loose, "without strict it stays available for function calling"


def test_structured_output_reaches_groq_when_openrouter_is_spent(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS)
    fake.install(monkeypatch, {model: [fake.daily_cap_error()] for model in ("model/a",)})
    verdict = routed([groq_plan(provider)]).with_structured_output(IntentRouterResponse).invoke(
        [{"role": "user", "content": "hi"}]
    )
    assert verdict.question_relevancy is True
    assert provider.calls == [GROQ_MODELS[0]]


def test_structured_output_async_reaches_groq(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS)
    fake.install(monkeypatch, {"model/a": [fake.daily_cap_error()]})
    verdict = asyncio.run(
        routed([groq_plan(provider)]).with_structured_output(IntentRouterResponse).ainvoke(
            [{"role": "user", "content": "hi"}]
        )
    )
    assert verdict.question_relevancy is True


def test_the_streaming_path_never_asks_for_a_json_schema():
    """Groq cannot combine structured outputs with streaming, so it never does.

    The streamed answer is plain text: the router builds no runnable, and the
    steps for a call carry no response_format.
    """
    import inspect

    from api.agents.model_router import RoutedChatModel

    source = inspect.getsource(RoutedChatModel._stream)
    assert "response_format" not in source and "json_schema" not in source
    assert "structured" not in source, "streaming is plain text generation"


# --- streaming --------------------------------------------------------------


def test_groq_streams_when_openrouter_produced_nothing(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS, tokens=2)
    calls = fake.install(monkeypatch, {model: [fake.server_error(503)] for model in ("model/a", "model/b")})
    streamed = "".join(c.content for c in routed([groq_plan(provider)], pool=("model/a", "model/b")).stream("hi"))
    assert calls == ["model/a", "model/b"], "the OpenRouter pool produced nothing"
    assert provider.calls == [GROQ_MODELS[0]], "Groq streams the answer"
    assert streamed.strip().startswith(GROQ_MODELS[0]), "the streamed text is the Groq model's"


def test_no_splice_across_providers(monkeypatch):
    """An OpenRouter prefix and a Groq answer must never share one stream."""
    provider = FakeProvider("groq", GROQ_MODELS, tokens=2)
    fake.install(monkeypatch, {"model/a": [fake.server_error(503)]}, tokens_before_failure=1, stream_text="openrouter partial")
    emitted = []
    with pytest.raises(Exception):
        for chunk in routed([groq_plan(provider)]).stream("hi"):
            emitted.append(chunk.content)
    assert provider.calls == [], "Groq must not start after output began"
    assert "".join(emitted).strip() == "openrouter"


def test_async_stream_keeps_the_no_splice_rule(monkeypatch):
    provider = FakeProvider("groq", GROQ_MODELS, tokens=2)
    fake.install(monkeypatch, {"model/a": [fake.server_error(503)]}, tokens_before_failure=1, stream_text="partial")
    emitted = []

    async def main():
        async for chunk in routed([groq_plan(provider)]).astream([{"role": "user", "content": "hi"}]):
            emitted.append(chunk.content)

    with pytest.raises(Exception):
        asyncio.run(main())
    assert provider.calls == []
    assert "".join(emitted).strip() == "partial"


# --- rate limits ------------------------------------------------------------


def test_a_groq_429_rotates_to_the_next_groq_model(monkeypatch):
    """A Groq rate limit is availability, so the next model is tried once."""
    first, second = FakeProvider("g1", GROQ_MODELS[:1]), FakeProvider("g2", GROQ_MODELS[1:])
    first_fail = fake.FakeOpenRouterError(429, {"error": {"message": "Rate limit exceeded"}}, {"retry-after": "2"})
    plans = [
        ProviderPlan(name="g1", models=GROQ_MODELS[:1], build=lambda mid: _raise_model(first, mid, first_fail)),
        ProviderPlan(name="g2", models=GROQ_MODELS[1:], build=second.build),
    ]
    fake.install(monkeypatch, {m: [fake.daily_cap_error()] for m in ("model/a",)})
    result = routed(plans).invoke("hi")
    assert result.content == "answer"
    assert second.calls == [GROQ_MODELS[1]]


def _raise_model(provider: FakeProvider, model_id: str, error: Exception):
    provider.calls.append(model_id)
    raise error


def test_a_retry_after_header_does_not_cause_a_storm(monkeypatch):
    """The 429 is absorbed by rotation, never by an immediate re-ask."""
    provider = FakeProvider("groq", GROQ_MODELS)
    error = fake.FakeOpenRouterError(429, {"error": {"message": "Rate limit exceeded"}}, {"retry-after": "2"})
    plan = ProviderPlan(name="g1", models=GROQ_MODELS, build=lambda mid: _raise_model(provider, mid, error))
    fake.install(monkeypatch, {m: [fake.daily_cap_error()] for m in ("model/a",)})
    with pytest.raises(Exception):
        routed([plan]).invoke("hi")
    assert provider.calls == GROQ_MODELS, "each model exactly once, no repeats"
