"""A 413 the body proves is a token window moves to the next provider.

Groq, and providers behind it, refuse a request larger than their token window
with HTTP 413 rather than 429, and name both figures:

    "on tokens per minute (TPM): Limit 8000, Requested 8154, ..."

The window belongs to the organisation, not to one model, so a sibling model on
the same provider is refused for the same reason at the same cost. The walk
therefore skips to the next provider. Every other 413 stays unrecoverable:
OpenRouter answers a payload too large for its proxy with 413, and that request
would fail identically on every model it could be sent to.

The window value is never configured. It is read from the body, because it
belongs to the account and the account's own value is the authoritative one.
"""

import httpx
import pytest

import fake_openrouter as fake
from api.agents.errors import (
    CEILING_PERMANENT,
    CEILING_TRANSIENT,
    describe,
    hold_error,
    is_fallback_worthy,
    status_of,
    token_ceiling,
    trip,
)
from api.agents.model_router import RoutedChatModel
from api.agents.routing.policy import NEXT_PROVIDER, RAISE, action
from api.agents.routing.steps import ProviderPlan

# --- errors -----------------------------------------------------------------


def groq_413(limit: int = 8000, requested: int = 8154, used: int | None = None) -> Exception:
    """The real 413 shape, with the window figures the provider reported."""
    used_clause = f"Used {used}, " if used is not None else ""
    message = (
        "Request too large for model `openai/gpt-oss-120b` in organization "
        "`org_test` service tier `on_demand` on tokens per minute (TPM): "
        f"Limit {limit}, {used_clause}Requested {requested}, please reduce your "
        "message size and try again."
    )
    return _error(413, {"message": message, "type": "tokens", "code": "rate_limit_exceeded"})


def openrouter_413() -> Exception:
    """A payload too large for OpenRouter's proxy. Every model would refuse it."""
    return _error(
        413,
        {
            "error": {"message": "Request payload too large", "code": 413},
            "error_type": "payload_too_large",
        },
    )


def _error(status: int, body: dict) -> Exception:
    from openai import APIStatusError

    response = httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return APIStatusError(f"Error code: {status}", response=response, body=body)


def verdict(error, *, openrouter=False, last_of_provider=False, last_step=False) -> str:
    return action(
        error, openrouter=openrouter, last_of_provider=last_of_provider, last_step=last_step
    )


@pytest.fixture(autouse=True)
def clean_cooldown(monkeypatch):
    """Every test starts with no provider being held off."""
    from api.agents.errors import cooldown

    monkeypatch.setattr(cooldown, "_cooldown", cooldown._Cooldown())


# --- the detector -----------------------------------------------------------


def test_requested_over_limit_is_a_permanent_ceiling():
    assert status_of(groq_413(8000, 8154)) == 413
    assert token_ceiling(groq_413(8000, 8154)) == CEILING_PERMANENT


def test_requested_under_limit_is_a_transient_ceiling():
    assert token_ceiling(groq_413(8000, 6278, used=4373)) == CEILING_TRANSIENT


def test_the_window_value_is_read_from_the_body_not_configured():
    """A different account's window is recognised without any code change."""
    assert token_ceiling(groq_413(limit=250_000, requested=260_000)) == CEILING_PERMANENT
    assert token_ceiling(groq_413(limit=1_000, requested=400)) == CEILING_TRANSIENT


# --- the policy decision ----------------------------------------------------


def test_a_permanent_ceiling_goes_to_the_next_provider():
    assert verdict(groq_413(8000, 8154)) == NEXT_PROVIDER


def test_a_transient_ceiling_goes_to_the_next_provider():
    assert verdict(groq_413(8000, 6278, used=4373)) == NEXT_PROVIDER


def test_a_ceiling_never_rotates_inside_its_own_provider():
    """The verb must be NEXT_PROVIDER, so no sibling model is attempted."""
    taken = verdict(groq_413(), last_of_provider=False, last_step=False)
    assert taken == NEXT_PROVIDER
    assert taken != "rotate"


# --- everything that is not a proven ceiling still raises ------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(openrouter_413(), id="openrouter payload_too_large"),
        pytest.param(_error(413, {"message": "Request payload too large"}), id="bare 413"),
        pytest.param(_error(413, {"error": {"code": 413, "message": "too large"}}), id="openrouter shape"),
        pytest.param(_error(413, {"message": "limit reached", "type": "tokens"}), id="tokens but no figures"),
        pytest.param(_error(413, {"message": "Limit 8000, Requested 8154"}), id="figures but no signal"),
        pytest.param(_error(413, {}), id="empty body"),
        pytest.param(_error(413, {"message": "tpm"}), id="signal only"),
    ],
)
def test_an_ambiguous_413_still_raises(error):
    assert token_ceiling(error) is None, "an unproven 413 must not be read as a ceiling"
    assert is_fallback_worthy(error) is False, "an unproven 413 must not be fallback worthy"
    assert verdict(error) == RAISE, "an unproven 413 must still raise"
    assert verdict(error, openrouter=True) == RAISE, "an unproven 413 must still raise from the pool"


def test_an_oversized_request_does_not_walk_the_pool(monkeypatch):
    """A payload limit fails everywhere, so no other model may be paid for."""
    calls = fake.install(monkeypatch, {name: [openrouter_413()] for name in ("a", "b", "c")})
    routed = RoutedChatModel(models=["a", "b", "c"], options={"api_key": "test"})

    with pytest.raises(Exception) as raised:
        routed.invoke("a very long question")

    assert calls == ["a"], "the pool must stop at the first unrecoverable 413"
    assert status_of(raised.value) == 413


def test_the_openrouter_413_contract_is_pinned_to_raise():
    """OpenRouter's own 413 is a proxy payload limit, and stays unrecoverable."""
    error = openrouter_413()
    assert status_of(error) == 413
    assert token_ceiling(error) is None
    assert verdict(error, openrouter=True) == RAISE


# --- the walker -------------------------------------------------------------


class _Answering:
    """A built client that answers, streaming or not."""

    def __init__(self, text: str = "answered", words: tuple[str, ...] = ()):
        self.words = words
        self.text = "".join(words) if words else text
        self.bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        for word in self.words:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


class _Ceiling(_Answering):
    """A built client that is refused on every call, as a real one would be.

    The refusal is raised by the call, not by the constructor: `ChatOpenAI(...)`
    makes no request, so in production a 413 always arrives after the model has
    been built.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise groq_413()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        raise groq_413()

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        raise groq_413()
        yield  # pragma: no cover - makes this a generator, like a real stream

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        raise groq_413()
        yield  # pragma: no cover - makes this a generator, like a real stream


def two_providers(asked: dict) -> list[ProviderPlan]:
    """Groq, which is refused on its first model, then a provider that answers."""
    def build(model_id: str):
        asked.setdefault(model_id.split("-")[0], []).append(model_id)
        if model_id.startswith("groq"):
            return _Ceiling()
        return _Answering(words=("from ", "google"))

    return [
        ProviderPlan(name="groq", models=["groq-a", "groq-b", "groq-c"], build=build),
        ProviderPlan(name="google", models=["google-1"], build=build),
    ]


def test_a_ceiling_skips_sibling_models_and_reaches_the_next_provider():
    """The observed case: Groq model A 413s, B and C are never asked, Google is."""
    asked: dict = {}
    routed = RoutedChatModel(models=[], options={"api_key": "test"}, fallbacks=two_providers(asked))

    assert routed.invoke("laptops with 16gb ram").content == "from google"
    assert asked["groq"] == ["groq-a"], f"sibling models must be skipped, got {asked['groq']}"
    assert asked["google"] == ["google-1"], f"the next provider must be tried, got {asked['google']}"


def test_each_step_is_attempted_at_most_once():
    """A pool of ceilings terminates instead of looping."""
    asked: list[str] = []

    def build(model_id: str):
        asked.append(model_id)
        raise groq_413()

    plan = ProviderPlan(name="groq", models=["g-1", "g-2", "g-3"], build=build)
    routed = RoutedChatModel(models=[], options={}, fallbacks=[plan])

    with pytest.raises(Exception) as raised:
        routed.invoke("a question")

    assert asked == ["g-1"], "the provider is skipped, so one attempt is enough"
    assert status_of(raised.value) == 413


def test_a_transient_ceiling_is_held_by_the_pool_failure_path(monkeypatch):
    """A full window is remembered, so the next turn is not paid for again."""
    from api.agents.routing.walker import after_failure

    fake.install(monkeypatch, {"a": [], "b": [], "c": []})
    routed = RoutedChatModel(models=["a", "b", "c"], options={"api_key": "test"})

    steps = routed._steps()
    steps[-1] = steps[-1]._replace(build=lambda: None)
    kind, _, to_raise = after_failure(groq_413(8000, 6278, used=4373), steps, len(steps) - 1)

    assert kind == "raise"
    held = hold_error()
    assert held is not None, "a full window is held, not forgotten"
    assert is_fallback_worthy(held), "a held ceiling is walked away from, not re-raised"
    assert to_raise is not None


def test_a_permanent_ceiling_is_not_held_against_the_provider(monkeypatch):
    """One oversized request must not disable the provider for smaller ones."""
    from api.agents.routing.walker import after_failure

    fake.install(monkeypatch, {"a": [], "b": [], "c": []})
    routed = RoutedChatModel(models=["a", "b", "c"], options={"api_key": "test"})

    steps = routed._steps()
    steps[-1] = steps[-1]._replace(build=lambda: None)
    after_failure(groq_413(8000, 8154), steps, len(steps) - 1)

    assert hold_error() is None, "a request larger than the window is its own problem"


def test_a_held_ceiling_is_never_re_raised(monkeypatch):
    """The held error must be walked away from, or every turn would fail on it."""
    from api.agents.routing.steps import steps_for_call

    fake.install(monkeypatch, {"a": [], "b": []})
    asked: list[str] = []
    routed = RoutedChatModel(
        models=["a", "b"],
        options={"api_key": "test"},
        fallbacks=[ProviderPlan(name="groq", models=["g-1"], build=lambda m: _Answering())],
    )

    trip(groq_413(), 30.0)
    assert hold_error() is not None

    remaining = steps_for_call(routed.models, routed.options, routed.fallbacks, lambda *a, **k: None)
    assert [step.provider for step in remaining] == ["groq"], "only the primary pool is cut"
    assert asked == []


def test_cooldown_never_cuts_a_non_primary_provider(monkeypatch):
    """The held store only ever removes the primary pool, by design."""
    from api.agents.routing.steps import steps_for_call

    fake.install(monkeypatch, {"a": [], "b": []})
    routed = RoutedChatModel(
        models=["a", "b"],
        options={"api_key": "test"},
        fallbacks=[ProviderPlan(name="groq", models=["g-1", "g-2"], build=lambda m: _Answering())],
    )
    assert [step.provider for step in routed._steps()] == ["openrouter", "openrouter", "groq", "groq"]

    trip(groq_413(), 30.0)
    remaining = steps_for_call(routed.models, routed.options, routed.fallbacks, lambda *a, **k: None)
    assert [step.provider for step in remaining] == ["groq", "groq"]


# --- the routes that carry the call ----------------------------------------


class _Failing(_Answering):
    """A built client refused on every call by the given error."""

    def __init__(self, error: Exception = None, text="answered", words=()):
        super().__init__(text=text, words=words)
        self.error = error

    def _refuse(self):
        if self.error is not None:
            raise self.error
        raise AssertionError("this model was built to answer")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse()

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse()
        yield  # pragma: no cover - a generator, like a real stream

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._refuse()
        yield  # pragma: no cover - a generator, like a real stream


def test_streaming_still_rotates_an_availability_failure():
    """A 503 is a model problem, so the next model of the same provider is tried."""
    asked: list[str] = []

    def build(model_id: str):
        asked.append(model_id)
        if model_id == "g-a":
            return _Failing(error=_error(503, {"error": {"code": 503, "message": "overloaded"}}))
        return _Answering(words=("from ", "b"))

    plan = ProviderPlan(name="groq", models=["g-a", "g-b"], build=build)
    routed = RoutedChatModel(models=[], options={}, fallbacks=[plan])

    assert "".join(c.content for c in routed.stream("q")) == "from b"
    assert asked == ["g-a", "g-b"], "a 503 must still rotate inside the provider"


def test_streaming_still_rotates_a_provider_specific_429():
    asked: list[str] = []

    def build(model_id: str):
        asked.append(model_id)
        if model_id == "g-a":
            return _Failing(
                error=_error(429, {"error": {"code": 429, "message": "slow", "metadata": {"provider_name": "x"}}})
            )
        return _Answering(words=("from ", "b"))

    plan = ProviderPlan(name="groq", models=["g-a", "g-b"], build=build)
    routed = RoutedChatModel(models=[], options={}, fallbacks=[plan])

    assert "".join(c.content for c in routed.stream("q")) == "from b"
    assert asked == ["g-a", "g-b"]


def test_streaming_still_advances_past_an_account_quota(monkeypatch):
    """A quota belongs to the account, so the primary pool is skipped whole.

    The quota branch is deliberately primary-pool only, as it is in `action`, so
    this drives the OpenRouter pool rather than a fallback provider.
    """
    asked: list[str] = []

    def build(model_id: str, **options):
        asked.append(model_id)
        if model_id.startswith("model/"):
            return _Failing(error=_error(429, {"error": {"code": 429, "message": "free-models-per-day"}}))
        return _Answering(words=("from ", "google"))

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    fallbacks = [ProviderPlan(name="google", models=["google-1"], build=lambda m: _Answering(words=("from ", "google")))]
    routed = RoutedChatModel(models=["model/a", "model/b"], options={}, fallbacks=fallbacks)

    assert "".join(c.content for c in routed.stream("q")) == "from google"
    assert asked == ["model/a"], f"the quota pool must be skipped whole, got {asked}"


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_streaming_client_errors_are_unchanged(status):
    """No client error other than a proven ceiling starts a fallback."""
    asked: list[str] = []

    def build(model_id: str):
        asked.append(model_id)
        return _Failing(error=_error(status, {"error": {"code": status, "message": "no"}}))

    fallbacks = [
        ProviderPlan(name="groq", models=["g-a", "g-b"], build=build),
        ProviderPlan(name="google", models=["google-1"], build=build),
    ]
    routed = RoutedChatModel(models=[], options={}, fallbacks=fallbacks)

    with pytest.raises(Exception) as raised:
        list(routed.stream("q"))
    assert status_of(raised.value) == status
    assert asked == ["g-a"], f"a {status} must not fall through, got {asked}"


def test_streaming_partial_output_blocks_handover():
    """Tokens are on the wire, so no other provider is asked, ever."""
    asked: list[str] = []

    class _Partial(_Failing):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            yield ChatGenerationChunk(message=AIMessageChunk(content="half an "))
            raise groq_413()

    def build(model_id: str):
        asked.append(model_id)
        if model_id == "groq-a":
            return _Partial()
        raise AssertionError(f"must not be asked after output was emitted: {model_id}")

    fallbacks = [
        ProviderPlan(name="groq", models=["groq-a", "groq-b", "groq-c"], build=build),
        ProviderPlan(name="google", models=["google-1"], build=build),
    ]
    routed = RoutedChatModel(models=[], options={}, fallbacks=fallbacks)

    chunks = []
    with pytest.raises(Exception):
        for chunk in routed.stream("a question"):
            chunks.append(chunk.content)

    assert "".join(chunks) == "half an ", "the emitted prefix must be kept"
    assert asked == ["groq-a"], f"no handover after output, got {asked}"


def test_astream_partial_output_blocks_handover():
    import asyncio

    asked: list[str] = []

    class _Partial(_Failing):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            yield ChatGenerationChunk(message=AIMessageChunk(content="half an "))
            raise groq_413()

    def build(model_id: str):
        asked.append(model_id)
        if model_id == "groq-a":
            return _Partial()
        raise AssertionError(f"must not be asked after output was emitted: {model_id}")

    fallbacks = [
        ProviderPlan(name="groq", models=["groq-a", "groq-b"], build=build),
        ProviderPlan(name="google", models=["google-1"], build=build),
    ]
    routed = RoutedChatModel(models=[], options={}, fallbacks=fallbacks)

    async def collect():
        chunks = []
        async for chunk in routed.astream("a question"):
            chunks.append(chunk.content)
        return chunks

    with pytest.raises(Exception):
        asyncio.run(collect())
    assert asked == ["groq-a"], f"no handover after output, got {asked}"


def test_streaming_keeps_tools_bound_on_the_answering_model():
    """A ceiling before output must not cost the tool binding."""
    from langchain_core.tools import tool

    asked: dict = {}
    built: list[_Answering] = []
    original = _Answering.__init__

    def spy_init(self, text="answered", words=()):
        original(self, text=text, words=words or ("from ", "google"))
        built.append(self)

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "x"

    _Answering.__init__ = spy_init
    try:
        routed = RoutedChatModel(
            models=[], options={"api_key": "test"}, fallbacks=two_providers(asked)
        )
        text = "".join(c.content for c in routed.bind_tools([lookup]).stream("a question"))
    finally:
        _Answering.__init__ = original

    assert text == "from google"
    assert asked["groq"] == ["groq-a"]
    assert any(m.bound_tools and m.bound_tools[0].name == "lookup" for m in built), (
        "tools must reach the model that answered"
    )


def test_sync_generation_advances_to_the_next_provider():
    asked: dict = {}
    routed = RoutedChatModel(models=[], options={"api_key": "test"}, fallbacks=two_providers(asked))

    assert routed.invoke("a question").content == "from google"
    assert asked["groq"] == ["groq-a"]


def test_async_generation_advances_to_the_next_provider():
    import asyncio

    asked: dict = {}
    routed = RoutedChatModel(models=[], options={"api_key": "test"}, fallbacks=two_providers(asked))

    assert asyncio.run(routed.ainvoke("a question")).content == "from google"
    assert asked["groq"] == ["groq-a"]


def test_streaming_advances_before_any_output_and_never_splices():
    """Streaming honours NEXT_PROVIDER, so the failing provider is skipped whole."""
    asked: dict = {}
    routed = RoutedChatModel(models=[], options={"api_key": "test"}, fallbacks=two_providers(asked))

    text = "".join(chunk.content for chunk in routed.stream("a question"))

    assert text == "from google", "the answer must come from one provider only"
    assert asked["groq"] == ["groq-a"], f"sibling models must be skipped, got {asked['groq']}"
    assert asked["google"] == ["google-1"], f"the next provider must be tried, got {asked['google']}"


def test_astream_advances_before_any_output_and_never_splices():
    """The asynchronous twin skips the same way."""
    import asyncio

    asked: dict = {}
    routed = RoutedChatModel(models=[], options={"api_key": "test"}, fallbacks=two_providers(asked))

    async def collect():
        return "".join([chunk.content async for chunk in routed.astream("a question")])

    assert asyncio.run(collect()) == "from google"
    assert asked["groq"] == ["groq-a"], f"sibling models must be skipped, got {asked['groq']}"
    assert asked["google"] == ["google-1"]


def test_streaming_ambiguous_413_raises_without_asking_anything_else():
    """A 413 with no ceiling evidence stops the stream where it happened."""
    asked: list[str] = []

    class _Ambiguous(_Answering):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            raise openrouter_413()
            yield  # pragma: no cover - a generator, like a real stream

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            raise openrouter_413()
            yield  # pragma: no cover - a generator, like a real stream

    def build(model_id: str):
        asked.append(model_id)
        return _Ambiguous()

    fallbacks = [
        ProviderPlan(name="groq", models=["groq-a", "groq-b", "groq-c"], build=build),
        ProviderPlan(name="google", models=["google-1"], build=build),
    ]
    routed = RoutedChatModel(models=[], options={}, fallbacks=fallbacks)

    with pytest.raises(Exception) as raised:
        list(routed.stream("a question"))
    assert status_of(raised.value) == 413
    assert asked == ["groq-a"], f"nothing else may be asked, got {asked}"

    asked.clear()
    with pytest.raises(Exception):
        list(routed.stream("a question"))
    assert asked == ["groq-a"], f"the async path must behave the same, got {asked}"


def test_streaming_after_partial_output_does_not_hand_over():
    """Tokens are on the wire, so the stream stays with the failing provider."""
    asked: list[str] = []

    class _Partial(_Ceiling):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            yield ChatGenerationChunk(message=AIMessageChunk(content="half an "))
            raise groq_413()

    def build(model_id: str):
        asked.append(model_id)
        if model_id.startswith("groq"):
            return _Partial()
        raise AssertionError("must not be asked after output was emitted")

    plan = ProviderPlan(name="groq", models=["groq-a"], build=build)
    routed = RoutedChatModel(models=[], options={}, fallbacks=[plan])

    chunks = []
    with pytest.raises(Exception):
        for chunk in routed.stream("a question"):
            chunks.append(chunk.content)

    assert "".join(chunks) == "half an ", "the emitted prefix must be kept"
    assert asked == ["groq-a"]


def test_tool_calling_survives_the_provider_transition():
    """The tools are rebound on the model that answers, not lost in the walk."""
    from langchain_core.tools import tool

    asked: dict = {}
    built: list[_Answering] = []

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return "x"

    original = _Answering.__init__

    def spy_init(self, text="answered", words=()):
        original(self, text=text, words=words or ("from ", "google"))
        built.append(self)

    _Answering.__init__ = spy_init
    try:
        routed = RoutedChatModel(
            models=[], options={"api_key": "test"}, fallbacks=two_providers(asked)
        )
        assert routed.bind_tools([lookup]).invoke("a question").content == "from google"
    finally:
        _Answering.__init__ = original

    assert asked["groq"] == ["groq-a"]
    assert any(model.bound_tools and model.bound_tools[0].name == "lookup" for model in built), (
        "tools must reach the model that answered"
    )


def test_structured_output_keeps_the_strict_schema_filter():
    """A ceiling still walks the strict-schema step list, not the full model list."""
    seen: list[str] = []

    def build(model_id: str):
        seen.append(model_id)
        if model_id == "g-a":
            raise groq_413()
        return _StructuredAnswering()

    plan = ProviderPlan(
        name="groq",
        models=["g-a", "g-b", "g-c"],
        build=build,
        strict_structured_output=["g-b"],
    )
    routed = RoutedChatModel(models=[], options={}, fallbacks=[plan])

    result = routed.with_structured_output({"type": "object"}, method="json_schema", strict=True).invoke("q")

    assert result == {"ok": True}
    assert seen == ["g-b"], "only the strict-schema model is asked, and only once"


class _StructuredAnswering:
    def with_structured_output(self, schema, **kwargs):
        class _Inner:
            def invoke(self, input, config=None, **kw):
                return {"ok": True}

        return _Inner()


# --- the security guarantees are untouched ---------------------------------


def test_a_ceiling_still_reaches_the_client_sanitised():
    """Rotating does not change what the client is told, or what the log keeps."""
    import json

    from api.api.endpoints import _as_http_error

    error = groq_413()
    http_error = _as_http_error(error)

    body = json.dumps(http_error.detail)
    assert "org_test" not in body, "the organization id must not reach the client"
    assert "console.groq.com" not in body, "a billing link must not reach the client"
    assert "on_demand" not in body, "the service tier must not reach the client"
    assert "tokens per minute" not in body, "the raw body must not reach the client"
    assert "token" in http_error.detail["error"].lower(), "the caller is still told what happened"
    assert "org_test" in str(describe(error)), "the raw body is still kept server side"
