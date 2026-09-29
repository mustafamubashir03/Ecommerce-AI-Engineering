"""The asynchronous router paths and the timeout that reaches the client."""

import asyncio

import pytest

import fake_openrouter as fake
from api.agents.model_router import RoutedChatModel

POOL = ["model/a", "model/b", "model/c"]


def routed(pool=POOL) -> RoutedChatModel:
    return RoutedChatModel(models=list(pool), options={"api_key": "test"})


# --- the async path must never call the synchronous one --------------------


def test_async_structured_output_calls_ainvoke(monkeypatch):
    """A 429 on the async path must be reached through `.ainvoke`."""
    calls: list[str] = []

    def build(model_id: str, **options):
        class Structured:
            def invoke(self, *args, **kwargs):
                calls.append("invoke")
                raise AssertionError("the sync path was used on an async call")

            async def ainvoke(self, *args, **kwargs):
                calls.append("ainvoke")
                if model_id == "model/a":
                    raise fake.provider_limit_error()
                return {"question_relevancy": True, "answer": "", "via": model_id}

        class Model:
            def with_structured_output(self, schema, **kw):
                return Structured()

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    result = asyncio.run(routed().with_structured_output(dict).ainvoke([{"role": "user", "content": "hi"}]))
    assert result["via"] == "model/b", "the fallback model answered"
    assert "invoke" not in calls, "no synchronous call was made on the async path"
    assert calls.count("ainvoke") == 2, "both attempts went through ainvoke"


def test_async_fallback_stops_on_account_quota(monkeypatch):
    """Same policy on the async path: a quota refusal stops the walk."""
    calls: list[str] = []

    def build(model_id: str, **options):
        class Structured:
            def invoke(self, *args, **kwargs):
                raise AssertionError("the sync path was used on an async call")

            async def ainvoke(self, *args, **kwargs):
                calls.append(model_id)
                raise fake.daily_cap_error()

        class Model:
            def with_structured_output(self, schema, **kw):
                return Structured()

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    with pytest.raises(Exception):
        asyncio.run(routed().with_structured_output(dict).ainvoke([{"role": "user", "content": "hi"}]))
    assert calls == ["model/a"], "account quota must not walk the pool on the async path either"


def test_agenerate_awaits_the_delegate(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    calls: list[str] = []

    def build(model_id: str, **options):
        class Model:
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                calls.append(model_id)
                if model_id == "model/a":
                    raise fake.server_error(503)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="async answer"))])

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)
    result = asyncio.run(routed().ainvoke([HumanMessage(content="hello")]))
    assert result.content == "async answer"
    assert calls == ["model/a", "model/b"]


def test_agenerate_does_not_block_the_event_loop(monkeypatch):
    """The loop stays responsive while the provider call is awaited."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    ticks = 0

    def build(model_id: str, **options):
        class Model:
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                await asyncio.sleep(0.2)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="slow"))])

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    counter = {"ticks": 0}

    async def main():
        async def ticker():
            for _ in range(20):
                await asyncio.sleep(0.01)
                counter["ticks"] += 1

        await asyncio.gather(routed().ainvoke([HumanMessage(content="hi")]), ticker())

    asyncio.run(main())
    ticks = counter["ticks"]
    assert ticks >= 15, f"the loop only ran {ticks} times while awaiting the model"


def test_astream_does_not_splice_after_output(monkeypatch):
    calls: list[str] = []

    def build(model_id: str, **options):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        class Model:
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                calls.append(model_id)
                yield ChatGenerationChunk(message=AIMessageChunk(content="one two "))
                raise fake.server_error(503)

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    emitted = []

    async def main():
        async for chunk in routed().astream("hello"):
            emitted.append(chunk.content)

    with pytest.raises(Exception):
        asyncio.run(main())
    assert calls == ["model/a"], "no second model after partial output"
    assert "".join(emitted) == "one two "


def test_astream_falls_back_before_any_output(monkeypatch):
    calls: list[str] = []

    def build(model_id: str, **options):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        class Model:
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                calls.append(model_id)
                if model_id == "model/a":
                    raise fake.server_error(503)
                yield ChatGenerationChunk(message=AIMessageChunk(content="from b"))

        return Model()

    monkeypatch.setattr("api.agents.model_router.build_chat_model", build)

    async def main():
        return "".join([chunk.content async for chunk in routed().astream("hello")])

    assert asyncio.run(main()) == "from b"
    assert calls == ["model/a", "model/b"]


def test_synchronous_paths_are_unchanged(monkeypatch):
    """The sync entry points still use the sync delegate methods."""
    fake.install(monkeypatch, {"model/a": []})
    assert routed().invoke("hello").content == "answer"
    streamed = "".join(chunk.content for chunk in routed().stream("hello"))
    assert streamed.strip() == "answer"


# --- the timeout that actually reaches the client --------------------------


def test_configured_timeout_reaches_the_client_as_seconds(monkeypatch):
    """config.yaml says seconds; the client must get seconds, not milliseconds."""
    import api.agents.llm as llm
    from api.agents.providers import openrouter
    from api.core.settings import get_settings

    seen: dict = {}

    def fake_init(target: str, **options):
        seen.update(options)
        return object()

    monkeypatch.setattr(openrouter, "init_chat_model", fake_init)
    llm.build_chat_model("poolside/laguna-s-2.1:free")

    settings = get_settings().llm
    assert settings.timeout_seconds == 90.0, "config.yaml is written in seconds"
    # The OpenRouter integration takes milliseconds; the SDK divides by 1000
    # before handing the value to httpx, so this must be the converted value.
    assert seen["timeout"] == 90_000


def test_no_stale_second_based_timeout_key_remains():
    from api.core.settings import get_settings

    fields = type(get_settings().llm).model_fields
    assert "timeout" not in fields, "the ambiguous `timeout` key must be gone"
    assert "timeout_units" not in fields, "per provider units are no longer configured"
    assert "timeout_seconds" in fields


def _read_timeout(value):
    """The read timeout httpx will enforce, whatever shape it stored."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("read")
    if hasattr(value, "read"):
        return value.read
    return float(value)


def test_actual_client_timeout_is_ninety_seconds(monkeypatch):
    """Read the built request, not the config, to prove the final value.

    The chain is: config.yaml `timeout_seconds` -> `_client_options` converts to
    milliseconds -> ChatOpenRouter maps it to the SDK's `timeout_ms` -> the SDK
    divides by 1000 and hands it to httpx's `build_request`, which stores it on
    the request's timeout extension. That last value is the one httpx enforces.
    """
    import httpx

    from api.agents.llm import build_chat_model

    model = build_chat_model("poolside/laguna-s-2.1:free")
    assert type(model).__name__ == "ChatOpenRouter"

    seen: dict = {}

    def spy(self, request, **kwargs):
        seen["timeout"] = request.extensions.get("timeout")
        raise RuntimeError("no network in this test")

    monkeypatch.setattr(httpx.Client, "send", spy)
    with pytest.raises(RuntimeError):
        model.invoke("hello")

    seconds = _read_timeout(seen.get("timeout"))
    print(f"\n  configured 90 s -> httpx read timeout {seconds} s")
    assert seconds == 90.0, f"expected 90 seconds at the client, got {seconds!r}"


def test_timeout_is_not_the_httpx_default(monkeypatch):
    """Guard against the timeout silently reverting to httpx's 5 second default."""
    import httpx

    from api.agents.llm import build_chat_model

    model = build_chat_model("poolside/laguna-s-2.1:free")
    seen: dict = {}

    def spy(self, request, **kwargs):
        seen["timeout"] = request.extensions.get("timeout")
        raise RuntimeError("stop")

    monkeypatch.setattr(httpx.Client, "send", spy)
    with pytest.raises(RuntimeError):
        model.invoke("hello")

    seconds = _read_timeout(seen["timeout"])
    assert seconds not in (None, 5.0), f"the timeout fell back to httpx's default ({seconds})"


def test_timeout_conversion_happens_once():
    """One conversion, at the factory. No other module scales the value."""
    import inspect

    import api.agents.llm as llm
    import api.agents.model_router as router

    for module in (llm, router):
        source = inspect.getsource(module)
        assert "timeout_seconds * 1000" in source or "timeout_seconds" not in source, (
            f"{module.__name__} scales the timeout itself"
        )
