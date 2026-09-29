"""Stands in for the OpenRouter SDK, so the routing tests never touch the network.

The fakes reproduce the real shape of what OpenRouter returns, taken from the
live API rather than invented: the status code, the JSON error body, and the
rate limit headers, all as attributes on the raised exception.
"""

import json
from typing import Any, Iterator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# The real body OpenRouter returned for an exhausted free daily cap.
DAILY_CAP_BODY = {
    "error": {
        "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
        "code": 429,
        "metadata": {
            "headers": {
                "X-RateLimit-Limit": "50",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1790640000000",
            },
            "limit_source": "openrouter_free_tier_daily",
            "remedy_hint": "Wait for the daily reset, or purchase credits to raise your free-model daily limit.",
            "provider_name": None,
        },
    },
    "user_id": "user_test",
}

# A 429 that belongs to one provider rather than to the account.
PROVIDER_LIMIT_BODY = {
    "error": {
        "message": "Provider is temporarily rate-limited upstream",
        "code": 429,
        "metadata": {"limit_source": "provider", "provider_name": "Some-Provider"},
    }
}


class FakeOpenRouterError(Exception):
    """Shaped like the openrouter SDK's error classes: status, body, headers."""

    def __init__(
        self,
        status_code: int,
        body: Any,
        headers: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.body = body if isinstance(body, str) else json.dumps(body)
        self.headers = headers or {}
        super().__init__(f"Error code: {status_code} - {self.body}")


def daily_cap_error() -> FakeOpenRouterError:
    return FakeOpenRouterError(
        429,
        DAILY_CAP_BODY,
        {"x-ratelimit-limit": "50", "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1790640000000"},
    )


def provider_limit_error() -> FakeOpenRouterError:
    return FakeOpenRouterError(429, PROVIDER_LIMIT_BODY, {"retry-after": "2"})


def server_error(status: int = 503) -> FakeOpenRouterError:
    return FakeOpenRouterError(status, {"error": {"message": "upstream unavailable"}})


def bad_request_error() -> FakeOpenRouterError:
    return FakeOpenRouterError(400, {"error": {"message": "invalid request"}})


def auth_error() -> FakeOpenRouterError:
    return FakeOpenRouterError(401, {"error": {"message": "No auth credentials found"}})


def payment_error() -> FakeOpenRouterError:
    return FakeOpenRouterError(402, {"error": {"message": "Insufficient credits"}})


def timeout_error() -> TimeoutError:
    return TimeoutError("the request timed out")


class CallLog:
    """Shared record of which model each attempt used.

    A plain object rather than a list, because pydantic copies list fields when
    it validates a model, which would leave the test's own list untouched.
    """

    def __init__(self) -> None:
        self.models: list[str] = []

    def __eq__(self, other) -> bool:
        return self.models == other

    def __repr__(self) -> str:
        return f"CallLog({self.models})"


class ScriptedModel(BaseChatModel):
    """A model that fails a set number of times, then answers.

    `script` maps a model id to a list of actions, each either an Exception to
    raise or the text to answer with. An empty list means the model is healthy.
    """

    script: dict = {}
    log: Any = None
    model_id: str = ""
    tokens_before_failure: int = 0
    stream_text: str = "answer"

    @property
    def _llm_type(self) -> str:
        return "scripted-openrouter"

    def _next(self, model_id: str):
        self.log.models.append(model_id)
        steps = self.script.get(model_id) or []
        return steps.pop(0) if steps else None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        action = self._next(self.model_id)
        if isinstance(action, Exception):
            raise action
        text = self.stream_text if action is None else str(action)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        action = self._next(self.model_id)
        if isinstance(action, Exception):
            if self.tokens_before_failure:
                # Emit tokens, then fail: the caller has already sent output.
                for piece in self.stream_text.split(" ")[: self.tokens_before_failure]:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece + " "))
            raise action
        text = self.stream_text if action is None else str(action)
        for word in text.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptedStructuredModel(ScriptedModel):
    """A model whose `with_structured_output` honours the same script."""

    schema_result: Any = None

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class _Structured:
            def invoke(self, input, config=None, **kw):
                action = outer._next(outer.model_id)
                if isinstance(action, Exception):
                    raise action
                return outer.schema_result

        return _Structured()


def install(
    monkeypatch,
    script: dict,
    tokens_before_failure: int = 0,
    stream_text: str = "answer",
    structured_factory=None,
    schema_result: Any = None,
) -> CallLog:
    """Point the router at scripted models. Returns the shared call log."""
    from api.agents import model_router

    log = CallLog()

    def fake_build(model_id: str, **options) -> ScriptedModel:
        factory = structured_factory if structured_factory else ScriptedModel
        extra = {"schema_result": schema_result} if structured_factory else {}
        return factory(
            script={model_id: list(script.get(model_id, []))},
            log=log,
            model_id=model_id,
            tokens_before_failure=tokens_before_failure,
            stream_text=stream_text,
            **extra,
        )

    # The router owns the pool's client construction, so this is where the
    # scripted models have to go in.
    monkeypatch.setattr(model_router, "build_chat_model", fake_build)
    return log


class ToolCallingModel(BaseChatModel):
    """A model that calls the retrieval tool once, then answers from it.

    Used by the contract tests so the graph, the tool and the response shape
    are exercised for real while the provider is not.
    """

    answer: str = "Here is what we stock."
    cite: bool = True
    fail_with: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-caller"

    def _wanted_a_tool(self, messages) -> bool:
        """Call the tool once per turn, whatever the history length is.

        Only the messages after the newest human message matter, so a
        checkpointed conversation from an earlier run cannot change the answer.
        """
        last_human = max(
            (index for index, message in enumerate(messages) if message.type == "human"),
            default=-1,
        )
        return not any(message.type == "tool" for message in messages[last_human + 1 :])

    def _answer_text(self, messages) -> str:
        if not self.cite:
            return self.answer
        from langchain_core.messages import ToolMessage

        ids: list[str] = []
        for message in messages:
            if isinstance(message, ToolMessage) and isinstance(message.content, str):
                for piece in message.content.split("["):
                    if "]" in piece:
                        ids.append(piece.split("]")[0])
        return f"{self.answer} " + " ".join(f"[{value}]" for value in ids)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.fail_with is not None:
            raise self.fail_with
        if self._wanted_a_tool(messages):
            call = AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_data_tool", "args": {"query": "washing machine"}, "id": "c1"}
                ],
            )
        else:
            call = AIMessage(content=self._answer_text(messages))
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools, **kwargs):
        return self
