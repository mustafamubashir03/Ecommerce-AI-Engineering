import re
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool


class FakeProviderError(Exception):
    """A provider failure shaped like the real SDK's, with a body attached."""

    def __init__(self, message: str, status_code: int = 500, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = httpx.Response(
            status_code,
            text=body,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )


def provider_error(status: int = 500, message: str = "upstream unavailable", body: dict | None = None):
    """A provider failure carrying a body, as a real one does."""
    payload = body if body is not None else {"error": {"code": status, "message": message}}
    import json

    return FakeProviderError(f"Error code: {status}", status, json.dumps(payload))


def server_error(status: int = 503):
    return provider_error(status)


def daily_cap_error():
    return provider_error(429, body={"error": {"code": 429, "message": "free-models-per-day"}})


def payload_too_large_error():
    return provider_error(413, body={"error": {"code": 413, "message": "Request payload too large"}})


class ScriptedModel(BaseChatModel):
    """A model that fails a set number of times, then answers."""

    script: dict = {}
    calls: list = []
    model_id: str = ""
    tokens_before_failure: int = 0
    stream_text: str = "answer"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, model_id: str):
        self.calls.append(model_id)
        steps = self.script.get(model_id) or []
        return steps.pop(0) if steps else None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        action = self._next(self.model_id)
        if isinstance(action, Exception):
            raise action
        text = self.stream_text if action is None else str(action)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        action = self._next(self.model_id)
        if isinstance(action, Exception):
            if self.tokens_before_failure:
                for piece in self.stream_text.split(" ")[: self.tokens_before_failure]:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece + " "))
            raise action
        text = self.stream_text if action is None else str(action)
        for word in text.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    def bind_tools(self, tools, **kwargs):
        return self


class ToolCallingModel(BaseChatModel):

    calls: list = []
    answer: str = "Here are the machines we have in stock."
    cite: bool = True

    @property
    def _llm_type(self) -> str:
        return "tool-calling"

    def _wanted_a_tool(self, messages) -> bool:
        return not any(getattr(message, "type", "") == "tool" for message in messages)

    def _citations(self, messages) -> str:
        if not self.cite:
            return ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "tool":
                ids = re.findall(r"\[([A-Z0-9]{10})\]", message.content or "")
                return " " + " ".join(f"[{product_id}]" for product_id in ids) if ids else ""
        return ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self._wanted_a_tool(messages):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "retrieve_data_tool", "args": {"query": "washing machines"}, "id": "call_1"}
                            ],
                        )
                    )
                ]
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer + self._citations(messages)))]
        )

    def bind_tools(self, tools, **kwargs):
        return self


def install(monkeypatch, script: dict, tokens_before_failure: int = 0, stream_text: str = "answer") -> list:
    """Point the app at a scripted model. Returns the shared call log."""
    import api.agents.llm as llm_module
    import api.agents.agent as agent_module

    calls: list = []

    def build(*args: Any, **kwargs: Any) -> ScriptedModel:
        return ScriptedModel(
            script={key: list(value) for key, value in script.items()},
            calls=calls,
            model_id="scripted",
            tokens_before_failure=tokens_before_failure,
            stream_text=stream_text,
        )

    monkeypatch.setattr(llm_module, "get_chat_model", build)
    monkeypatch.setattr(agent_module, "get_chat_model", build)
    monkeypatch.setattr(agent_module, "get_instructor_client", lambda: _Structured(build()))
    return calls


class _Structured:
    """A `with_structured_output` shaped object standing in for instructor."""

    def __init__(self, model):
        self._model = model

    def with_structured_output(self, schema, **kwargs):
        class _Inner:
            def invoke(self, input, config=None, **kw):
                from api.agents.prompts import IntentRouterResponse

                return IntentRouterResponse(question_relevancy=True, answer="")

            async def ainvoke(self, input, config=None, **kw):
                return self.invoke(input)

        return _Inner()


@tool
def retrieve_data_tool(query: str) -> str:
    """Search the product catalogue for products in stock."""
    return "[B0TEST0001] (rating: 4.0) A test product."
