"""Google, the external fallback.

One subclass is needed, and only for the call configuration: Google's SDK runs a
bound tool itself, loops, and returns only the final text. The graph then never
sees a `ToolMessage`, so the products the tool found are lost and the answer
carries no citations. This application runs the tool loop in `create_agent`, so
Google is asked to return the tool call instead of executing it.
"""

import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


class AgentOwnedToolLoop(ChatGoogleGenerativeAI):
    """Google with automatic function calling switched off.

    Each call path has to carry the opt out, and each one delegates to its
    parent in the parent's own shape: a generator with `yield from`, an async
    generator with `async for`, and a coroutine with `await`. Awaiting an async
    generator raises, so the shape has to match the method being overridden.
    """

    _NO_AFC = {"disable": True}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["automatic_function_calling"] = self._NO_AFC
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["automatic_function_calling"] = self._NO_AFC
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["automatic_function_calling"] = self._NO_AFC
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs["automatic_function_calling"] = self._NO_AFC
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


def build() -> BaseChatModel | None:
    """The Google fallback, or None when it is not usable.

    The key is read from the environment variable named in config.yaml, which is
    `GOOGLE_API_KEY`. This integration takes its timeout in seconds, the same
    unit config.yaml uses, so unlike OpenRouter nothing is converted here.
    """
    settings = get_settings().llm
    google = settings.google
    if not google.enabled or not google.model:
        return None

    api_key = (os.getenv(google.api_key_env) or "").strip()
    if not api_key:
        logger.warning(
            "google fallback is enabled but %s is not set, so it is unavailable.",
            google.api_key_env,
        )
        return None

    return AgentOwnedToolLoop(
        model=google.model,
        google_api_key=api_key,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        timeout=google.timeout_seconds,
    )
