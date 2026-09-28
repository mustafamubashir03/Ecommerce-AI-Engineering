"""Groq, reached through its OpenAI compatible endpoint.

No Groq SDK: the project already depends on the OpenAI client and Groq speaks
that protocol, so the same client is pointed at Groq's own base url. Nothing is
ever sent to OpenAI directly.
"""

import logging
import os

from langchain_core.language_models import BaseChatModel

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


def _key_or_warn(api_key_env: str) -> str | None:
    """The configured key, or None with the reason logged once per build."""
    api_key = (os.getenv(api_key_env) or "").strip()
    if not api_key:
        logger.warning(
            "groq fallback is enabled but %s is not set, so it is unavailable.",
            api_key_env,
        )
    return api_key or None


def build(model_id: str) -> BaseChatModel | None:
    """One Groq model, or None when Groq is not usable.

    Deliberately optional: a missing key or a disabled block means there is no
    fallback, never a startup failure.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings().llm
    groq = settings.groq
    if not groq.enabled:
        return None

    api_key = _key_or_warn(groq.api_key_env)
    if not api_key:
        return None

    return ChatOpenAI(
        model=model_id,
        base_url=groq.base_url,
        api_key=api_key,
        temperature=settings.temperature,
        timeout=groq.timeout_seconds,
        max_retries=settings.max_retries,
    )
