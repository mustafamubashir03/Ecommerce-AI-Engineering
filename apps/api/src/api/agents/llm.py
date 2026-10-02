import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


_PLACEHOLDER_KEY = "not-needed"


def _api_key() -> str:
    llm = get_settings().llm
    if not llm.api_key_env or not llm.api_key_env.strip():
        return _PLACEHOLDER_KEY

    key = (get_settings().provider_api_key(llm.api_key_env) or "").strip()
    if not key:
        raise RuntimeError(
            f"No LLM API key found. Set {llm.api_key_env} in .env, "
            f"or point llm.api_key_env at the variable that holds it. "
            f"To use a provider that needs no key, leave llm.api_key_env empty."
        )
    return key


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    llm = get_settings().llm
    return ChatOpenAI(
        model=llm.model,
        base_url=llm.base_url,
        api_key=_api_key(),
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        timeout=llm.timeout_seconds,
        max_retries=llm.max_retries,
        streaming=llm.streaming,
    )


@lru_cache(maxsize=1)
def get_instructor_client():
    import instructor
    from openai import OpenAI

    llm = get_settings().llm
    return instructor.from_openai(
        OpenAI(
            base_url=llm.base_url,
            api_key=_api_key(),
            timeout=llm.timeout_seconds,
            max_retries=llm.max_retries,
        ),
        mode=instructor.Mode.TOOLS,
    )


def provider_label() -> str:
    """Human readable description of what the agents are talking to."""
    llm = get_settings().llm
    return f"{llm.model} via {llm.base_url}"
