"""The one chat model the agents talk to.

One client, one model, configured entirely from `config.yaml`. Because it speaks
OpenAI's protocol, almost any provider works with no code change: set `llm.model`,
`llm.base_url` and `llm.api_key_env` and restart. The list of endpoints and the
env var each one uses lives in the `llm` block of `config.yaml`, not here.

`intent_router` needs a structured answer from whatever model is configured, and
not every OpenAI-compatible model supports JSON schema natively, so that one call
goes through `instructor`, which gets it from any of them.
"""

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


def _api_key() -> str:
    """The configured key, or a clear error naming the env var to set."""
    llm = get_settings().llm
    key = (get_settings().provider_api_key(llm.api_key_env) or "").strip()
    if not key:
        raise RuntimeError(
            f"No LLM API key found. Set {llm.api_key_env} in .env, "
            f"or point llm.api_key_env at the variable that holds it."
        )
    return key


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """The chat model, built once and shared.

    `max_retries` is left at the SDK's own default: a single provider means a
    transient failure has nowhere else to go, so retrying it is the only useful
    response, and the SDK already backs off and respects `Retry-After`.
    """
    llm = get_settings().llm
    return ChatOpenAI(
        model=llm.model,
        base_url=llm.base_url,
        api_key=_api_key(),
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        timeout=llm.timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_instructor_client():
    """An instructor client on the same provider, for structured answers.

    Only the intent router needs this. Every other call is an ordinary chat turn
    with tools, which `get_chat_model` already covers. The model is named on the
    request rather than here, because the OpenAI client has no default model.
    """
    import instructor
    from openai import OpenAI

    llm = get_settings().llm
    return instructor.from_openai(
        OpenAI(
            base_url=llm.base_url,
            api_key=_api_key(),
            timeout=llm.timeout_seconds,
        ),
        mode=instructor.Mode.TOOLS,
    )


def provider_label() -> str:
    """Human readable description of what the agents are talking to."""
    llm = get_settings().llm
    return f"{llm.model} via {llm.base_url}"
