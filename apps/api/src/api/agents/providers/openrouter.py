"""OpenRouter, the provider that owns the model pool.

Everything provider specific comes from `config.yaml`: the `init_chat_model`
prefix, the base url, and which env var holds the key. Adding a provider, or
switching to one, never means editing code.
"""

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


def active_provider() -> str:
    """Configured provider, or the first one with a key present."""
    settings = get_settings().llm
    if settings.active != "auto":
        return settings.active
    for name in settings.auto_detect:
        if get_settings().provider_api_key(name):
            return name
    raise RuntimeError(
        "No LLM API key found. Set one of "
        + ", ".join(get_settings().llm.key_envs.values())
        + " in .env"
    )


def model_prefix() -> str:
    """The `init_chat_model` prefix, e.g. `openrouter`, from config.yaml."""
    prefix = get_settings().llm.model_prefix
    if not prefix:
        raise KeyError("llm.model_prefix is not set in config.yaml")
    return prefix


def client_options(provider: str, temperature: float | None) -> dict:
    """The options every model in the pool is built with.

    `timeout_seconds` is the only place a unit conversion happens: the OpenRouter
    integration takes its timeout in milliseconds (the openrouter SDK divides
    `timeout_ms` by 1000 before handing it to httpx), so the seconds written in
    config.yaml are converted here and nowhere else.
    """
    settings = get_settings().llm
    options: dict[str, Any] = {
        "temperature": settings.temperature if temperature is None else temperature,
        "timeout": int(settings.timeout_seconds * 1000),
        "max_retries": settings.max_retries,
    }
    if settings.max_tokens:
        # Free tiers reject oversized max_tokens, so this stays bounded.
        options["max_tokens"] = settings.max_tokens
    base_url = settings.base_urls.get(provider)
    if base_url:
        options["base_url"] = base_url
    # init_chat_model reads the api key from the provider's own env var.
    if get_settings().provider_api_key_env(provider):
        options["api_key"] = get_settings().provider_api_key(provider)
    return options


def build(model_id: str, provider: str = None, temperature: float = None, **overrides: Any) -> BaseChatModel:
    """One concrete model, addressed by its configured id.

    `overrides` carries the client options, so the router can rebuild the same
    configured model for each id it falls back to.
    """
    provider = provider or active_provider()
    options = client_options(provider, temperature)
    options.update(overrides)
    return init_chat_model(f"{model_prefix()}:{model_id}", **options)
