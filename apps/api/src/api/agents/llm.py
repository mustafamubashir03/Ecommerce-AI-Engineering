"""The chat model the agents use, and the providers behind it.

A thin seam over `agents.providers`, kept because the agents, the graph and the
tests all import from here. Building a concrete model, building the fallback
plans, and handing back the one routed model are the only three things here; how
a provider is reached lives in `providers/`, and how a failed attempt is handled
lives in `routing/`.
"""

import logging
from typing import TYPE_CHECKING, List

from langchain_core.language_models import BaseChatModel

from api.agents.providers import google, groq, openrouter
from api.core.settings import get_settings

if TYPE_CHECKING:  # the router imports this module, so only the type is needed
    from api.agents.routing import ProviderPlan

logger = logging.getLogger(__name__)

_cache: dict = {}


# --- the pool's provider ----------------------------------------------------


def build_chat_model(
    model_id: str,
    provider: str = None,
    temperature: float = None,
    **overrides,
) -> BaseChatModel:
    """One concrete model from the pool's provider."""
    return openrouter.build(model_id, provider, temperature, **overrides)


# --- the fallback providers -------------------------------------------------


def build_google_model() -> BaseChatModel | None:
    """The Google fallback, or None when it is not usable."""
    return google.build()


def build_groq_model(model_id: str) -> BaseChatModel | None:
    """One Groq model, or None when Groq is not usable."""
    return groq.build(model_id)


def build_fallback_providers() -> "List[ProviderPlan]":
    """The non OpenRouter providers, in the configured order.

    Each one is a small plan rather than a second routing implementation: an
    ordered model list plus the one function that builds a model from an id.
    """
    from api.agents.routing import ProviderPlan

    settings = get_settings().llm
    plans: "List[ProviderPlan]" = []

    for name in settings.fallback_order:
        if name == "groq":
            pool = settings.groq.model_pool()
            if pool and settings.groq.enabled:
                plans.append(
                    ProviderPlan(
                        name="groq",
                        models=pool,
                        build=build_groq_model,
                        strict_structured_output=list(settings.groq.strict_structured_output),
                    )
                )
        elif name == "google":
            model = build_google_model()
            if model is not None:
                plans.append(ProviderPlan(name="google", models=["*"], build=lambda _id, m=model: m))

    return plans


# --- what the agents get ----------------------------------------------------


def get_chat_model(provider: str = None, temperature: float = None) -> BaseChatModel:
    """The chat model the agents use.

    Returns a model that walks the configured pool of OpenRouter models, so the
    callers keep working with one model and never hardcode a model id.
    """
    from api.agents.model_router import RoutedChatModel

    provider = provider or openrouter.active_provider()
    settings = get_settings().llm
    temperature = settings.temperature if temperature is None else temperature

    cache_key = (provider, temperature)
    if cache_key in _cache:
        return _cache[cache_key]

    model = RoutedChatModel(
        models=get_settings().model_pool(),
        options=openrouter.client_options(provider, temperature),
        fallbacks=build_fallback_providers(),
    )
    _cache[cache_key] = model
    return model


def provider_label() -> str:
    """Human readable description of what the agents are talking to."""
    pool = get_settings().model_pool()
    return f"openrouter: {len(pool)} models, primary {pool[0]}" if pool else "openrouter: no model"
