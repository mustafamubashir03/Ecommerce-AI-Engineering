"""How a request finds a model that can answer it.

Kept apart from the chat model in `model_router`, so the fallback policy can be
read without reading the LangChain plumbing, and changed without touching it.
"""

from api.agents.routing.policy import (
    NEXT_PROVIDER,
    PRIMARY,
    RAISE,
    ROTATE,
    action,
    cooldown_wait,
    next_provider_index,
)
from api.agents.routing.steps import (
    ProviderPlan,
    Step,
    build_steps,
    steps_for_call,
    wants_strict_schema,
)
from api.agents.routing.streaming import astream, stream
from api.agents.routing.structured import RoutedStructuredOutput
from api.agents.routing.walker import (
    NO_MODEL_MESSAGE,
    after_failure,
    first_success,
    first_success_async,
)

__all__ = [
    "NEXT_PROVIDER",
    "NO_MODEL_MESSAGE",
    "PRIMARY",
    "RAISE",
    "ROTATE",
    "after_failure",
    "ProviderPlan",
    "RoutedStructuredOutput",
    "Step",
    "action",
    "astream",
    "build_steps",
    "cooldown_wait",
    "first_success",
    "first_success_async",
    "next_provider_index",
    "stream",
    "steps_for_call",
    "wants_strict_schema",
]
