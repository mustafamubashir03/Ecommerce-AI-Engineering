"""What to do about one failed attempt.

The whole fallback policy of the application is these few decisions, kept
apart from the model plumbing so it can be read, and changed, on its own.

    OpenRouter model pool  ->  next provider's models  ->  next provider  ->  stop

Each provider is tried once, in its configured order, and nothing routes back,
so there is no recursion. A provider is only skipped forward for an availability
failure; a bad request, a bad key or an unsupported feature is surfaced instead
of being retried somewhere that cannot do any better.
"""

from typing import TYPE_CHECKING, List

from api.agents.errors import is_account_quota, is_fallback_worthy, wait_for
from api.core.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from api.agents.routing.steps import Step

# The provider that owns the model pool, and the one whose refusals are shared.
PRIMARY = "openrouter"

# What to do about a failure.
ROTATE = "rotate"  # same provider, next model
NEXT_PROVIDER = "next_provider"  # this provider is spent, move on
RAISE = "raise"  # surface it, retrying cannot help


def action(
    error: BaseException, *, openrouter: bool, last_of_provider: bool, last_step: bool
) -> str:
    """What to do about one failed attempt. The whole policy lives here.

    * OpenRouter account quota -> move to the next provider. Rotating through the
      pool would only rediscover the same refusal, so the caller records a
      cooldown instead.
    * availability failure      -> rotate inside the provider, and once the
      provider is spent, move to the next one.
    * application/request error -> surface. A 400, 401, 403, 404, 413 or 422
      fails the same way on every other provider, and retrying hides the cause.
    """
    if openrouter and is_account_quota(error):
        return NEXT_PROVIDER
    if not is_fallback_worthy(error):
        return RAISE
    if last_step:
        return RAISE
    return NEXT_PROVIDER if last_of_provider else ROTATE


def cooldown_wait(error: BaseException) -> float:
    """How long to hold a provider off, from its own retry headers."""
    return wait_for(error, get_settings().llm.retries.backoff_seconds)


def next_provider_index(steps: List["Step"], index: int) -> int:
    """The first step belonging to the next provider, or the end of the list."""
    provider = steps[index].provider
    for position in range(index + 1, len(steps)):
        if steps[position].provider != provider:
            return position
    return len(steps)
