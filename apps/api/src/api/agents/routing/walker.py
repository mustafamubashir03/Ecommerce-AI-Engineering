"""Walking the steps until one answers.

The decision made after a failure is written once, in `FailureWalk.after`, and
the synchronous and asynchronous drivers are thin loops around it. They used to
be two copies of the same policy, which is how the two drift apart.
"""

import logging
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from api.agents.provider_errors import (
    claim_failure_walk,
    describe,
    is_account_quota,
    record_pool_failure,
    release_failure_walk,
    share_failure,
    trip,
)
from api.agents.routing.policy import (
    NEXT_PROVIDER,
    PRIMARY,
    RAISE,
    ROTATE,
    action,
    cooldown_wait,
    next_provider_index,
)
from api.agents.routing.steps import Step

logger = logging.getLogger(__name__)

NO_MODEL_MESSAGE = (
    "No LLM model is configured. Set OPENROUTER_MODELS (and optionally "
    "GROQ_MODELS) in .env, or llm.pool in config.yaml."
)


class FailureWalk:
    """One pass over the steps, and the decision taken after each failure.

    Holds the claim on the shared walk, so concurrent requests hitting the same
    provider refusal only pay for one discovery of it.
    """

    def __init__(self) -> None:
        self._claimed = claim_failure_walk()

    def after(
        self, error: BaseException, steps: List[Step], index: int
    ) -> Tuple[str, int, Optional[BaseException]]:
        """Decide what a failure means. Returns (kind, index, to_raise).

        `kind` is "next" to keep walking at `index`, or "raise" to stop. The
        exception to raise is returned rather than raised here, so the caller
        decides whether it is inside a `try` that still has to release the walk.
        """
        step = steps[index]
        last_step = index == len(steps) - 1
        openrouter = step.provider == PRIMARY
        more_in_provider = not last_step and steps[index + 1].provider == step.provider
        action_taken = action(
            error, openrouter=openrouter, last_of_provider=not more_in_provider, last_step=last_step
        )

        if action_taken is RAISE or action_taken == RAISE:
            return "raise", index, record_pool_failure(error) if openrouter and last_step else None

        # An account quota is the account's, not the model's, so hold the whole
        # provider off instead of paying for the same refusal on every model.
        if openrouter and is_account_quota(error):
            trip(error, cooldown_wait(error))
        elif openrouter and last_step:
            record_pool_failure(error)

        if not self._claimed and openrouter:
            # Another request is already walking the pool for the same reason.
            # Wait for its verdict rather than doubling the load.
            if share_failure(error) is not None:
                # That walk already decided this provider is down, so skip it
                # instead of paying for the same walk again.
                following = next_provider_index(steps, index)
                if following >= len(steps):
                    return "raise", index, None
                return "next", following, None
            self._claimed = claim_failure_walk()

        # ROTATE stays inside the provider; NEXT_PROVIDER skips the rest of it,
        # so a known account quota is not paid for again.
        following = index + 1 if action_taken is ROTATE or action_taken == ROTATE else next_provider_index(steps, index)
        if following >= len(steps):
            # Nothing left to try, so this failure is the answer.
            return "raise", index, record_pool_failure(error) if openrouter else None

        logger.warning(
            "%s/%s failed with %s. Trying the next %s.",
            step.provider,
            step.model_id,
            describe(error),
            "model" if (action_taken is ROTATE or action_taken == ROTATE) else "provider",
        )
        return "next", following, None

    def release(self) -> None:
        release_failure_walk()


def first_success(steps: List[Step], attempt: Callable[[Step], Any]) -> Any:
    """Try each step in order, using the shared policy for every failure."""
    if not steps:
        raise RuntimeError(NO_MODEL_MESSAGE)

    walk = FailureWalk()
    try:
        index = 0
        while index < len(steps):
            try:
                return attempt(steps[index])
            except Exception as error:
                kind, index, to_raise = walk.after(error, steps, index)
                if kind == "raise":
                    if to_raise is not None:
                        raise to_raise from error
                    raise
    finally:
        walk.release()
    raise AssertionError("unreachable")


async def first_success_async(
    steps: List[Step], attempt: Callable[[Step], Awaitable[Any]]
) -> Any:
    """The asynchronous twin of `first_success`, on the same policy.

    Each attempt is awaited, so a provider call never blocks the event loop.
    """
    if not steps:
        raise RuntimeError(NO_MODEL_MESSAGE)

    walk = FailureWalk()
    try:
        index = 0
        while index < len(steps):
            try:
                return await attempt(steps[index])
            except Exception as error:
                kind, index, to_raise = walk.after(error, steps, index)
                if kind == "raise":
                    if to_raise is not None:
                        raise to_raise from error
                    raise
    finally:
        walk.release()
    raise AssertionError("unreachable")
