"""Streaming across the pool, with one rule: no handover after output.

A provider that fails before its first token can be replaced, so the next one
streams instead. Once tokens have been sent they belong to that provider:
switching would mean concatenating two answers into one response, so the failure
is raised and the client sees a truncated stream it can retry, never a spliced
one. That holds across providers too, so an OpenRouter prefix and a Groq answer
can never end up in the same stream.
"""

import logging
from typing import Any, AsyncIterator, Iterator, List

from langchain_core.outputs import ChatGenerationChunk

from api.agents.errors import describe, record_pool_failure, trip
from api.agents.routing.policy import (
    PRIMARY,
    RAISE,
    ROTATE,
    action,
    cooldown_wait,
    next_provider_index,
)
from api.agents.routing.steps import Step

logger = logging.getLogger(__name__)


def _after_no_output(error: BaseException, step: Step, steps: List[Step], index: int) -> int:
    """Handle a failure that happened before any token was emitted.

    Raises when the failure is one retrying cannot fix, and otherwise returns the
    step to try next. The decision comes from the same `action` call and with the
    same inputs the walker uses, so a NEXT_PROVIDER verdict skips the rest of this
    provider's models here exactly as it does there.
    """
    last_step = index == len(steps) - 1
    more_in_provider = not last_step and steps[index + 1].provider == step.provider
    taken = action(
        error,
        openrouter=step.provider == PRIMARY,
        last_of_provider=not more_in_provider,
        last_step=last_step,
    )
    if taken == RAISE:
        raise error
    if step.provider == PRIMARY:
        record_pool_failure(error)
    logger.warning(
        "%s/%s produced no output before failing (%s). Trying the next %s.",
        step.provider,
        step.model_id,
        describe(error),
        "model" if taken == ROTATE else "provider",
    )
    return index + 1 if taken == ROTATE else next_provider_index(steps, index)


def _after_partial_output(error: BaseException) -> None:
    """Tokens are already on the wire, so this stream is now that provider's.

    The failure is held so the next request does not retry it immediately, and
    then surfaced, rather than being papered over with another provider's text.
    """
    raise trip(error, cooldown_wait(error))


def stream(
    steps: List[Step],
    delegate: Any,
    messages: List[Any],
    stop: Any,
    run_manager: Any,
    kwargs: dict,
) -> Iterator[ChatGenerationChunk]:
    """Yield chunks from the first step that produces any output."""
    index = 0
    while index < len(steps):
        step = steps[index]
        model, tool_kwargs = delegate(step)
        started = False
        try:
            for chunk in model._stream(
                messages, stop=stop, run_manager=run_manager, **{**tool_kwargs, **kwargs}
            ):
                started = True
                yield chunk
            return
        except Exception as error:
            if started:
                _after_partial_output(error)
            index = _after_no_output(error, step, steps, index)
    raise AssertionError("unreachable")


async def astream(
    steps: List[Step],
    delegate: Any,
    messages: List[Any],
    stop: Any,
    run_manager: Any,
    kwargs: dict,
) -> AsyncIterator[ChatGenerationChunk]:
    """The asynchronous twin of `stream`, on the same no-splice rule."""
    index = 0
    while index < len(steps):
        step = steps[index]
        model, tool_kwargs = delegate(step)
        started = False
        try:
            async for chunk in model._astream(
                messages, stop=stop, run_manager=run_manager, **{**tool_kwargs, **kwargs}
            ):
                started = True
                yield chunk
            return
        except Exception as error:
            if started:
                _after_partial_output(error)
            index = _after_no_output(error, step, steps, index)
    raise AssertionError("unreachable")
