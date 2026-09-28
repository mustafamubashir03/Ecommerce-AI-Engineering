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

from api.agents.provider_errors import describe, record_pool_failure, trip
from api.agents.routing.policy import PRIMARY, RAISE, action, cooldown_wait
from api.agents.routing.steps import Step

logger = logging.getLogger(__name__)


def _after_no_output(error: BaseException, step: Step) -> None:
    """Handle a failure that happened before any token was emitted.

    Raises when the failure is one retrying cannot fix, and records the pool
    failure, so this is the only place that decision is made for streaming.
    """
    if action(
        error, openrouter=step.provider == PRIMARY, last_of_provider=True, last_step=False
    ) == RAISE:
        raise error
    if step.provider == PRIMARY:
        record_pool_failure(error)
    logger.warning(
        "%s/%s produced no output before failing (%s). Trying the next provider.",
        step.provider,
        step.model_id,
        describe(error),
    )


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
    for step in steps:
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
            _after_no_output(error, step)


async def astream(
    steps: List[Step],
    delegate: Any,
    messages: List[Any],
    stop: Any,
    run_manager: Any,
    kwargs: dict,
) -> AsyncIterator[ChatGenerationChunk]:
    """The asynchronous twin of `stream`, on the same no-splice rule."""
    for step in steps:
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
            _after_no_output(error, step)
