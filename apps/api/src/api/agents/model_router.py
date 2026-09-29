"""The chat model the agents use: one that moves on when a provider cannot answer.

OpenRouter can rotate models for us with a `models` array in the request
(https://openrouter.ai/docs/guides/routing/model-fallbacks). That was not used
here because it cannot do the three things this application needs:

* it rotates on any error, including the context length and malformed request
  errors that are this application's own fault, which then hides the real cause;
* it cannot tell an exhausted account quota from a single model's rate limit, so
  it would walk the whole pool discovering the same refusal;
* it cannot report which model answered, which the response and the logs need.

So the rotation lives here. This module is only the chat model itself; the
decisions it makes live in the `routing` package next to it:

    `routing.policy`      what a failure means
    `routing.steps`       the ordered attempts
    `routing.walker`      walking them until one answers
    `routing.streaming`   the no-splice rule for streamed answers
    `routing.structured`  structured output, on the same policy
"""

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field

from api.agents.llm import build_chat_model
from api.agents.routing.policy import PRIMARY
from api.agents.routing.steps import ProviderPlan, Step, steps_for_call, wants_strict_schema
from api.agents.routing.streaming import astream, stream
from api.agents.routing.structured import RoutedStructuredOutput
from api.agents.routing.walker import first_success, first_success_async

logger = logging.getLogger(__name__)

__all__ = [
    "PRIMARY",
    "ProviderPlan",
    "RoutedChatModel",
    "RoutedStructuredOutput",
    "Step",
    "build_chat_model",
]


class RoutedChatModel(BaseChatModel):
    """A chat model that moves on when one is unavailable.

    Behaves like the single model it wraps, so `create_agent`,
    `with_structured_output` and the graph's streaming path all work unchanged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: List[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    bound_tools: Optional[list] = None
    bound_tool_kwargs: dict = Field(default_factory=dict)
    # Providers other than OpenRouter, asked in order after the pool is spent.
    fallbacks: List[ProviderPlan] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "routed"

    # --- steps ---------------------------------------------------------------

    def _steps(self, strict_only: bool = False) -> List[Step]:
        return steps_for_call(self.models, self.options, self.fallbacks, build_chat_model, strict_only)

    def _delegate_parts(self, model: BaseChatModel) -> tuple[BaseChatModel, dict]:
        """The raw model plus the call kwargs the agent's tools need.

        `bind_tools` returns a RunnableBinding whose tools are only applied by
        `invoke`. Calling `_generate` or `_stream` on that binding reaches the
        wrapped model with the tools silently dropped, so the agent would run
        with no tools at all. The binding is therefore unwrapped and its kwargs
        are passed to the per step call, where they take effect.
        """
        if self.bound_tools is None:
            return model, {}
        bound = model.bind_tools(self.bound_tools, **self.bound_tool_kwargs)
        return getattr(bound, "bound", model), dict(getattr(bound, "kwargs", {}) or {})

    def _delegate(self, step: Step) -> tuple[BaseChatModel, dict]:
        """Build one step and unwrap its tool binding, in that order."""
        return self._delegate_parts(step.build())

    # --- public chat model surface -------------------------------------------

    def bind_tools(self, tools: list, **kwargs: Any) -> "RoutedChatModel":
        return self.model_copy(
            update={"bound_tools": list(tools), "bound_tool_kwargs": dict(kwargs)}
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return RoutedStructuredOutput(
            steps_for_call(
                self.models,
                self.options,
                self.fallbacks,
                build_chat_model,
                wants_strict_schema(kwargs),
            ),
            schema,
            kwargs,
        )

    # --- one answer ----------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        def attempt(step: Step) -> ChatResult:
            model, tool_kwargs = self._delegate(step)
            return model._generate(
                messages, stop=stop, run_manager=run_manager, **{**tool_kwargs, **kwargs}
            )

        return first_success(self._steps(), attempt)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Await the delegate's own async step, so the loop is never blocked."""

        async def attempt(step: Step) -> ChatResult:
            model, tool_kwargs = self._delegate(step)
            return await model._agenerate(
                messages, stop=stop, run_manager=run_manager, **{**tool_kwargs, **kwargs}
            )

        return await first_success_async(self._steps(), attempt)

    # --- a streamed answer ---------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from stream(self._steps(), self._delegate, messages, stop, run_manager, kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in astream(
            self._steps(), self._delegate, messages, stop, run_manager, kwargs
        ):
            yield chunk
