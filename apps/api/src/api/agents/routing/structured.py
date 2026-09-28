"""`with_structured_output` across the pool, then across the providers.

A structured call is an ordinary attempt as far as the walker is concerned, so
it gets the same rotation and the same refusal handling as a chat turn. What
makes it different is the strict JSON Schema case: some providers cannot take
it, so those models are filtered out of the step list before the walk starts.
"""

from typing import Any, List

from langchain_core.runnables import Runnable

from api.agents.routing.steps import Step
from api.agents.routing.walker import first_success, first_success_async


class RoutedStructuredOutput(Runnable):
    """One schema, tried against every step in order."""

    def __init__(self, steps: List[Step], schema: Any, structured_kwargs: dict):
        self.steps = list(steps)
        self.schema = schema
        self.structured_kwargs = dict(structured_kwargs)

    def _runnable(self, step: Step) -> Runnable:
        return step.build().with_structured_output(self.schema, **self.structured_kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        def attempt(step: Step) -> Any:
            return self._runnable(step).invoke(input, config, **kwargs)

        return first_success(self.steps, attempt)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        async def attempt(step: Step) -> Any:
            return await self._runnable(step).ainvoke(input, config, **kwargs)

        return await first_success_async(self.steps, attempt)
