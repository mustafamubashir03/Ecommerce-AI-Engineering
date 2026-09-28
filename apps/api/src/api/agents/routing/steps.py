"""The ordered list of attempts, and how it is cut down per call.

A `Step` is one attempt: a provider, a model id, and how to build the client.
`ProviderPlan` is a provider that is not OpenRouter, with its own model list and
its own builder, so adding a provider never touches the rotation policy.
"""

from typing import Any, Callable, List, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field

from api.agents.provider_errors import hold_error, is_account_quota, is_fallback_worthy
from api.agents.routing.policy import PRIMARY


class ProviderPlan(BaseModel):
    """A provider that is not OpenRouter, with its own ordered model list.

    A plan, not a second router: it carries the models to try and the one
    function that turns a model id into a client, so the rotation policy in
    `policy.py` stays the only implementation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    models: List[str] = Field(default_factory=list)
    build: Callable[[str], Any]
    # When set, a strict JSON Schema request is only sent to these models.
    strict_structured_output: Optional[List[str]] = None

    def models_for(self, strict_only: bool) -> List[str]:
        if not strict_only or self.strict_structured_output is None:
            return self.models
        allowed = set(self.strict_structured_output)
        return [model for model in self.models if model in allowed]


class Step(NamedTuple):
    """One attempt: who answers, with what, and how to build it."""

    provider: str
    model_id: str
    build: Callable[[], Any]


def wants_strict_schema(structured_kwargs: dict) -> bool:
    """True when the caller asked for a strict JSON Schema response."""
    return structured_kwargs.get("method") == "json_schema" and bool(
        structured_kwargs.get("strict")
    )


def build_steps(
    models: List[str],
    options: dict,
    fallbacks: List[ProviderPlan],
    primary_builder: Callable[[str], Any],
    strict_only: bool = False,
) -> List[Step]:
    """Every attempt, in order: the OpenRouter pool, then each provider.

    Each step builds the plain model; the agent's tools are applied per call by
    the router, because `bind_tools` only takes effect through `invoke`.

    `primary_builder` is passed in rather than imported, so the caller owns the
    pool's client construction and can substitute it.
    """
    steps = [
        Step(PRIMARY, model_id, (lambda mid=model_id: primary_builder(mid, **options)))
        for model_id in models
    ]
    for plan in fallbacks:
        for model_id in plan.models_for(strict_only):
            steps.append(Step(plan.name, model_id, (lambda p=plan, mid=model_id: p.build(mid))))
    return steps


def steps_for_call(
    models: List[str],
    options: dict,
    fallbacks: List[ProviderPlan],
    primary_builder: Callable[[str], Any],
    strict_only: bool = False,
) -> List[Step]:
    """The steps to try, with a provider that is known to be unavailable cut.

    A refusal that is still current costs no request, so the OpenRouter steps
    are dropped rather than paid for again. If nothing else is left, the stored
    refusal is the answer.
    """
    steps = build_steps(models, options, fallbacks, primary_builder, strict_only)
    held = hold_error()
    if held is None:
        return steps
    if not (is_account_quota(held) or is_fallback_worthy(held)):
        raise held
    remaining = [step for step in steps if step.provider != PRIMARY]
    if not remaining:
        raise held
    return remaining
