"""Prompt text and structured response models for the shopping assistant.

The prompt wording itself lives in `config.yaml` under `agent.prompts` so it can
be tuned without touching code.
"""

from pydantic import BaseModel, Field

from api.core.settings import get_settings


class IntentRouterResponse(BaseModel):
    """Decides whether a question is about the product catalogue."""

    question_relevancy: bool = Field(
        description="True when the question can be answered with products in stock."
    )
    answer: str = Field(
        description="Empty when relevant, otherwise why the question is out of scope."
    )


def prompt(name: str) -> str:
    return get_settings().agent.prompts[name].strip()
