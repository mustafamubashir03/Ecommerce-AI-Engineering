from typing import List, Literal, Optional

from pydantic import BaseModel, Field



class RagRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline.")


class AgentRequest(BaseModel):
    query: str = Field(..., description="The query to be sent to the shopping assistant agent.")
    thread_id: Optional[str] = Field(
        default=None, description="Conversation id, keeps the history across turns."
    )


class RAGUsedContext(BaseModel):
    """One product card in the answer, as the UI renders it."""

    id: str = Field(description="The ID of the item used to answer the question.")
    image_url: str = Field(description="The image url of the item used to answer the question.")
    price: Optional[float] = Field(description="The price of the item used to answer the question.")
    description: str = Field(description="A brief description of why this item was useful.")
    rating: Optional[float] = Field(
        default=None, description="Average customer rating of the item, when stored."
    )


class RagResponse(BaseModel):
    request_id: str = Field(..., description="The request id.")
    answer: str = Field(..., description="Answer to the question")
    question_relevancy: bool = Field(
        default=False, description="Whether the question was judged to be about the catalogue."
    )
    used_context: List[RAGUsedContext] = Field(
        description="List of items used to answer the question"
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation id, echo back on the next turn to keep the history.",
    )
    trace_id: str = Field(
        default="",
        description=(
            "Id of the trace this answer was produced under, to quote back when "
            "submitting feedback. Empty when tracing is switched off."
        ),
    )


class FeedbackRequest(BaseModel):

    feedback_score: Optional[int] = Field(
        default=None,
        description="The vote: 1 for helpful, -1 for not, null when no vote was cast.",
    )
    feedback_text: str = Field(
        default="", description="Anything the user typed. Only recorded when it is not blank."
    )
    trace_id: str = Field(
        default="",
        description="The trace_id that came back with the answer being rated.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="The conversation this answer belongs to, recorded alongside the feedback.",
    )
    feedback_source_type: Literal["api", "model"] = Field(
        default="api",
        description=(
            "Who the feedback is from. 'api' is a person using the app, 'model' is a "
            "model assessing its own output. Only these two exist; anything else is "
            "rejected by the SDK before it is sent."
        ),
    )


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request id, to quote when reporting a problem.")
    status: str = Field(
        ..., description="'recorded' once the feedback has been written."
    )




class ReferenceItem(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question, as provided in the context.")
    description: str = Field(description="A brief description of why this item was useful.")


class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[ReferenceItem] = Field(description="List of items used to answer the question")
