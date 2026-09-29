"""Every structured model in the application, in one place.

Three groups, all here so there is one file to look at when a field is wrong:

    the request/response contract   what the HTTP API accepts and returns
    the RAG answer                  what the `/rag/` pipeline returns
    the intent router's answer      what the first graph node returns

Anything that belongs to exactly one node is defined next to that node instead.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# --- what the HTTP API accepts and returns ----------------------------------


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


# --- what the RAG pipeline returns ------------------------------------------


class ReferenceItem(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question, as provided in the context.")
    description: str = Field(description="A brief description of why this item was useful.")


class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[ReferenceItem] = Field(description="List of items used to answer the question")
