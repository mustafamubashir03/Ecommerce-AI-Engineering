from typing import Optional
from pydantic import BaseModel, Field
from typing import List




class ReferenceItem(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question, as provided in the context.")
    description: str = Field(description="A brief description of why this item was useful.")

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question.")
    image_url:str = Field(description="The image url of the item used to answer the question.")
    price: Optional[float] = Field(description="The price of the item used to answer the question.")
    description: str = Field(description="A brief description of why this item was useful.")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[ReferenceItem] = Field(description="List of items used to answer the question")
class RagRequest(BaseModel):
    query:str = Field(...,description="The query to be used in the RAG pipeline.")

class RagResponse(BaseModel):
    request_id: str = Field(...,description="The request id.")
    answer:str = Field(...,description="Answer to the question")
    used_context: List[RAGUsedContext] = Field(description="List of items used to answer the question")