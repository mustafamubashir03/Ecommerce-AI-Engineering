from api.api.models import RAGUsedContext
from fastapi import APIRouter, Request
import logging
from api.api.models import RagRequest, RagResponse
from api.agents.retrieval_generation import rag_pipeline_wrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request:Request, payload:RagRequest)->RagResponse:
    answer = rag_pipeline_wrapper(payload.query)
    return RagResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[RAGUsedContext(**item) for item in answer["used_context"]]
    )


api_router = APIRouter()
api_router.include_router(rag_router,prefix="/rag",tags=["rag"])


