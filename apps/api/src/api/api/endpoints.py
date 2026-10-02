import asyncio
import json
import logging
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.agents.graph import run_agent
from api.agents.errors import ProviderError, describe
from api.agents.rag import rag_agent_stream_wrapper, rag_pipeline_wrapper
from api.agents.text import optional
from api.core.tracing import get_trace_client
from api.models import (
    AgentRequest,
    FeedbackRequest,
    FeedbackResponse,
    RAGUsedContext,
    RagRequest,
    RagResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
agent_router = APIRouter()
feedback_router = APIRouter()

PROVIDER_STATUS = frozenset({401, 402, 403, 404, 422, 429})


@rag_router.post("/")
def rag(request: Request, payload: RagRequest) -> RagResponse:
    """Direct RAG: one hybrid retrieval pass, no agent loop."""
    answer = rag_pipeline_wrapper(payload.query)
    return RagResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[RAGUsedContext(**item) for item in answer["used_context"]],
    )


@agent_router.post("/")
def agent(request: Request, payload: AgentRequest) -> RagResponse:
    """Blocking agent turn, kept for clients that do not stream."""
    try:
        state = run_agent(payload.query, thread_id=payload.thread_id)
    except Exception as error:  # noqa: BLE001 - reported with the real status and body
        raise _as_http_error(error) from error
    logger.info("agent answered %s (relevant=%s)", request.state.request_id, state["question_relevancy"])
    return _response(request, state)


@agent_router.post("/stream")
async def agent_stream(request: Request, payload: AgentRequest) -> StreamingResponse:
    return StreamingResponse(
        _events(request, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _user_message(failure: ProviderError) -> str:
    if failure.status == 413:
        return (
            "That question needs more tokens than this model allows in one "
            "request. Please ask a shorter or more specific question."
        )
    if failure.status == 429:
        return "The model provider is rate limiting requests right now. Please try again shortly."
    if failure.status in (400, 422):
        return "The model provider rejected the request. Please rephrase the question."
    if failure.status in (401, 402, 403):
        return "The model provider refused our access to this model. Please try again later."
    if failure.status == 404:
        return "This model is not available at the provider right now."
    if failure.status is None:
        return "The model provider could not be reached. Please try again."
    if failure.status is not None and 500 <= failure.status < 600:
        return "The model provider is temporarily unavailable. Please try again."
    return "The assistant could not complete this request."


def _as_http_error(error: Exception) -> HTTPException:
    """Surface the provider's real status, with a message written for the caller."""
    failure = describe(error)
    logger.error("request failed, provider said: %s", failure)
    detail = {"error": _user_message(failure)}
    if failure.status in PROVIDER_STATUS:
        return HTTPException(status_code=failure.status, detail=detail)
    if isinstance(error, ProviderError):
        detail["status"] = failure.status
        return HTTPException(status_code=502, detail=detail)
    return HTTPException(status_code=500, detail=detail)


def _response(request: Request, state: dict) -> RagResponse:
    return RagResponse(
        request_id=request.state.request_id,
        answer=optional(state.get("answer")),
        question_relevancy=bool(state.get("question_relevancy", False)),
        used_context=[RAGUsedContext(**item) for item in state.get("used_context", [])],
        thread_id=state.get("thread_id"),
        trace_id=optional(state.get("trace_id")),
    )


@feedback_router.post("/")
def feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    trace_id = (payload.trace_id or "").strip()
    comment = (payload.feedback_text or "").strip()

    if not trace_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "No trace to attach this feedback to. Tracing may be switched off."},
        )
    if payload.feedback_score is None and not comment:
        raise HTTPException(
            status_code=400,
            detail={"error": "Nothing to record: send a score, a comment, or both."},
        )

    client = get_trace_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Feedback is unavailable because tracing is not configured."},
        )

    source_info = {"thread_id": payload.thread_id} if payload.thread_id else None

    def write(key: str, value) -> None:
        client.create_feedback(
            trace_id=trace_id,
            key=key,
            value=value,
            feedback_source_type=payload.feedback_source_type,
            source_info=source_info,
        )

    try:
        if payload.feedback_score is not None:
            write("thumbs", payload.feedback_score)
        if len(comment) > 0:
            write("comment", comment)
    except Exception as error:  # noqa: BLE001 - the caller must be told it was not saved
        logger.error("feedback was not recorded for trace %s: %s", trace_id, error)
        raise HTTPException(
            status_code=502,
            detail={"error": "The feedback could not be recorded. Please try again."},
        ) from error

    logger.info(
        "feedback recorded for trace %s from %s", trace_id, payload.feedback_source_type
    )
    return FeedbackResponse(request_id=request.state.request_id, status="recorded")


async def _events(request: Request, payload: AgentRequest):
    """Run the blocking graph on a worker thread and forward its events."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    finished = object()

    def produce() -> None:
        delivered = False
        try:
            for event in rag_agent_stream_wrapper(payload.query, payload.thread_id):
                if event["type"] == "result":
                    delivered = True
                    event = {"type": "result", "payload": _response(request, event["payload"]).model_dump()}
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as error:
            failure = describe(error)
            if delivered:
                logger.warning("stream failed after the result was sent: %s", failure)
            else:
                logger.error("stream failed, provider said: %s", failure)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "type": "error",
                        "message": _user_message(failure),
                        "status": failure.status,
                    },
                )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, finished)

    threading.Thread(target=produce, daemon=True).start()

    while True:
        event = await queue.get()
        if event is finished:
            break
        yield f"data: {json.dumps(event)}\n\n"

    yield 'data: {"type": "done"}\n\n'


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
