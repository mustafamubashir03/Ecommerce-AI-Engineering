import asyncio
import json
import logging
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.agents.graph import run_agent, stream_agent
from api.agents.provider_errors import ProviderError, describe
from api.agents.text import optional
from api.api.models import RAGUsedContext
from api.api.models import AgentRequest, RagRequest, RagResponse
from api.agents.retrieval_generation import rag_pipeline_wrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
agent_router = APIRouter()

# The provider owns the meaning of its own status codes, so they are passed
# through untouched and only the ones that mean "the caller sent something we
# cannot serve" are translated.
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
    """Server sent events: token chunks as the agent writes, then the result.

    The result event carries exactly the same body as POST /agent/, so a
    streaming client sees one response contract rather than two.
    """
    return StreamingResponse(
        _events(request, payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _as_http_error(error: Exception) -> HTTPException:
    """Surface the provider's real status and body, never a summary of our own."""
    failure = describe(error)
    logger.error("request %s failed: %s", getattr(error, "request_id", "-"), failure)
    detail = {"error": str(failure)}
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
    )


async def _events(request: Request, payload: AgentRequest):
    """Run the blocking graph on a worker thread and forward its events."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    finished = object()
    request_id = request.state.request_id

    def produce() -> None:
        delivered = False
        try:
            for event in stream_agent(payload.query, payload.thread_id):
                if event["type"] == "result":
                    delivered = True
                    # Same shape as the blocking endpoint, request id included.
                    event = {"type": "result", "payload": _response(request, event["payload"]).model_dump()}
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as error:
            failure = describe(error)
            if delivered:
                # The answer and its products have already been sent, so the
                # turn is complete as far as the client is concerned. Reporting
                # a failure now would show a finished answer next to an error
                # banner, so the stream is closed cleanly instead.
                logger.warning("stream failed after the result was sent: %s", failure)
            else:
                logger.error("stream failed: %s", failure)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(failure), "status": failure.status},
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
