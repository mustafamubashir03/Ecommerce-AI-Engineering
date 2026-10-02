import logging
from typing import Iterator

logger = logging.getLogger(__name__)

STREAM_MODES = ("values", "messages", "debug", "custom")

NODE_LABELS = {
    "intent_router_node": "Checking what you are asking about",
    "agent_node": "Looking through the products in stock",
}

_FALLBACK_STATUS = "Working on it"


def _status_for(chunk: dict) -> str | None:
    if not isinstance(chunk, dict) or chunk.get("type") != "task":
        return None
    payload = chunk.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not name:
        return None
    label = NODE_LABELS.get(name)
    return label if label else _FALLBACK_STATUS


def _as_mode_chunk(item) -> tuple[str, object]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0], item[1]
    return "", item


def stream_graph_events(
    graph,
    initial_state: dict,
    config: dict,
    to_result,
    *,
    modes: tuple[str, ...] = STREAM_MODES,
) -> Iterator[dict]:
    result_sent = False
    announced: set[str] = set()

    for item in graph.stream(initial_state, config=config, stream_mode=list(modes)):
        mode, chunk = _as_mode_chunk(item)

        if mode == "messages":
            continue

        elif mode == "debug":
            status = _status_for(chunk)
            if status and status not in announced:
                announced.add(status)
                yield {"type": "status", "text": status}

        elif mode == "custom":
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("token")
            if text:
                yield {"type": "token", "text": text}

        elif mode == "values":
            if result_sent or not isinstance(chunk, dict):
                continue
            if chunk.get("answer"):
                result_sent = True
                yield {"type": "result", "payload": to_result(chunk)}
