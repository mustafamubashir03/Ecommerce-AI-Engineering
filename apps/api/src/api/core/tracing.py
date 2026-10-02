import functools
import logging
import re
from contextvars import ContextVar
from functools import lru_cache

from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, traceable
from langsmith.utils import LangSmithRetry

logger = logging.getLogger(__name__)

_SPAN_USAGE: ContextVar[dict | None] = ContextVar("span_usage", default=None)

_QUOTA_REFUSAL = re.compile(
    r"LangSmithRateLimitError"
    r"|rate limit exceeded"
    r"|tenant exceeded usage limits"
    r"|usage limit",
    re.IGNORECASE,
)


class _DemoteQuotaNoise(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _QUOTA_REFUSAL.search(message):
            record.levelno = logging.DEBUG
            record.levelname = logging.DEBUG
        return True


def _quiet_exhausted_quota() -> None:
    """Install the filter on langsmith's ingest logger, once."""
    sdk_logger = logging.getLogger("langsmith.client")
    if not any(isinstance(existing, _DemoteQuotaNoise) for existing in sdk_logger.filters):
        sdk_logger.addFilter(_DemoteQuotaNoise())


def _report_ingest_failure(error: BaseException) -> None:
    logger.debug("LangSmith trace upload skipped: %s: %s", type(error).__name__, error)


@lru_cache(maxsize=1)
def get_trace_client() -> Client | None:
    import os

    if os.environ.get("LANGSMITH_TRACING", "").strip().lower() not in ("true", "1", "yes"):
        return None
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        return None

    client = Client(
        retry_config=LangSmithRetry(status=0),
        tracing_error_callback=_report_ingest_failure,
    )

    _quiet_exhausted_quota()

    import langsmith.run_trees as run_trees

    run_trees._CLIENT = client

    return client


def traced(name: str, **kwargs):
    span = traceable(name=name, client=get_trace_client(), **kwargs)

    def decorator(func):
        traced_func = span(func)

        @functools.wraps(func)
        def wrapper(*args, **kw):
            recorded: dict = {}

            def attach_usage(run_tree) -> None:
                if recorded:
                    run_tree.extra.setdefault("metadata", {}).update(recorded)

            marker = _SPAN_USAGE.set(recorded)
            try:
                passed = dict(kw.pop("langsmith_extra", None) or {})
                caller_hook = passed.pop("on_end", None)
                passed["on_end"] = (
                    (lambda run_tree: (attach_usage(run_tree), caller_hook(run_tree)))
                    if callable(caller_hook)
                    else attach_usage
                )
                return traced_func(*args, langsmith_extra=passed, **kw)
            finally:
                _SPAN_USAGE.reset(marker)

        return wrapper

    return decorator


def trace_usage(**counts) -> None:
    holder = _SPAN_USAGE.get()
    if holder is not None:
        holder.update({key: value for key, value in counts.items() if value is not None})


def current_trace_id() -> str:
    run_tree = get_current_run_tree()
    while run_tree is not None and getattr(run_tree, "parent_run_tree", None) is not None:
        run_tree = run_tree.parent_run_tree
    identifier = getattr(run_tree, "id", None) if run_tree is not None else None
    return str(identifier) if identifier else ""
