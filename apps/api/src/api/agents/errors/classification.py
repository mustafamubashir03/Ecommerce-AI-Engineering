import json
import re

_STATUS_IN_TEXT = re.compile(
    r"(?:code|status)\s*[:=]\s*(?P<status>[1-5]\d{2})\b|\bHTTP\s+(?P<http_status>[1-5]\d{2})\b",
    re.IGNORECASE,
)


class ProviderError(RuntimeError):
    """A provider call that failed, carrying its real status and body."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def body_of(error: BaseException) -> str:
    """The provider's own words, never a summary written by this app."""
    body = getattr(error, "body", None)
    if isinstance(body, (dict, list)):
        return json.dumps(body, separators=(",", ":"))
    if isinstance(body, str) and body.strip():
        return body.strip()
    text = getattr(getattr(error, "response", None), "text", None)
    return text.strip() if isinstance(text, str) and text.strip() else ""


def _error_object(error: BaseException) -> dict:
    """The `error` object a provider returns, or an empty dict."""
    raw = body_of(error)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    inner = parsed.get("error")
    return inner if isinstance(inner, dict) else parsed


def _as_status(value) -> int | None:
    """An int status, from an int or a numeric string. Nothing else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def status_of(error: BaseException) -> int | None:
    from_body = _as_status(_error_object(error).get("code"))
    if from_body is not None and not 200 <= from_body < 300:
        return from_body

    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        found = _as_status(candidate)
        if found is not None:
            return found

    match = _STATUS_IN_TEXT.search(str(error) or "")
    return int(match.group("status") or match.group("http_status")) if match else None


def describe(error: BaseException) -> ProviderError:
    """Wrap the raw failure, quoting the real status and body for the logs."""
    if isinstance(error, ProviderError):
        return error
    status = status_of(error)
    body = body_of(error)
    detail = body or str(error) or error.__class__.__name__
    message = f"HTTP {status}: {detail}" if status else (str(error) or detail)
    return ProviderError(message, status, body)
