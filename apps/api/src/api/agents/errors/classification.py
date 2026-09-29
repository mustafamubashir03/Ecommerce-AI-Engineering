"""Reading a provider failure: what it was, and what it means.

Pure functions about an exception. Nothing here remembers anything, and nothing
here invents a message: the caller always sees the provider's real status and its
real body.
"""

import json
import logging
import re
import time

logger = logging.getLogger(__name__)

# Statuses worth one more try: the provider or the network misbehaved.
RETRYABLE_STATUS = frozenset({408, 409, 425, 500, 502, 503, 504})
# Statuses that will answer the same way on every model: do not rotate for them.
NEVER_RETRY_STATUS = frozenset({400, 401, 402, 403, 404, 413, 422})

# Markers OpenRouter itself sends when the limit belongs to the account rather
# than to a model or one of its providers. Taken from the real error bodies:
#   "limit_source": "openrouter_free_tier_daily"
#   "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock ..."
ACCOUNT_QUOTA_MARKERS = (
    "openrouter_free_tier_daily",
    "free-models-per-day",
    "free_models_per_day",
    "insufficient_credits",
    "credits to unlock",
    "negative credit balance",
)


class ProviderError(RuntimeError):
    """A provider call that failed, carrying its real status and body."""

    def __init__(self, message: str, status: int | None = None, body: str = "", wait_seconds: float = 0.0):
        super().__init__(message)
        self.status = status
        self.body = body
        self.wait_seconds = wait_seconds


def _headers(error: BaseException) -> dict:
    """Response headers, whether the SDK hangs them off the error or a response."""
    for source in (error, getattr(error, "response", None)):
        headers = getattr(source, "headers", None)
        if headers:
            return {str(key).lower(): value for key, value in dict(headers).items()}
    return {}


_STATUS_IN_TEXT = re.compile(
    r"(?:code|status)\s*[:=]\s*(?P<status>[1-5]\d{2})\b|\bHTTP\s+(?P<http_status>[1-5]\d{2})\b",
    re.IGNORECASE,
)


def status_from_text(text: str) -> int | None:
    """The status a provider named in its own message, if it named one."""
    match = _STATUS_IN_TEXT.search(text or "")
    if not match:
        return None
    return int(match.group("status") or match.group("http_status"))


def _status_code(value) -> int | None:
    """An int status, from an int or a numeric string. Nothing else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def status_of(error: BaseException) -> int | None:
    """The status of a failure: what the provider meant, not what the wire said.

    Precedence matters, because a streamed answer opens its HTTP connection
    successfully and then reports the failure inside the stream. The transport
    status is then a plain 200 while the real status sits in the body, so:

    1. a non-2xx error code the provider put in the body wins;
    2. otherwise the transport status the exception carries;
    3. otherwise a number the provider named in its own prose, for failures
       raised mid-stream that carry no status attribute at all.

    The body is read as JSON, never scanned for loose numbers, so a product id
    or a price in a message cannot be mistaken for a status.
    """
    payload = _status_code(_error_object(error).get("code"))
    if payload is not None and not 200 <= payload < 300:
        return payload

    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for candidate in candidates:
        found = _status_code(candidate)
        if found is not None:
            return found
    return status_from_text(str(error))


def is_routing_restriction(error: BaseException) -> bool:
    """True when a 403 is OpenRouter refusing the route, not the credentials.

    OpenRouter gates some free models behind an agentic harness and says so
    with a routing field rather than an authentication message:

        "metadata": {"failed_routing_step": "Gate Free Endpoints by Agentic
        Harness", "routing_funnel": [...]}

    That rejection is about this one model, so the next one is worth trying. A
    403 without that field is treated as a real authorisation failure, so this
    only ever narrows the rule in one direction and never widens it.
    """
    if status_of(error) != 403:
        return False
    metadata = _error_object(error).get("metadata")
    if not isinstance(metadata, dict):
        return False
    return bool(str(metadata.get("failed_routing_step") or "").strip())


def body_of(error: BaseException) -> str:
    """The provider's own words, never a summary written by this app."""
    body = getattr(error, "body", None)
    if isinstance(body, (dict, list)):
        return json.dumps(body, separators=(",", ":"))
    if isinstance(body, str) and body.strip():
        return body.strip()
    text = getattr(getattr(error, "response", None), "text", None)
    return text.strip() if isinstance(text, str) and text.strip() else ""


def wait_for(error: BaseException, fallback: float) -> float:
    """Seconds the provider asked us to wait, from Retry-After or X-RateLimit-Reset."""
    if isinstance(error, ProviderError) and error.wait_seconds:
        return error.wait_seconds
    headers = _headers(error)
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    reset = headers.get("x-ratelimit-reset")
    if reset:
        try:
            # OpenRouter sends an absolute unix timestamp in milliseconds.
            remaining = float(reset) / 1000.0 - time.time()
            if remaining > 0:
                return min(remaining, 24 * 3600)
        except ValueError:
            pass
    return fallback


def _body_json(error: BaseException) -> dict:
    raw = body_of(error)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _error_object(error: BaseException) -> dict:
    """The `error` object OpenRouter returns, or an empty dict."""
    parsed = _body_json(error)
    inner = parsed.get("error")
    if isinstance(inner, dict):
        return inner
    return parsed


def is_account_quota(error: BaseException) -> bool:
    """True when the limit is the account's, so no other model can help.

    Distinguishes the two things a 429 can mean. A per-account daily cap or a
    credit problem applies to every model on the key, so trying the rest of the
    pool only wastes requests and time. A 429 that names a provider, or that
    carries no account marker, is model specific and worth rotating.
    """
    status = status_of(error)
    if status == 402:
        return True
    if status != 429:
        return False

    inner = _error_object(error)
    metadata = inner.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("provider_name"):
        # A named provider means the limit is on that provider, not the account.
        return False

    haystack = " ".join(
        str(value)
        for value in (
            inner.get("message"),
            metadata.get("limit_source"),
            inner.get("code"),
            body_of(error),
        )
        if value
    ).lower()
    return any(marker in haystack for marker in ACCOUNT_QUOTA_MARKERS)


# A provider whose token window is full, or whose window the request outgrows,
# answers with HTTP 413 rather than 429 and names both figures in the body:
#
#   "on tokens per minute (TPM): Limit 8000, Requested 8154, ..."
#
# Both numbers are read from the body rather than configured: the window belongs
# to the organisation, and the account's own value is the authoritative one.
_TOKEN_WINDOW_SIGNAL = re.compile(r"tokens?\s+per\s+minute|\bTPM\b", re.IGNORECASE)
_TOKEN_WINDOW_FIGURES = re.compile(r"Limit\s*([\d,]+).*?Requested\s*([\d,]+)", re.IGNORECASE | re.DOTALL)

# What a proven token ceiling means for the provider that refused.
CEILING_TRANSIENT = "transient"  # the window may merely be full right now
CEILING_PERMANENT = "permanent"  # the request outgrows the whole window


def token_ceiling(error: BaseException) -> str | None:
    """How a 413 is scoped, or None when the body does not prove it is a window.

    A 413 on its own says nothing useful: OpenRouter answers a payload that is
    too large for its proxy with 413, and that request would be refused by every
    model it could be sent to. So this only reports a ceiling when the body
    carries two independent pieces of evidence, a token-window signal and a
    parseable Limit/Requested pair. Anything else returns None, and the caller
    keeps treating the 413 as unrecoverable.

    PERMANENT means the request alone is larger than the whole window, so no
    amount of waiting can satisfy it. TRANSIENT means the request would fit in
    the window but the window is currently full.
    """
    if status_of(error) != 413:
        return None

    inner = _error_object(error)
    body = body_of(error)
    message = " ".join(
        str(value) for value in (inner.get("type"), inner.get("message"), body) if value
    )
    if not _TOKEN_WINDOW_SIGNAL.search(message):
        return None

    figures = _TOKEN_WINDOW_FIGURES.search(body)
    if not figures:
        return None

    limit, requested = (int(value.replace(",", "")) for value in figures.groups())
    return CEILING_PERMANENT if requested > limit else CEILING_TRANSIENT


def is_fallback_worthy(error: BaseException) -> bool:
    """True when trying the next model in the pool could plausibly succeed."""
    if is_account_quota(error):
        return False
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status = status_of(error)
    if status is None:
        return False
    if status == 403 and is_routing_restriction(error):
        # The provider refused this one route, not the key, so the next model
        # is worth trying. A 403 without that evidence still stops the walk.
        return True
    if token_ceiling(error) is not None:
        # A 413 the body proves is a token window, rather than a payload or a
        # context limit. This is read here as well as in the policy so a ceiling
        # held by the cooldown is walked away from instead of being re-raised.
        return True
    if status in NEVER_RETRY_STATUS:
        # A bad request, a bad key or a missing model fails the same way on
        # every model, so rotating hides the real cause instead of fixing it.
        return False
    return status == 429 or status in RETRYABLE_STATUS


def describe(error: BaseException) -> ProviderError:
    """Wrap the raw failure, quoting the real status and body."""
    if isinstance(error, ProviderError):
        return error
    status = status_of(error)
    body = body_of(error)
    detail = body or str(error) or error.__class__.__name__
    message = f"HTTP {status}: {detail}" if status else (str(error) or detail)
    return ProviderError(message, status, body, wait_for(error, 0.0))
