"""Reading a provider failure: what it was, and what it means.

Pure functions about an exception. Nothing here remembers anything, and nothing
here invents a message: the caller always sees the provider's real status and its
real body.
"""

import json
import logging
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


def status_of(error: BaseException) -> int | None:
    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
    return None


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


def is_fallback_worthy(error: BaseException) -> bool:
    """True when trying the next model in the pool could plausibly succeed."""
    if is_account_quota(error):
        return False
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status = status_of(error)
    if status is None:
        return False
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
