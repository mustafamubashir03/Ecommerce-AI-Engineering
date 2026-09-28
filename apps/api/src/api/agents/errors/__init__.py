"""How the application reacts to a provider failure.

Split by what the code does:

    `classification`  read the failure: its status, its body, whether rotating
                      could help, whether the limit is the account's
    `cooldown`        remember it, so the next turn costs no request

Nothing here invents an error message: the caller always sees the real HTTP
status and the real response body, and a request the provider already refused is
not sent again while the refusal is current.
"""

from api.agents.errors.classification import (
    ACCOUNT_QUOTA_MARKERS,
    NEVER_RETRY_STATUS,
    RETRYABLE_STATUS,
    ProviderError,
    body_of,
    describe,
    is_account_quota,
    is_fallback_worthy,
    status_of,
    wait_for,
)
from api.agents.errors.cooldown import (
    claim_failure_walk,
    hold_error,
    record_pool_failure,
    release_failure_walk,
    share_failure,
    trip,
)

__all__ = [
    "ACCOUNT_QUOTA_MARKERS",
    "NEVER_RETRY_STATUS",
    "RETRYABLE_STATUS",
    "ProviderError",
    "body_of",
    "claim_failure_walk",
    "describe",
    "hold_error",
    "is_account_quota",
    "is_fallback_worthy",
    "record_pool_failure",
    "release_failure_walk",
    "share_failure",
    "status_of",
    "trip",
    "wait_for",
]
