"""Remembering a refusal, so the next turn costs no request.

The point of this module is narrow: when a provider refuses a key rather than a
model, every later request would otherwise pay to discover that again. One
stored refusal, held for as long as the provider asked us to wait, means the
rotation moves straight to the next provider.
"""

import logging
import time

from api.agents.errors.classification import ProviderError, describe

logger = logging.getLogger(__name__)

# How long a pool-wide failure is remembered when the provider gave no wait hint.
STORM_GUARD_SECONDS = 5.0


class _Cooldown:
    """Remembers a refusal for as long as the provider asked us to wait.

    Process local by design: this is one API process with one key, and the
    alternative (Redis or a database row) would add shared state for a limit
    that OpenRouter already enforces server side. A multi-instance deployment
    would let each instance walk the pool once, which is why that limitation is
    recorded here rather than solved with infrastructure.
    """

    def __init__(self) -> None:
        self._until = 0.0
        self._error: ProviderError | None = None

    def trip(self, error: ProviderError, seconds: float) -> None:
        self._error = error
        self._until = time.monotonic() + max(seconds, 0.0)
        logger.warning(
            "provider refused the call (status %s). Not calling again for %.0fs: %s",
            error.status,
            seconds,
            error,
        )

    def hold(self) -> ProviderError | None:
        if self._error is not None and time.monotonic() < self._until:
            return self._error
        self._error = None
        self._until = 0.0
        return None

    def extend(self, error: ProviderError) -> None:
        """Record a failure discovered without a wait hint, briefly."""
        if self._error is not None and self._error.status == error.status:
            self._until = max(self._until, time.monotonic() + STORM_GUARD_SECONDS)
            return
        self.trip(error, STORM_GUARD_SECONDS)


_cooldown = _Cooldown()


def hold_error() -> ProviderError | None:
    """The stored refusal if the provider told us to wait, otherwise None."""
    return _cooldown.hold()


def trip(error: BaseException, seconds: float) -> ProviderError:
    """Record a refusal so the next turn costs no request."""
    wrapped = describe(error)
    wrapped.wait_seconds = max(wrapped.wait_seconds, seconds)
    _cooldown.trip(wrapped, wrapped.wait_seconds)
    return wrapped


def record_pool_failure(error: BaseException) -> ProviderError:
    """Remember a failure the whole pool agreed on, briefly.

    The provider gave no wait hint, but every model failing the same way means
    the next request should not pay for the same walk again.
    """
    _cooldown.extend(describe(error))
    return _cooldown.hold() or describe(error)
