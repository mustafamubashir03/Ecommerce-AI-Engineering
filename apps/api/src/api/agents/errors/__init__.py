"""How the application reads a provider failure.

    `classification`  the real status, the real body, and both wrapped together

The endpoint logs that raw failure and sends the caller a message written for
them, so a provider body never reaches a browser.
"""

from api.agents.errors.classification import (
    ProviderError,
    body_of,
    describe,
    status_of,
)

__all__ = ["ProviderError", "body_of", "describe", "status_of"]
