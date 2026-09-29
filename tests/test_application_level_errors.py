"""Regression tests for application-level provider errors.

Every exception here is the real `openrouter.errors.OpenRouterError` family,
constructed the way the SDK constructs it, with bodies copied from real
responses. A plain `Exception` would not prove anything here, because the
transport status of a streamed failure is a perfectly ordinary 200.
"""

import json

import httpx
import pytest

from api.agents.errors import (
    is_account_quota,
    is_fallback_worthy,
    is_routing_restriction,
    status_of,
)
from api.agents.routing.policy import RAISE, ROTATE, action

# --- bodies captured from real OpenRouter responses ------------------------

# A streamed answer whose transport was 200 and whose payload carried the
# failure. This is the one that ended live requests.
SSE_503_BODY = json.dumps(
    {
        "id": "gen-1790676083-oCiehG03GvjANqOYGMqP",
        "error": {
            "message": "Upstream error from Nvidia: Service temporarily overloaded",
            "code": 503,
            "metadata": {"error_type": "provider_overloaded"},
        },
    }
)

# A 403 OpenRouter produced because the model is gated behind a harness, not
# because the credentials are wrong.
CAPABILITY_403_BODY = json.dumps(
    {
        "error": {
            "message": (
                "thinkingmachines/inkling:free is only available on agentic "
                "harnesses. Try plugging it into a coding agent or productivity "
                "app listed on https://openrouter.ai/apps"
            ),
            "code": 403,
            "metadata": {
                "routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 1}],
                "failed_routing_step": "Gate Free Endpoints by Agentic Harness",
            },
        }
    }
)

# A 401 from a real invalid key: "User not found."
AUTH_401_BODY = json.dumps({"error": {"message": "User not found.", "code": 401, "metadata": None}})


def openrouter_error(body: str, transport_status: int):
    """Build the real SDK error, so `status_code` is the transport status."""
    from openrouter.errors import OpenRouterError

    response = httpx.Response(
        status_code=transport_status,
        content=body.encode("utf-8"),
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    return OpenRouterError("Provider returned error", response, body)


def verdict(error, last_of_provider: bool = False) -> str:
    """The routing decision, as the walker would take it.

    The default is the rotating case: the pool has more models after this one,
    so a failure moves to the next model of the same provider.
    """
    return action(error, openrouter=True, last_of_provider=last_of_provider, last_step=False)


# --- the live failure ------------------------------------------------------


def test_a_streamed_503_is_read_from_the_body_not_the_transport():
    """The transport succeeded, so its 200 must not mask the real failure."""
    error = openrouter_error(SSE_503_BODY, transport_status=200)
    assert status_of(error) == 503
    assert is_fallback_worthy(error)
    assert verdict(error) == ROTATE


def test_a_streamed_500_rotates():
    error = openrouter_error(
        json.dumps({"error": {"message": "upstream error", "code": 500}}), transport_status=200
    )
    assert status_of(error) == 500
    assert is_fallback_worthy(error)
    assert verdict(error) == ROTATE


def test_a_streamed_provider_specific_429_rotates():
    error = openrouter_error(
        json.dumps(
            {
                "error": {
                    "message": "Provider returned error",
                    "code": 429,
                    "metadata": {"provider_name": "Poolside", "limit_source": "upstream_provider_shared_pool"},
                }
            }
        ),
        transport_status=200,
    )
    assert status_of(error) == 429
    assert not is_account_quota(error), "a named provider is model specific"
    assert is_fallback_worthy(error)
    assert verdict(error) == ROTATE


def test_a_streamed_400_still_stops():
    error = openrouter_error(
        json.dumps({"error": {"message": "malformed request", "code": 400}}), transport_status=200
    )
    assert status_of(error) == 400
    assert not is_fallback_worthy(error)
    assert verdict(error) == RAISE


def test_a_successful_transport_with_no_error_is_not_a_failure():
    """A real 200 with no error payload must not become a status of its own."""
    error = openrouter_error(json.dumps({"id": "gen-1", "choices": []}), transport_status=200)
    assert status_of(error) == 200
    assert not is_fallback_worthy(error)


# --- 403: capability restriction versus authorisation ----------------------


def test_a_capability_403_rotates_because_the_route_was_refused():
    error = openrouter_error(CAPABILITY_403_BODY, transport_status=403)
    assert status_of(error) == 403
    assert is_routing_restriction(error)
    assert is_fallback_worthy(error), "another model is worth trying"
    assert verdict(error) == ROTATE


def test_a_plain_403_still_stops():
    """The global 403 policy is only narrowed, never widened."""
    error = openrouter_error(
        json.dumps({"error": {"message": "Forbidden", "code": 403}}), transport_status=403
    )
    assert status_of(error) == 403
    assert not is_routing_restriction(error)
    assert not is_fallback_worthy(error)
    assert verdict(error) == RAISE


def test_a_real_authentication_failure_is_untouched():
    error = openrouter_error(AUTH_401_BODY, transport_status=401)
    assert status_of(error) == 401
    assert not is_routing_restriction(error)
    assert not is_fallback_worthy(error)


# --- the narrow parsing rules must survive --------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "model B0TEST0001 answered with no problem",
        "the price is 19.99 for that washer",
        "200 OK from upstream",
        "no numbers at all",
    ],
)
def test_a_bare_number_in_prose_is_never_a_status(message):
    """Free text must not be mined for statuses; only the structured body is."""

    class Prose(Exception):
        pass

    assert status_of(Prose(message)) is None
    assert not is_fallback_worthy(Prose(message))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Upstream error from Nvidia: overloaded (code: 503)", 503),
        ("status=503", 503),
        ("HTTP 503: upstream unavailable", 503),
    ],
)
def test_a_status_named_in_prose_is_still_recognised(text, expected):
    class Prose(Exception):
        pass

    assert status_of(Prose(text)) == expected
