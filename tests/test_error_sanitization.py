"""What the client is told when a provider fails.

The live Groq 413 looked like this once it had been through the API:

    HTTP 413: {"message":"Request too large for model `openai/gpt-oss-120b` in
    organization `org_01khkq8n11eq8r2hcq0jxfabyd` service tier `on_demand` on
    tokens per minute (TPM): Limit 8000, Requested 8154, ... Upgrade to Dev Tier
    today at https://console.groq.com/settings/billing","type":"tokens",...}

An account id, a service tier, a billing link and a raw body do not belong in a
response sent to a browser. These tests pin the sanitized message in place of it,
and pin the routing decision as it was, because the two are separate concerns:
413 still raises rather than rotates.

Groq is reached through the OpenAI client pointed at Groq's base url, so the real
error class here is `openai.APIStatusError`, which is what actually raised.
"""

import json
import logging

import httpx
import pytest

import fake_openrouter as fake
from api.agents.errors import is_fallback_worthy, status_of
from api.agents.model_router import RoutedChatModel
from api.agents.routing.policy import RAISE, action

# The internal details from the live body. None of them may reach the client.
ORG_ID = "org_01khkq8n11eq8r2hcq0jxfabyd"
BILLING_URL = "https://console.groq.com/settings/billing"
SERVICE_TIER = "on_demand"

GROQ_413_BODY = {
    "message": (
        f"Request too large for model `openai/gpt-oss-120b` in organization "
        f"`{ORG_ID}` service tier `{SERVICE_TIER}` on tokens per minute (TPM): "
        f"Limit 8000, Requested 8154, please reduce your message size and try "
        f"again. Need more tokens? Upgrade to Dev Tier today at {BILLING_URL}"
    ),
    "type": "tokens",
    "code": "request_too_large",
}

# Everything a provider body is allowed to contain and still leak through.
LEAKS = (ORG_ID, BILLING_URL, SERVICE_TIER, "on tokens per minute", "gpt-oss-120b", "request_too_large")


def groq_error(status: int = 413, body: dict | None = None) -> Exception:
    """The real SDK error, carrying the real body the way the SDK carries it."""
    from openai import APIStatusError

    payload = GROQ_413_BODY if body is None else body
    response = httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return APIStatusError(f"Error code: {status}", response=response, body=payload)


# --- the routing decision is unchanged --------------------------------------


def test_a_groq_413_is_still_classified_413_and_raises():
    error = groq_error()
    assert status_of(error) == 413
    assert is_fallback_worthy(error) is False
    assert action(error, openrouter=False, last_of_provider=False, last_step=False) == RAISE


def test_a_groq_413_does_not_ask_another_model(monkeypatch):
    """The pool stops at the 413 rather than walking on, as it did before."""
    calls = fake.install(monkeypatch, {"model/a": [groq_error()], "model/b": []})
    routed = RoutedChatModel(models=["model/a", "model/b"], options={"api_key": "test"})

    with pytest.raises(Exception) as raised:
        routed.invoke("which washing machines do you have?")

    assert calls == ["model/a"], "a 413 must not rotate to the next model"
    assert status_of(raised.value) == 413


# --- the client never sees the provider's words -----------------------------


@pytest.fixture
def client(monkeypatch):
    """The real app, with only the agent turn replaced by a failing one."""
    from fastapi.testclient import TestClient

    import api.api.endpoints as endpoints_module
    from api.app import app

    def failing_run_agent(query, thread_id=None):
        raise groq_error()

    def failing_stream_agent(query, thread_id=None):
        raise groq_error()
        yield  # pragma: no cover - makes this a generator, like the real one

    monkeypatch.setattr(endpoints_module, "run_agent", failing_run_agent)
    monkeypatch.setattr(endpoints_module, "stream_agent", failing_stream_agent)
    return TestClient(app, raise_server_exceptions=False)


def _assert_sanitised(payload: str) -> None:
    lowered = payload.lower()
    for leak in LEAKS:
        assert leak.lower() not in lowered, f"{leak!r} reached the client: {payload}"
    assert "http 413:" not in lowered, f"the raw describe() string reached the client: {payload}"


def test_the_blocking_response_carries_no_provider_internals(client):
    response = client.post("/agent/", json={"query": "laptops with 16gb ram"})

    body = response.text
    _assert_sanitised(body)
    # 413 is not a provider status we pass through, and an SDK error is not our
    # own ProviderError, so this was a 500 with a raw body before the fix and is
    # a 500 with a written message now. Only the message changed.
    assert response.status_code == 500, "the existing status mapping is unchanged"
    assert set(json.loads(body)["detail"]) == {"error"}, "no extra provider fields added"


def test_the_blocking_message_explains_the_token_limit(client):
    detail = client.post("/agent/", json={"query": "laptops with 16gb ram"}).json()["detail"]
    message = detail["error"].lower()

    assert "token" in message, f"the user is not told what went wrong: {detail['error']}"
    assert "shorter" in message or "specific" in message, detail["error"]


def test_the_stream_error_event_carries_no_provider_internals(client):
    with client.stream("POST", "/agent/stream", json={"query": "laptops with 16gb ram"}) as response:
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert frames, "the stream must still report the failure"
    _assert_sanitised("\n".join(frames))

    events = [json.loads(frame[6:]) for frame in frames]
    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["status"] == 413, "the real status is still reported"
    assert "token" in events[0]["message"].lower()


# --- every status is covered, not just 413 ----------------------------------


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422, 429, 500, 503])
def test_no_status_can_leak_the_body(monkeypatch, status):
    from api.api.endpoints import _as_http_error

    http_error = _as_http_error(groq_error(status, body={"message": f"secret detail for {status}"}))

    _assert_sanitised(json.dumps(http_error.detail))
    assert "secret detail" not in json.dumps(http_error.detail)


def test_an_unreachable_provider_gets_a_message(status=None):
    from api.agents.errors import describe
    from api.api.endpoints import _as_http_error

    http_error = _as_http_error(ConnectionError("connection refused to 10.0.0.7:6333"))
    assert http_error.status_code == 500
    assert "reached" in http_error.detail["error"].lower()
    assert "10.0.0.7" not in json.dumps(http_error.detail)
    assert describe(ConnectionError("x")).status is None


# --- the detail is not lost, it is logged -----------------------------------


def test_the_server_side_log_keeps_the_full_provider_body(caplog):
    from api.agents.errors import describe
    from api.api.endpoints import _as_http_error

    with caplog.at_level(logging.ERROR, logger="api.api.endpoints"):
        _as_http_error(groq_error())

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert ORG_ID in logged, "the operator still needs the account id to debug this"
    assert BILLING_URL in logged
    assert describe(groq_error()).body, "the raw body is still carried on the failure"
