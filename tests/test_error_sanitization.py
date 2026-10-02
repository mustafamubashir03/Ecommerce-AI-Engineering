import json
import logging

import httpx
import pytest

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

LEAKS = (ORG_ID, BILLING_URL, SERVICE_TIER, "on tokens per minute", "gpt-oss-120b", "request_too_large")


def provider_error(status: int = 413, body: dict | None = None) -> Exception:
    """The real SDK error, carrying the real body the way the SDK carries it."""
    from openai import APIStatusError

    payload = GROQ_413_BODY if body is None else body
    response = httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return APIStatusError(f"Error code: {status}", response=response, body=payload)


def assert_sanitised(payload: str) -> None:
    lowered = payload.lower()
    for leak in LEAKS:
        assert leak.lower() not in lowered, f"{leak!r} reached the client: {payload}"
    assert "http 413:" not in lowered, f"the raw describe() string reached the client: {payload}"


def test_a_413_is_classified_413():
    from api.agents.errors import status_of

    assert status_of(provider_error()) == 413


def test_a_status_reported_inside_a_200_body_wins_over_the_transport():
    """A streamed failure arrives on a 200, so the body is the only place it can be."""
    from api.agents.errors import status_of

    body = json.dumps({"error": {"message": "upstream overloaded", "code": 503}})
    error = provider_error(200, {"error": {"message": "upstream overloaded", "code": 503}})
    error.body = body
    assert status_of(error) == 503


@pytest.fixture
def client(monkeypatch):
    """The real app, with only the agent turn replaced by a failing one."""
    from fastapi.testclient import TestClient

    import api.api.endpoints as endpoints_module
    from api.app import app

    def failing_run_agent(query, thread_id=None):
        raise provider_error()

    def failing_stream_agent(query, thread_id=None):
        raise provider_error()
        yield  # pragma: no cover - makes this a generator, like the real one

    monkeypatch.setattr(endpoints_module, "run_agent", failing_run_agent)
    monkeypatch.setattr(endpoints_module, "rag_agent_stream_wrapper", failing_stream_agent)
    return TestClient(app, raise_server_exceptions=False)


def test_the_blocking_response_carries_no_provider_internals(client):
    response = client.post("/agent/", json={"query": "laptops with 16gb ram"})

    assert_sanitised(response.text)
    assert response.status_code == 500
    assert set(json.loads(response.text)["detail"]) == {"error"}, "no extra provider fields added"


def test_the_blocking_message_explains_the_token_limit(client):
    detail = client.post("/agent/", json={"query": "laptops with 16gb ram"}).json()["detail"]
    message = detail["error"].lower()

    assert "token" in message, f"the user is not told what went wrong: {detail['error']}"
    assert "shorter" in message or "specific" in message, detail["error"]


def test_the_stream_error_event_carries_no_provider_internals(client):
    with client.stream("POST", "/agent/stream", json={"query": "laptops with 16gb ram"}) as response:
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert frames, "the stream must still report the failure"
    assert_sanitised("\n".join(frames))

    events = [json.loads(frame[6:]) for frame in frames]
    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["status"] == 413, "the real status is still reported"
    assert "token" in events[0]["message"].lower()


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422, 429, 500, 503])
def test_no_status_can_leak_the_body(status):
    from api.api.endpoints import _as_http_error

    http_error = _as_http_error(provider_error(status, {"message": f"secret detail for {status}"}))

    assert_sanitised(json.dumps(http_error.detail))
    assert "secret detail" not in json.dumps(http_error.detail)


def test_an_unreachable_provider_gets_a_message():
    from api.api.endpoints import _as_http_error

    http_error = _as_http_error(ConnectionError("connection refused to 10.0.0.7:6333"))
    assert http_error.status_code == 500
    assert "reached" in http_error.detail["error"].lower()
    assert "10.0.0.7" not in json.dumps(http_error.detail)


def test_the_server_side_log_keeps_the_full_provider_body(caplog):
    from api.agents.errors import describe
    from api.api.endpoints import _as_http_error

    with caplog.at_level(logging.ERROR, logger="api.api.endpoints"):
        _as_http_error(provider_error())

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert ORG_ID in logged, "the operator still needs the account id to debug this"
    assert BILLING_URL in logged
    assert describe(provider_error()).body, "the raw body is still carried on the failure"
