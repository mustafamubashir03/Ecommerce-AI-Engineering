"""429 classification robustness and the cooldown that stops repeat traffic.

The policy is unchanged. These tests exist so a change in OpenRouter's response
shape cannot quietly turn an account-wide quota into a walk of the whole pool.
"""


import pytest

import fake_openrouter as fake
from api.agents.model_router import RoutedChatModel
from api.agents.errors import (
    describe,
    hold_error,
    is_account_quota,
    is_fallback_worthy,
    trip,
)

POOL = [f"model/{name}" for name in "abcdefghijkl"]


def routed(pool=POOL) -> RoutedChatModel:
    return RoutedChatModel(models=list(pool), options={"api_key": "test"})


def error_with(body, status: int = 429) -> fake.FakeOpenRouterError:
    return fake.FakeOpenRouterError(status, body, {"retry-after": "1"})


# --- the markers that must keep working ------------------------------------


def test_real_daily_cap_body_is_account_level():
    assert is_account_quota(fake.daily_cap_error())
    assert not is_fallback_worthy(fake.daily_cap_error())


def test_limit_source_marker_alone_is_enough():
    """The structured marker decides even when the message is reworded."""
    error = error_with({"error": {"message": "Rate limit exceeded", "metadata": {"limit_source": "openrouter_free_tier_daily"}}})
    assert is_account_quota(error)


def test_credit_markers_are_account_level():
    for marker in ("insufficient_credits", "Add 10 credits to unlock"):
        assert is_account_quota(error_with({"error": {"message": marker}})), marker


def test_payment_required_is_account_level():
    assert is_account_quota(fake.payment_error())


def test_named_provider_stays_fallback_worthy():
    assert not is_account_quota(fake.provider_limit_error())
    assert is_fallback_worthy(fake.provider_limit_error())


# --- shapes we do not recognise --------------------------------------------


def test_unknown_429_body_is_treated_as_model_specific():
    """Unmarked 429 rotates, which is the policy the project chose."""
    error = error_with({"error": {"message": "Rate limit exceeded"}})
    assert not is_account_quota(error)
    assert is_fallback_worthy(error)


def test_malformed_json_body_is_treated_as_model_specific():
    """A body that is not JSON must not raise, and must not stop the walk."""
    error = fake.FakeOpenRouterError(429, "{not json at all", {})
    assert not is_account_quota(error)
    assert is_fallback_worthy(error)


def test_empty_body_is_treated_as_model_specific():
    error = fake.FakeOpenRouterError(429, "", {})
    assert not is_account_quota(error)
    assert is_fallback_worthy(error)


def test_missing_error_object_is_treated_as_model_specific():
    assert not is_account_quota(error_with({"unexpected": "shape"}))
    assert is_fallback_worthy(error_with({"unexpected": "shape"}))


def test_error_object_of_the_wrong_type_does_not_raise():
    for body in (
        {"error": "a string, not an object"},
        {"error": ["a", "list"]},
        {"error": None},
        [],
        "a bare string",
        None,
    ):
        assert not is_account_quota(error_with(body)), body


def test_unexpected_limit_source_is_treated_as_model_specific():
    error = error_with({"error": {"message": "slow down", "metadata": {"limit_source": "something_new_2027"}}})
    assert not is_account_quota(error)
    assert is_fallback_worthy(error)


def test_metadata_of_the_wrong_type_does_not_raise():
    for metadata in ("a string", ["a", "list"], 42):
        error = error_with({"error": {"message": "slow down", "metadata": metadata}})
        assert not is_account_quota(error)


def test_provider_name_null_still_allows_account_detection():
    """The real body carries provider_name: null alongside the daily marker."""
    error = error_with(
        {"error": {"message": "free-models-per-day", "metadata": {"provider_name": None}}}
    )
    assert is_account_quota(error)


def test_provider_name_present_overrides_a_vague_message():
    error = error_with(
        {"error": {"message": "rate limit reached, please add credits if needed", "metadata": {"provider_name": "Acme"}}}
    )
    assert not is_account_quota(error), "a named provider means the limit is not the account's"


def test_a_format_change_cannot_walk_the_pool_twice(monkeypatch):
    """A burst after a pool-wide failure must not repeat the walk.

    Even if a future OpenRouter change hid the account marker, the second
    request is answered from the recorded failure instead of paying again.
    """
    import api.agents.model_router as router_module

    counter = {"n": 0}

    def build(model_id, **options):
        counter["n"] += 1
        # An unrecognised shape: still a 429, but with no account marker.
        raise error_with({"error": {"message": "Rate limit exceeded"}})

    monkeypatch.setattr(router_module, "build_chat_model", build)

    with pytest.raises(Exception):
        routed().invoke("first")
    first_cost = counter["n"]
    assert first_cost == len(POOL), "the first request walks the pool once"

    with pytest.raises(Exception):
        routed().invoke("second")
    assert counter["n"] == first_cost, "the second request costs no provider request"


# --- cooldown, sequentially -------------------------------------------------


def test_cooldown_makes_following_requests_free(monkeypatch):
    counter = {"n": 0}

    def build(model_id, **options):
        counter["n"] += 1
        raise fake.daily_cap_error()

    import api.agents.model_router as router_module

    original = router_module.build_chat_model
    router_module.build_chat_model = build
    try:
        for attempt in (1, 2, 3):
            with pytest.raises(Exception):
                routed(["model/a", "model/b", "model/c"]).invoke(f"turn {attempt}")
            if attempt == 1:
                assert counter["n"] == 1, "the quota stops the walk at the first model"
    finally:
        router_module.build_chat_model = original

    assert counter["n"] == 1, f"three turns cost {counter['n']} provider requests, expected 1"
    assert hold_error() is not None, "the refusal is held"


def test_held_error_reports_the_provider_wording():
    trip(fake.daily_cap_error(), 60)
    held = hold_error()
    assert "free-models-per-day" in str(held)
    assert "quota exhausted" not in str(held).lower()


# --- cooldown, concurrently -------------------------------------------------
def test_cooldown_is_process_local_and_says_so():
    """No shared-state client is imported, and the limit is documented."""
    import ast
    import inspect

    from api.agents.errors import cooldown as module

    source = inspect.getsource(module)
    assert "Process local" in source or "process local" in source, "the limitation is documented"

    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for shared_client in ("redis", "psycopg", "boto3", "sqlalchemy", "httpx"):
        assert shared_client not in imported, f"{shared_client} would make the cooldown shared"
    assert imported <= {"logging", "threading", "time", "api"}, imported


def test_describe_never_invents_a_message():
    """Whatever the provider said is what the caller sees."""
    for error in (
        fake.daily_cap_error(),
        fake.provider_limit_error(),
        fake.server_error(500),
        error_with({"weird": True}),
    ):
        message = str(describe(error))
        assert message, "there is always something to show"
        assert "quota exhausted" not in message.lower()

def test_a_status_named_only_in_the_message_is_still_read():
    """A failure part way through a stream often has no status_code.

    The SDK reports it as prose, and treating that as "unknown" made every
    transient 503 end the turn instead of rotating, which is how one request in
    three was failing while the pool still had models left to try.
    """
    import api.agents.errors.classification as classification

    class StreamingError(Exception):
        pass

    transient = StreamingError(
        "OpenRouter API returned an error during streaming: Upstream error from "
        "Nvidia: Service temporarily overloaded (code: 503)"
    )
    assert classification.status_of(transient) == 503
    assert classification.is_fallback_worthy(transient), "a transient 503 must rotate"

    # A refusal the caller has to act on still stops the turn.
    auth = StreamingError("Error code: 401 - no credentials")
    assert classification.status_of(auth) == 401
    assert not classification.is_fallback_worthy(auth)

    # A product id must not be mistaken for a status.
    nothing = StreamingError("model B0TEST0001 answered with no problem")
    assert classification.status_of(nothing) is None
    assert not classification.is_fallback_worthy(nothing)
