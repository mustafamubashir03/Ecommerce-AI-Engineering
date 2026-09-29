"""Test configuration: the app's own src on the path, and no network needed."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

# Set before the app loads .env, which does not override existing values. Keeps
# the test output free of tracing traffic and of any outbound tracing request.
os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture(autouse=True)
def _no_cooldown(monkeypatch):
    """Every test starts with no stored refusal and no walk in progress, so
    tests do not leak into each other through the process wide state.

    The state itself lives in `errors.cooldown`, which is where it has to be
    reset: patching a name on the re-exporting module would leave the real one
    untouched.
    """
    from api.agents.errors import cooldown

    monkeypatch.setattr(cooldown._cooldown, "_error", None)
    monkeypatch.setattr(cooldown._cooldown, "_until", 0.0)
    yield
