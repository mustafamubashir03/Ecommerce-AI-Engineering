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
def _no_conversation_state(monkeypatch):
    """Every test starts with an empty conversation history.

    The graph keeps its checkpoint in module state, so without this a thread id
    used by one test would be visible to the next.
    """
    from api.agents import checkpointer

    monkeypatch.setattr(checkpointer, "get_checkpointer", lambda: checkpointer.InMemorySaver())
