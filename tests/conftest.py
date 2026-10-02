"""Test configuration: the app's own src on the path, and no network needed."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture(autouse=True)
def _no_conversation_state(monkeypatch):
    from api.agents import checkpointer

    monkeypatch.setattr(checkpointer, "get_checkpointer", lambda: checkpointer.InMemorySaver())
