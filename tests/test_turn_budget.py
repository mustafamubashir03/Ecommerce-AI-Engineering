from typing import List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from api.agents import agent as agent_module
from api.agents.agent import agent_middleware, get_agent

PRODUCT = {
    "id": "B0TEST0001",
    "image_url": "https://example.test/one.jpg",
    "price": 19.9,
    "description": "Portable washer.",
    "rating": 4.2,
}


def test_the_three_guards_are_installed():
    names = [getattr(m, "name", None) or type(m).__name__ for m in agent_middleware()]
    assert names == [
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ContextEditingMiddleware",
    ]


def test_the_model_call_is_not_replaced_by_a_middleware():
    import inspect

    from api.agents import agent as agent_module

    node_source = inspect.getsource(agent_module.agent_node)
    assert ".invoke(" not in node_source, (
        "invoke aggregates the stream, which is the thing being avoided"
    )
    assert "_stream_writer" in node_source, "chunks must be written out as they arrive"
    assert "write(" in node_source, "and written on every chunk, not collected first"

    turn_source = inspect.getsource(agent_module._stream_agent_turn)
    assert ".stream(" in turn_source, "the turn must be driven as a stream"
    assert "stream_mode=" in turn_source, "and must ask for the chunk mode"

    assert '"messages"' in turn_source
    assert '"values"' in turn_source


def test_a_node_call_without_a_graph_run_does_not_raise(monkeypatch):
    """The blocking endpoint calls this node outside a stream, which must work."""
    from api.agents import agent as agent_module

    monkeypatch.setattr(
        agent_module, "_stream_agent_turn", lambda history, question: iter([])
    )

    result = agent_module.agent_node({"initial_query": "washing machines", "messages": []})

    assert result["answer"] == ""
    assert result["trace_id"] == ""


def test_summarization_is_not_installed():
    names = [type(m).__name__ for m in agent_middleware()]
    assert "SummarizationMiddleware" not in names


def test_the_model_call_limit_bounds_a_turn():
    from langchain.agents.middleware import ModelCallLimitMiddleware

    limits = [m for m in agent_middleware() if isinstance(m, ModelCallLimitMiddleware)]
    assert limits, "a turn must have a ceiling on model calls"
    assert limits[0].thread_limit is None, "bounded per run, not per conversation"
    assert limits[0].run_limit is not None and limits[0].run_limit <= 8, (
        "six calls is generous for one question; more is tokens for nothing"
    )
    assert limits[0].exit_behavior == "end", (
        "hitting the limit should end the turn with the answer so far, not fail it"
    )


def test_the_tool_call_limit_bounds_searching():
    from langchain.agents.middleware import ToolCallLimitMiddleware

    limits = [m for m in agent_middleware() if isinstance(m, ToolCallLimitMiddleware)]
    assert limits, "a turn must have a ceiling on tool calls"
    assert limits[0].run_limit is not None and limits[0].run_limit <= 6


def test_tool_output_is_cleared_before_it_can_grow_without_limit():
    from langchain.agents.middleware import ContextEditingMiddleware

    editing = [m for m in agent_middleware() if isinstance(m, ContextEditingMiddleware)]
    assert editing, "tool output needs a ceiling of its own"

    edit = editing[0].edits[0]
    assert edit.trigger <= 6000, (
        f"trigger of {edit.trigger} is above what this provider will accept"
    )
    assert edit.keep >= 1, "at least the newest result has to survive, the agent is reading it"


def test_the_agent_is_built_with_them():
    """A middleware list that is built and then not passed is a silent no-op."""
    import inspect

    source = inspect.getsource(get_agent)
    assert "middleware=agent_middleware()" in source, "the guards never reached create_agent"


class RunawayModel(BaseChatModel):

    calls: int = 0
    reached: List[int] = []

    @property
    def _llm_type(self) -> str:
        return "runaway"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        type(self).reached.append(self.calls)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "retrieve_data",
                                "args": {"query": "again"},
                                "id": f"call_{self.calls}",
                            }
                        ],
                    )
                )
            ]
        )


@pytest.fixture
def runaway_agent(monkeypatch):
    """The real agent, driven by a model that never stops searching."""
    import api.agents.graph as graph_module
    from langgraph.checkpoint.memory import InMemorySaver

    RunawayModel.calls = 0
    RunawayModel.reached = []

    monkeypatch.setattr(agent_module, "get_chat_model", lambda *a, **k: RunawayModel())
    get_agent.cache_clear()
    monkeypatch.setattr(graph_module, "get_checkpointer", lambda: InMemorySaver())
    monkeypatch.setattr(
        graph_module,
        "intent_router_node",
        lambda state: {"question_relevancy": True, "answer": ""},
    )
    monkeypatch.setattr(graph_module, "graph", graph_module.build_graph())
    yield RunawayModel
    get_agent.cache_clear()


def test_a_model_that_never_stops_is_stopped(runaway_agent):
    from api.agents.graph import run_agent

    state = run_agent("washing machines", thread_id="runaway")

    assert runaway_agent.calls <= 8, (
        f"made {runaway_agent.calls} model calls; the ceiling should have ended the turn sooner"
    )
    assert state is not None


def test_the_ceiling_is_reached_and_the_turn_still_returns(runaway_agent):
    """Stopping early is only useful if the turn still produces something."""
    from api.agents.graph import run_agent

    state = run_agent("washing machines", thread_id="runaway-2")

    assert "answer" in state
    assert runaway_agent.reached, "the model was never called, so nothing was proven"
