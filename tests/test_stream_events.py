import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from api.agents.stream import STREAM_MODES, stream_graph_events

QUESTION = "which washing machines do you have?"
TOOL_OUTPUT = "[B0TEST0001] (rating: 4.2) a portable washer, £189.99 from the shop"
ANSWER = "Here are the machines we have in stock."


class FakeGraph:
    """A graph that replays a fixed list of `(mode, chunk)` pairs."""

    def __init__(self, events):
        self.events = events
        self.calls = []

    def stream(self, initial_state, config=None, stream_mode=None):
        self.calls.append({"state": initial_state, "config": config, "modes": stream_mode})
        yield from self.events


def ai_chunk(text, node="model"):
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": node}))


def ai_message(text, node="model"):
    return ("messages", (AIMessage(content=text), {"langgraph_node": node}))


def task(name):
    return ("debug", {"type": "task", "payload": {"name": name, "input": {"initial_query": QUESTION}}})


def checkpoint(next_nodes, values):
    return ("debug", {"type": "checkpoint", "payload": {"next": next_nodes, "values": values}})


def values(**state):
    return ("values", state)


def run(events):
    graph = FakeGraph(events)
    out = list(stream_graph_events(graph, {"initial_query": QUESTION}, {"configurable": {}}, lambda s: s))
    return out, graph


def custom_chunk(text):
    """A chunk as the agent node writes it, from inside the turn."""
    return ("custom", {"token": text})


def test_a_run_of_chunks_is_emitted_chunk_by_chunk():
    events, _ = run([custom_chunk("Here "), custom_chunk("are "), custom_chunk("the machines.")])

    tokens = [event for event in events if event["type"] == "token"]
    assert [t["text"] for t in tokens] == ["Here ", "are ", "the machines."]


def test_the_answer_is_not_delivered_twice():
    events, _ = run([
        custom_chunk("Here are "),
        custom_chunk("the machines."),
        ai_message(ANSWER),
    ])

    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == "Here are the machines."
    assert streamed.count(ANSWER.strip()) <= 1


def test_a_custom_chunk_with_nothing_to_say_is_ignored():
    events, _ = run([("custom", {}), ("custom", {"token": ""}), custom_chunk("words")])

    assert [e["text"] for e in events if e["type"] == "token"] == ["words"]


def test_empty_text_is_not_emitted_but_whitespace_is():
    events, _ = run([custom_chunk(""), custom_chunk("Here"), custom_chunk(" "), custom_chunk("are")])

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "" not in tokens
    assert "".join(tokens) == "Here are"


def test_the_question_is_never_streamed_as_a_token():
    events, _ = run([("messages", (HumanMessage(content=QUESTION), {"langgraph_node": "agent_node"}))])

    assert not [e for e in events if e["type"] == "token"], "the human message is not the answer"


def test_the_tool_output_is_never_streamed_as_a_token():
    events, _ = run([("messages", (ToolMessage(content=TOOL_OUTPUT, tool_call_id="1"), {"langgraph_node": "agent_node"}))])

    assert not [e for e in events if e["type"] == "token"], "search results are not the answer"


def test_a_debug_checkpoint_does_not_leak_the_state():
    """A debug checkpoint carries the whole state, question included."""
    events, _ = run([checkpoint(["agent_node"], {"initial_query": QUESTION, "messages": []})])

    assert not events, f"a checkpoint must produce nothing to forward, got {events}"


def test_a_debug_task_does_not_leak_its_input():
    events, _ = run([task("agent_node")])

    for event in events:
        assert QUESTION not in str(event), "the task input carries the question"
        assert "input" not in event


def test_the_intent_router_answer_is_not_printed_as_the_answer():
    """The router answers with a structured payload for the graph, not prose."""
    events, _ = run([ai_message('{"question_relevancy": true}', node="intent_router_node")])

    assert not [e for e in events if e["type"] == "token"]


def test_reasoning_content_never_becomes_a_token():
    from langchain_core.messages import AIMessageChunk as Chunk

    from api.agents.agent import _answer_text

    chunk = Chunk(
        content="",
        content_blocks=[
            {"type": "reasoning", "reasoning": "the user wants a washer, I should search", "id": "r1"},
            {"type": "text", "text": "Here are the machines."},
        ],
    )

    emitted = _answer_text(chunk, {"langgraph_node": "model"})

    assert emitted == "Here are the machines."
    assert "search" not in emitted


def test_the_router_never_becomes_a_token():
    """Its answer is a structured payload for the graph, not prose."""
    from api.agents.agent import _answer_text

    from langchain_core.messages import AIMessage as Message

    assert _answer_text(Message(content='{"question_relevancy": true}'), {"langgraph_node": "intent_router_node"}) == ""


def test_a_running_node_is_announced_once():
    events, _ = run([task("agent_node"), task("agent_node"), task("intent_router_node")])

    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert len(statuses) == 2, f"each node is announced once, got {statuses}"


def test_a_status_is_a_sentence_and_carries_no_graph_structure():
    events, _ = run([task("agent_node")])

    status = next(e for e in events if e["type"] == "status")
    assert set(status) == {"type", "text"}
    assert "agent_node" not in status["text"], "a node name is not a sentence for a user"


def test_a_task_result_is_not_announced():
    """Only the start of a task is news; its result is not."""
    events, _ = run([("debug", {"type": "task_result", "payload": {"name": "agent_node"}})])

    assert not [e for e in events if e["type"] == "status"]


def test_an_unnamed_task_produces_nothing():
    events, _ = run([("debug", {"type": "task", "payload": {}})])

    assert not [e for e in events if e["type"] == "status"]


def test_a_turn_produces_one_result_even_with_many_state_snapshots():
    """LangGraph repeats `values` after every step; the answer must not repeat."""
    events, _ = run([
        values(initial_query=QUESTION, messages=[]),
        values(initial_query=QUESTION, question_relevancy=True, answer=ANSWER),
        values(initial_query=QUESTION, question_relevancy=True, answer=ANSWER),
        values(initial_query=QUESTION, question_relevancy=True, answer=ANSWER),
    ])

    results = [e for e in events if e["type"] == "result"]
    assert len(results) == 1


def test_a_state_with_no_answer_yields_no_result():
    events, _ = run([values(initial_query=QUESTION, messages=[])])

    assert not [e for e in events if e["type"] == "result"]


def test_the_result_carries_what_the_caller_made_of_the_state():
    graph = FakeGraph([values(answer=ANSWER)])

    out = list(stream_graph_events(graph, {}, {}, lambda state: {"answer": state["answer"], "n": 1}))

    assert out == [{"type": "result", "payload": {"answer": ANSWER, "n": 1}}]


def test_all_the_modes_are_requested_in_one_pass():
    _, graph = run([])

    assert set(graph.calls[0]["modes"]) == {"values", "messages", "debug", "custom"}
    assert set(STREAM_MODES) == {"values", "messages", "debug", "custom"}


def test_the_state_and_config_are_passed_through():
    state = {"initial_query": QUESTION}
    config = {"configurable": {"thread_id": "t1"}}
    graph = FakeGraph([])

    list(stream_graph_events(graph, state, config, lambda s: s))

    assert graph.calls[0]["state"] is state
    assert graph.calls[0]["config"] is config


@pytest.mark.parametrize("mode", ["values", "messages", "debug"])
def test_a_single_mode_stream_still_works(mode):
    """LangGraph yields bare payloads when only one mode is asked for."""
    graph = FakeGraph([ANSWER])

    out = list(stream_graph_events(graph, {}, {}, lambda s: s, modes=(mode,)))

    assert isinstance(out, list)
