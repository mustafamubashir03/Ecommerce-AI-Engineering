import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from api.agents.agent import _compact_history
from api.agents.tools import DESCRIPTION_CHARS, REVIEW_CHARS, _clip


def turn(question: str, tool_output: str, answer: str = "Here are some options. [B0ABC12345]"):
    """One completed turn, as the checkpointer would hold it."""
    return [
        HumanMessage(content=question),
        AIMessage(
            content="",
            tool_calls=[{"name": "retrieve_data", "args": {"query": question}, "id": "1"}],
        ),
        ToolMessage(content=tool_output, tool_call_id="1"),
        AIMessage(content=answer),
    ]


BIG = "x " * 40_000


def test_a_short_conversation_is_left_alone():
    messages = turn("do you have bedsheets?", "one product")
    assert _compact_history(messages) == messages


def test_an_empty_history_is_left_alone():
    assert _compact_history([]) == []


def test_older_turns_do_not_accumulate():
    """The whole point: the prompt must not grow by a tool result per question."""
    history = []
    for index in range(6):
        history.extend(turn(f"question {index}", BIG))

    compacted = _compact_history(history)

    kept_tool_results = [m for m in compacted if isinstance(m, ToolMessage)]
    assert len(kept_tool_results) == 1, (
        f"only the current turn's tool result may survive, kept {len(kept_tool_results)}"
    )


def test_the_prompt_does_not_grow_with_the_conversation():
    """The measurement that matters: size after turn 1 vs size after turn 10."""
    first = _compact_history(turn("q0", BIG))
    later = _compact_history([m for i in range(10) for m in turn(f"q{i}", BIG)])

    def size(messages):
        return sum(len(str(m.content)) for m in messages)

    assert size(later) < size(first) * 1.2, (
        f"ten turns ({size(later):,}) should cost about what one does ({size(first):,})"
    )


def test_earlier_questions_are_kept_for_context():
    """A follow-up is only meaningful if the model knows what it follows."""
    history = []
    for index in range(3):
        history.extend(turn(f"question {index}", BIG))

    compacted = _compact_history(history)
    questions = [m.content for m in compacted if isinstance(m, HumanMessage)]

    assert questions == ["question 0", "question 1", "question 2"]


def test_the_current_turn_is_never_broken_up():
    """A tool result without its call is a malformed request, not a short one."""
    history = [*turn("older question", BIG), *turn("current question", BIG)]
    compacted = _compact_history(history)

    tool_results = [m for m in compacted if isinstance(m, ToolMessage)]
    assert len(tool_results) == 1, "only the current turn's tool result survives"
    assert tool_results[0].content == BIG, "and it is passed through untouched"

    current = [i for i, m in enumerate(compacted) if getattr(m, "content", None) == "current question"]
    assert current, "the current question is kept"
    assert current[0] < compacted.index(tool_results[0]), "and it comes before its own result"


def test_a_retried_question_collapses_to_one_copy():
    """Retrying used to append the same question again on every attempt."""
    history = []
    for _ in range(3):
        history.extend(turn("Can you suggest best bathroom products?", BIG))

    questions = [m.content for m in _compact_history(history) if isinstance(m, HumanMessage)]

    assert questions == ["Can you suggest best bathroom products?"]


def test_a_question_asked_again_later_is_still_kept():
    """Only a repeat in a row is collapsed; coming back to it is real context."""
    history = []
    for index in range(3):
        history.extend(turn(f"question {index}", BIG))
    history.extend(turn("question 0", BIG))

    questions = [m.content for m in _compact_history(history) if isinstance(m, HumanMessage)]

    assert questions == ["question 0", "question 1", "question 2", "question 0"]


def test_history_without_a_question_is_left_alone():
    """Nothing to anchor on, so trimming would risk dropping a tool result."""
    messages = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
    ]
    assert _compact_history(messages) == messages


def test_a_long_description_is_clipped():
    assert len(_clip("word " * 5000, DESCRIPTION_CHARS)) < DESCRIPTION_CHARS + 10


def test_clipping_keeps_the_start_of_the_text():
    """The id and the opening of the description are what the answer needs."""
    clipped = _clip("This is a very long catalogue description. " * 100, DESCRIPTION_CHARS)
    assert clipped.startswith("This is a very long catalogue description")


def test_a_short_description_is_untouched():
    assert _clip("quiet and compact", DESCRIPTION_CHARS) == "quiet and compact"


def test_clipping_survives_empty_and_missing_text():
    for value in ("", "   ", None):
        assert _clip(value, DESCRIPTION_CHARS) == ""


def test_clipping_normalises_runs_of_whitespace():
    assert _clip("a\n\n  b\tc", REVIEW_CHARS) == "a b c"


def test_a_clipped_product_still_carries_its_id():
    """Without the id the model cannot cite it, so the bound must not eat it."""
    text = _clip("B0ABC12345 " + "description " * 5000, DESCRIPTION_CHARS)
    assert "B0ABC12345" in text


@pytest.mark.parametrize("limit", [DESCRIPTION_CHARS, REVIEW_CHARS])
def test_a_clipped_product_is_a_fraction_of_the_original(limit):
    original = "word " * 5000
    assert len(_clip(original, limit)) <= limit + 4
