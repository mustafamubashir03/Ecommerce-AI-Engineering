import pytest

from api.agents import agent as agent_module
from api.agents.prompts import prompt

RELEVANT = [
    "Can you suggest best bathroom products to buy?",
    "What bathroom products do you recommend?",
    "Which washing machine should I buy?",
    "What do buyers complain about with portable washing machines?",
    "Which of these products is better for a small apartment?",
    "Do you have anything suitable for a small kitchen?",
    "what should I get for a first apartment",
    "is anything in stock under 30 dollars",
    "how reliable are these",
    "what came in last month",
    "summarise what the reviews say",
    "tell me more about the second one",
]

IRRELEVANT = [
    "Who won the 1998 World Cup?",
    "What is the capital of France?",
    "Write a Python web scraper for me.",
    "Explain how a diesel engine works.",
    "Who was the first person to reach the South Pole?",
    "draft an email to my landlord",
]


def test_the_prompt_names_the_positive_categories():
    """Each of these was a question the gate used to get wrong."""
    text = prompt("intent_router").lower()
    for topic in ("recommend", "compar", "buy", "available", "in stock", "reviews", "complaint"):
        assert topic in text, f"the prompt never mentions {topic}"


def test_the_prompt_names_the_negative_categories():
    """Without these, 'not relevant' has no meaning and drifts."""
    text = prompt("intent_router").lower()
    assert "history" in text
    assert "sport" in text or "geography" in text
    assert "code" in text or "programming" in text


def test_the_prompt_treats_wording_as_secondary():
    text = prompt("intent_router").lower()
    assert "wording" in text, "the prompt must say the wording is secondary"
    assert "amazon" in text, "the wording example that used to be missing"


def test_the_prompt_breaks_ties_towards_relevant():
    """The conservative direction is the whole point of a gate on a catalogue."""
    text = prompt("intent_router").lower()
    assert "either way" in text or "doubt" in text
    assert "refusing" in text.lower(), "it must say refusing is the worse error"


def test_the_prompt_describes_the_both_json_fields():
    text = prompt("intent_router")
    assert "question_relevancy" in text
    assert "answer" in text


def test_the_schema_describes_the_subject_matter_not_the_coverage():
    field = agent_module.IntentRouterResponse.model_fields["question_relevancy"]
    described = (field.description or "").lower()
    assert "asking about" in described, "must describe the subject matter"
    assert "can be answered" not in described, "must not describe coverage"


def test_the_schema_still_takes_a_reason_only_when_irrelevant():
    field = agent_module.IntentRouterResponse.model_fields["answer"]
    described = (field.description or "").lower()
    assert "empty when relevant" in described
    assert "why" in described


@pytest.fixture(scope="module")
def classify():
    import os

    import api.agents.agent as agent

    if os.environ.get("RUN_LIVE_INTENT_TESTS", "").strip().lower() not in ("1", "true", "yes"):
        pytest.skip("set RUN_LIVE_INTENT_TESTS=1 to classify against the live model")

    def ask(question: str, repeats: int = 1):
        verdicts = []
        for _ in range(repeats):
            out = agent.intent_router_node({"initial_query": question})
            verdicts.append(out["question_relevancy"])
        return verdicts

    try:
        probe = ask("do you have bedsheets?")
    except Exception as error:
        pytest.skip(f"the configured model could not be reached: {type(error).__name__}")

    if not probe:
        pytest.skip("the configured model returned nothing to classify")
    return ask


@pytest.mark.parametrize("question", RELEVANT)
def test_a_product_question_stays_relevant(classify, question):
    assert all(classify(question)), f"refused a product question: {question!r}"


@pytest.mark.parametrize("question", IRRELEVANT)
def test_an_unrelated_question_stays_irrelevant(classify, question):
    assert not any(classify(question)), f"accepted an unrelated question: {question!r}"


@pytest.mark.parametrize(
    "question",
    [
        "Can you suggest best bathroom products to buy?",
        "What do buyers complain about with portable washing machines?",
    ],
)
def test_the_same_question_gets_the_same_answer_every_time(classify, question):
    verdicts = classify(question, repeats=5)
    assert len(set(verdicts)) == 1, f"unstable across five identical calls: {verdicts}"


class _Msg:
    def __init__(self, content):
        self.content = content


class _PlainJsonModel:
    """Stands in for a model that answers in text rather than a tool call."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _Msg(self.reply)


def install_plain_json(monkeypatch, reply, error=None):
    from instructor.core.exceptions import ResponseParsingError

    class _Completions:
        def __init__(self):
            self.seen = []

        def create(self, **kwargs):
            self.seen.append(kwargs)
            if error is not None:
                raise error
            raise ResponseParsingError("no tool call in the response")

    class _Instructor:
        def __init__(self, completions):
            self.chat = type("Chat", (), {"completions": completions})()

    completions = _Completions()
    monkeypatch.setattr(agent_module, "get_instructor_client", lambda: _Instructor(completions))
    model = _PlainJsonModel(reply)
    monkeypatch.setattr(agent_module, "get_chat_model", lambda: model)
    return completions, model


def test_the_fallback_reads_relevant_out_of_plain_json(monkeypatch):
    """The path that handles a model with no tool calling must keep working."""
    from instructor.core.exceptions import ResponseParsingError

    completions, model = install_plain_json(
        monkeypatch,
        '{"question_relevancy": true, "answer": ""}',
        error=ResponseParsingError("no tool call"),
    )

    out = agent_module.intent_router_node({"initial_query": "Can you suggest best bathroom products to buy?"})

    assert out == {"question_relevancy": True, "answer": ""}
    assert model.calls == 1, "the fallback asks the same model once more"
    assert len(completions.seen) == 1


def test_the_fallback_reads_irrelevant_out_of_plain_json(monkeypatch):
    from instructor.core.exceptions import ResponseParsingError

    install_plain_json(
        monkeypatch,
        'Here you go: {"question_relevancy": false, "answer": "That is a sports question."}',
        error=ResponseParsingError("no tool call"),
    )

    out = agent_module.intent_router_node({"initial_query": "Who won the 1998 World Cup?"})

    assert out["question_relevancy"] is False
    assert "sports" in out["answer"]


def test_unreadable_plain_text_is_still_treated_as_relevant(monkeypatch):
    """Unchanged behaviour, and it is the conservative direction."""
    from instructor.core.exceptions import ResponseParsingError

    install_plain_json(monkeypatch, "I would rather not say", error=ResponseParsingError("no tool call"))

    out = agent_module.intent_router_node({"initial_query": "Can you suggest best bathroom products to buy?"})

    assert out["question_relevancy"] is True


def test_a_plain_json_model_is_only_asked_once_more(monkeypatch):
    from instructor.core.exceptions import ResponseParsingError

    _, model = install_plain_json(
        monkeypatch,
        '{"question_relevancy": true, "answer": ""}',
        error=ResponseParsingError("no tool call"),
    )

    agent_module.intent_router_node({"initial_query": "anything?"})

    assert model.calls == 1, "the fallback must not become a loop"
