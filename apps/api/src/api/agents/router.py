"""First node of the graph: is this question about the catalogue at all?"""

from langsmith import traceable

from api.agents.llm import get_chat_model
from api.agents.prompts import IntentRouterResponse, prompt
from api.agents.text import optional


@traceable(name="intent_router", run_type="llm")
def intent_router_node(state) -> dict:
    """Structured output doubles as the routing decision.

    No cooldown check here on purpose: the model boundary is where a provider
    that is known to be unavailable is skipped, and where a fallback provider is
    used instead. Raising here would strand the request even though another
    provider could answer it.

    `method="function_calling"` is explicit because the graph streams this call
    and Groq documents that structured outputs cannot be combined with
    streaming. A tool call carries the same schema and is validated the same
    way, on every provider here.
    """
    response = get_chat_model().with_structured_output(
        IntentRouterResponse, method="function_calling"
    ).invoke(
        [
            {"role": "system", "content": prompt("intent_router")},
            {"role": "user", "content": optional(state.get("initial_query"))},
        ]
    )
    return {
        "question_relevancy": response.question_relevancy,
        "answer": response.answer,
    }


def intent_router_condition_edges(state) -> str:
    """Relevant questions go to the agent, everything else ends here.

    Both branches return the keys of the conditional edge map, so this must
    return "end" rather than langgraph's END. END is the *destination* those
    keys point at; returning it directly is not a key of the map, and the graph
    raises KeyError('__end__') instead of finishing the turn.
    """
    return "agent_node" if state.get("question_relevancy") else "end"
