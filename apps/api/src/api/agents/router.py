"""First node of the graph: is this question about the catalogue at all?"""

from langsmith import traceable

from api.agents.llm import get_chat_model
from api.agents.prompts import IntentRouterResponse, prompt
from api.agents.text import optional


@traceable(name="intent_router", run_type="llm")
def intent_router_node(state) -> dict:
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
    return "agent_node" if state.get("question_relevancy") else "end"
