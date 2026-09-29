"""Tool exposed to the shopping assistant agent.

`retrieve_data_tool` wraps the hybrid search from `retrieval_generation` and
returns plain text, which is what a tool must return to be fed back to the
model. Each product keeps its id in front of its text so the agent can quote it
and the graph can look up its image and price.
"""

from langchain.tools import tool

from api.core.settings import get_settings
from api.agents.retrieval import retrieve_data


@tool
def retrieve_data_tool(query: str, k: int = 0) -> str:
    """Search the product catalogue for products in stock.

    Args:
        query: What the products should have in common.
        k: How many products to return. 0 uses the configured default.
    """
    retrieval = get_settings().retrieval
    top_k = k or retrieval.top_k

    try:
        found = retrieve_data(query, min(int(top_k), retrieval.max_top_k))
    except Exception as error:
        return f"Product search is unavailable right now ({error}). Answer without it."

    products = [
        f"[{product_id}] (rating: {rating}) {description}"
        for product_id, description, rating in zip(
            found["retrieved_context_ids"],
            found["retrieved_context"],
            found["retrieved_context_ratings"],
        )
    ]
    if not products:
        return "No products matched this search. Try a different or broader wording."
    return "\n\n".join(products)
