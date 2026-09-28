"""Everything the application retrieves from the catalogue.

Kept as this module because the graph's tool, the `/rag/` endpoint and the
hydration step all import from here, and because those call sites should not
have to know which part of retrieval they are using. The parts themselves live
in the `retrieval` package next to it.
"""

from typing import List

from api.agents.retrieval import (
    fetch_product_payloads,
    generate_embedding,
    hydrate_used_context,
    process_context,
    qdrant_client,
    rag_pipeline,
    render_prompt,
    resolve_prompt_path,
    retrieve_data,
)

__all__ = [
    "fetch_product_payloads",
    "generate_embedding",
    "hydrate_used_context",
    "process_context",
    "qdrant_client",
    "rag_pipeline",
    "rag_pipeline_wrapper",
    "render_prompt",
    "resolve_prompt_path",
    "retrieve_data",
]


def rag_pipeline_wrapper(question: str, top_k: int = 5) -> dict:
    """The `/rag/` endpoint's shape: an answer plus the products behind it.

    The agent's own turn does not go through here; it retrieves with the same
    `retrieve_data` and hydrates with the same `hydrate_used_context`.
    """
    result = rag_pipeline(question, top_k=top_k)
    used_context = hydrate_used_context(
        [(item.id, item.description) for item in result.get("references", [])]
    )
    return {"answer": result["answer"], "used_context": used_context}
