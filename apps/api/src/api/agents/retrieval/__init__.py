"""Retrieval, split by what it does rather than by which file calls it.

`catalog`   the Qdrant and embedding calls, and turning ids into UI products
`pipeline`  retrieve, format, prompt, answer: the direct `/rag/` flow
"""

from api.agents.retrieval.catalog import (
    fetch_product_payloads,
    generate_embedding,
    hydrate_used_context,
    qdrant_client,
    retrieve_data,
)
from api.agents.retrieval.pipeline import (
    build_prompt,
    generate_answer,
    process_context,
    rag_pipeline,
)
from api.agents.retrieval.prompts import render_prompt, resolve_prompt_path

__all__ = [
    "build_prompt",
    "fetch_product_payloads",
    "generate_answer",
    "generate_embedding",
    "hydrate_used_context",
    "process_context",
    "qdrant_client",
    "rag_pipeline",
    "render_prompt",
    "resolve_prompt_path",
    "retrieve_data",
]
