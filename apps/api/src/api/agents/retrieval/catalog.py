"""The catalogue itself: embeddings, hybrid search, and stored product payloads.

This is the only place that talks to Qdrant or to the embedding provider. What
the rest of the application needs is deliberately narrow: `retrieve_data` for
the search results, and `fetch_product_payloads` to turn a product id into the
fields the UI shows.
"""

import os

import cohere
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Document, Prefetch
from qdrant_client.http.models import FusionQuery
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.core.settings import get_settings

_settings = get_settings()
_retrieval = _settings.retrieval
_fields = _retrieval.fields

qdrant_client = QdrantClient(url=_settings.qdrant_url)
_cohere = cohere.ClientV2(api_key=os.getenv(_settings.embedding.api_key_env))


@traceable(name="embed_query", run_type="embedding")
def generate_embedding(text: str) -> list:
    """One vector for one query, in the model's own configured dimensions."""
    response = _cohere.embed(
        model=_settings.embedding.model,
        inputs=[{"content": [{"type": "text", "text": text}]}],
        input_type=_settings.embedding.input_type,
        output_dimension=_settings.embedding.dimensions,
        embedding_types=_settings.embedding.embedding_types,
    )
    return response.embeddings.float[0]


@traceable(name="retrieving_data", run_type="retriever")
def retrieve_data(query: str, k: int | None = None) -> dict:
    """Hybrid search: dense vectors + BM25, fused with reciprocal rank fusion.

    The catalogue stores several chunks per product, so fusing dense and sparse
    hits can surface the same product more than once. Points are collapsed to one
    per product here, keeping the best score, so the model and the UI both see
    each product once and the requested count is worth asking for.
    """
    k = k or _retrieval.top_k
    limit = _retrieval.prefetch_limit
    results = qdrant_client.query_points(
        collection_name=_retrieval.collection,
        prefetch=[
            Prefetch(query=generate_embedding(query), using=_retrieval.dense_vector, limit=limit),
            Prefetch(
                query=Document(text=query, model=_retrieval.sparse_model),
                using=_retrieval.sparse_vector,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=_retrieval.fusion),
        limit=limit,
    )

    unique: dict[str, dict] = {}
    for point in results.points:
        payload = point.payload or {}
        product_id = payload.get(_fields.id)
        if not product_id or product_id in unique:
            continue
        unique[product_id] = {
            "id": product_id,
            "description": payload.get(_fields.description),
            "rating": payload.get(_fields.rating),
            "score": point.score,
        }

    ordered = list(unique.values())[:k]
    return {
        "retrieved_context_ids": [item["id"] for item in ordered],
        "retrieved_context": [item["description"] for item in ordered],
        "similarity_scores": [item["score"] for item in ordered],
        "retrieved_context_ratings": [item["rating"] for item in ordered],
    }


def fetch_product_payloads(product_ids) -> dict:
    """Fetch the stored payload for each product id (order preserved, no dupes).

    Uses scroll with a filter: this collection only declares named vectors, so a
    vector-less lookup is both correct and cheaper than a search.
    """
    payloads: dict[str, dict] = {}

    seen: set = set()
    for product_id in product_ids or []:
        if not product_id or product_id in seen:
            continue
        seen.add(product_id)
        records, _ = qdrant_client.scroll(
            collection_name=_retrieval.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=_fields.id,
                        match=MatchValue(value=product_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            continue
        payloads[product_id] = records[0].payload or {}

    return payloads


def hydrate_used_context(references) -> list:
    """Turn (id, description) references into UI-ready used-context items.

    The catalogue stores several chunks per product, so a search can return the
    same product more than once. Each product is therefore emitted once, in the
    order it was first referenced, and an empty description falls back to the
    stored product description.
    """
    references = list(references or [])
    payloads = fetch_product_payloads([item[0] for item in references])

    used_context = []
    seen: set = set()
    for product_id, description in references:
        if product_id in seen:
            continue
        payload = payloads.get(product_id)
        if not payload:
            continue
        # No image means the card would render blank, so the product is left out
        # rather than shown as an empty box.
        image_url = payload.get(_fields.image)
        if not image_url:
            continue
        seen.add(product_id)
        used_context.append(
            {
                "id": product_id,
                "image_url": image_url,
                "price": payload.get(_fields.price),
                "description": description or str(payload.get(_fields.description, ""))[:200],
                "rating": payload.get(_fields.rating),
            }
        )
    return used_context
