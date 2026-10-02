from typing import List

from langchain.tools import tool
from qdrant_client.conversions.common_types import Document, Prefetch
from qdrant_client.http.models import FusionQuery
from qdrant_client.models import FieldCondition, Filter, MatchAny

from api.agents.retrieval import generate_embedding, qdrant_client, retrieve_data
from api.core.settings import get_settings

REVIEWS_COLLECTION = "Amazon-reviews-collection-01-hybrid"
DENSE_VECTOR = "text-embedding-3-small"
SPARSE_VECTOR = "bm25"

DESCRIPTION_CHARS = 400
REVIEW_CHARS = 600


def _clip(text, limit: int) -> str:
    """Shorten to `limit` characters, on a word boundary where there is one."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return f"{cut}..."


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
        f"[{product_id}] (rating: {rating}) {_clip(description, DESCRIPTION_CHARS)}"
        for product_id, description, rating in zip(
            found["retrieved_context_ids"],
            found["retrieved_context"],
            found["retrieved_context_ratings"],
        )
    ]
    if not products:
        return "No products matched this search. Try a different or broader wording."
    return "\n\n".join(products)


@tool
def retrieve_reviews_data(query: str, item_list: List[str], k: int = 5) -> str:
    """Read what buyers said about specific products.

    Args:
        query: What to look for across the reviews.
        item_list: Product ids to read reviews for.
        k: How many reviews to return.
    """
    ids = [str(value).strip() for value in (item_list or []) if str(value).strip()]
    if not ids:
        return "No product ids were given, so there are no reviews to read."

    wanted = min(int(k or 5), 10)

    try:
        results = qdrant_client.query_points(
            collection_name=REVIEWS_COLLECTION,
            prefetch=[
                Prefetch(
                    query=generate_embedding(query),
                    using=DENSE_VECTOR,
                    limit=wanted * 4,
                    filter=Filter(
                        must=[FieldCondition(key="parent_asin", match=MatchAny(any=ids))]
                    ),
                ),
                Prefetch(
                    query=Document(text=query, model="qdrant/bm25"),
                    using=SPARSE_VECTOR,
                    limit=wanted * 4,
                    filter=Filter(
                        must=[FieldCondition(key="parent_asin", match=MatchAny(any=ids))]
                    ),
                ),
            ],
            query=FusionQuery(fusion=get_settings().retrieval.fusion),
            limit=wanted,
            with_payload=True,
        )
    except Exception as error:
        return f"Reviews are unavailable right now ({error}). Answer from the products alone."

    if not results.points:
        return f"No buyer reviews were found for {', '.join(ids)}."

    lines = []
    for point in results.points:
        payload = point.payload or {}
        lines.append(
            f"[{payload.get('parent_asin')}] customer review: "
            f"{_clip(payload.get('preprocessed_data'), REVIEW_CHARS)}"
        )
    return "\n\n".join(lines)
