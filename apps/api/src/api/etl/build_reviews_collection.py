import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Sequence

import cohere
import tiktoken
from qdrant_client import QdrantClient, models

from api.core.settings import get_settings

logger = logging.getLogger(__name__)

REVIEW_FILE = "data/Appliances.jsonl"
CACHE_FILE = "data/_reviews_for_items.jsonl"
PROGRESS_FILE = "data/_reviews_upsert_progress.json"

ITEMS_COLLECTION = "Amazon-items-collection-01-hybrid"
REVIEWS_COLLECTION = "Amazon-reviews-collection-01-hybrid"
REVIEW_NAMESPACE = uuid.UUID("6f1d4c7a-2b8e-4f3a-9c5d-0e7a1b2c3d4e")

DENSE_VECTOR = "text-embedding-3-small"
SPARSE_VECTOR = "bm25"
SPARSE_MODEL = "Qdrant/bm25"

MAX_REVIEW_CHARS = 1000
MAX_REVIEW_TOKENS = 512

EMBED_BATCH = 50
UPSERT_BATCH = 100
SLEEP_BETWEEN_BATCHES = 0.35

CHUNK_ROWS = 50_000

ENCODING = tiktoken.encoding_for_model("text-embedding-3-small")


def build_review_text(title: str, text: str, max_chars: int = MAX_REVIEW_CHARS) -> str:
    joined = " ".join(part for part in (title, text) if part)
    joined = " ".join(joined.split())
    if not joined:
        return ""

    if len(joined) > max_chars:
        joined = joined[:max_chars]

    tokens = ENCODING.encode(joined)
    if len(tokens) > MAX_REVIEW_TOKENS:
        joined = ENCODING.decode(tokens[:MAX_REVIEW_TOKENS])

    return joined.strip()


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def items_parent_asins(client: QdrantClient) -> set:
    """Every distinct product in the items collection."""
    asins: set = set()
    offset = None
    while True:
        points, offset = client.scroll(
            ITEMS_COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        for point in points:
            asin = point.payload.get("parent_asin")
            if asin:
                asins.add(asin)
        if offset is None:
            break
    return asins


def extract(force: bool = False, per_product: int = 1) -> Path:
    cache = Path(CACHE_FILE)
    if cache.exists() and not force:
        with cache.open(encoding="utf-8") as handle:
            existing = sum(1 for _ in handle)
        logger.info("cache already holds %s reviews, pass --force to rebuild", existing)
        return cache

    import pandas as pd

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    asins = items_parent_asins(client)
    logger.info(
        "items collection has %s distinct products; keeping %s review(s) each",
        f"{len(asins):,}",
        per_product,
    )

    kept: List[dict] = []
    read = 0

    with pd.read_json(
        REVIEW_FILE,
        lines=True,
        chunksize=CHUNK_ROWS,
        dtype={"parent_asin": "string", "title": "string", "text": "string", "helpful_vote": "Int64"},
    ) as reader:
        for frame in reader:
            read += len(frame)
            mine = frame[frame["parent_asin"].isin(asins)]
            if mine.empty:
                continue
            mine = mine.assign(
                preprocessed_data=[
                    build_review_text(title, text)
                    for title, text in zip(mine["title"].fillna(""), mine["text"].fillna(""))
                ]
            )
            mine = mine[mine["preprocessed_data"].str.len() > 0]
            if mine.empty:
                continue
            mine = mine.sort_values("helpful_vote", ascending=False, na_position="last")
            kept.append(
                mine[["parent_asin", "preprocessed_data"]]
                .drop_duplicates(subset=["parent_asin", "preprocessed_data"])
            )
            logger.info("  %s reviews matched of %s read", f"{len(kept) and sum(len(f) for f in kept):,}", f"{read:,}")

    if not kept:
        raise RuntimeError("no reviews matched the items collection")

    combined = pd.concat(kept, ignore_index=True)
    combined = combined.drop_duplicates(subset=["parent_asin", "preprocessed_data"])

    if per_product:
        combined = combined.groupby("parent_asin", sort=False).head(per_product)

    with cache.open("w", encoding="utf-8") as out:
        for row in combined.itertuples(index=False):
            out.write(
                json.dumps(
                    {"preprocessed_data": row.preprocessed_data, "parent_asin": str(row.parent_asin)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        "extracted %s reviews across %s products of %s read",
        f"{len(combined):,}",
        f"{combined['parent_asin'].nunique():,}",
        f"{read:,}",
    )
    return cache


def get_embedding(text: str) -> List[float]:
    """One vector for one text, in the items collection's own space."""
    return get_embeddings_batch([text])[0]


def get_embeddings_batch(texts: Sequence[str]) -> List[List[float]]:
    settings = get_settings()
    key = os.getenv(settings.embedding.api_key_env)
    if not key:
        raise RuntimeError(f"{settings.embedding.api_key_env} is not set in .env")

    client = cohere.ClientV2(api_key=key)
    response = client.embed(
        model=settings.embedding.model,
        inputs=[{"content": [{"type": "text", "text": text}]} for text in texts],
        input_type=settings.embedding.input_type,
        output_dimension=settings.embedding.dimensions,
        embedding_types=settings.embedding.embedding_types,
    )
    return [list(vector) for vector in response.embeddings.float]


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    if recreate and client.collection_exists(REVIEWS_COLLECTION):
        logger.warning("deleting existing %s", REVIEWS_COLLECTION)
        client.delete_collection(REVIEWS_COLLECTION)

    if not client.collection_exists(REVIEWS_COLLECTION):
        client.create_collection(
            collection_name=REVIEWS_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=get_settings().embedding.dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("created %s", REVIEWS_COLLECTION)
    else:
        logger.info("%s already exists", REVIEWS_COLLECTION)

    client.create_payload_index(
        collection_name=REVIEWS_COLLECTION,
        field_name="parent_asin",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _sparse_model():
    global _SPARSE
    if _SPARSE is None:
        from fastembed import SparseTextEmbedding

        _SPARSE = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=".cache/fastembed")
    return _SPARSE


_SPARSE = None


def _sparse(texts: Sequence[str]) -> List[models.SparseVector]:
    """The bm25 vectors for a batch of texts, computed locally."""
    return list(_sparse_model().embed(list(texts)))


def _point_id(record: dict) -> str:
    return str(uuid.uuid5(REVIEW_NAMESPACE, f"{record['parent_asin']}|{record['preprocessed_data']}"))


def _to_points(records: Sequence[dict], vectors: Sequence[Sequence[float]]) -> List[models.PointStruct]:
    sparse = _sparse([record["preprocessed_data"] for record in records])
    points = []
    for record, dense, sparse_vector in zip(records, vectors, sparse):
        points.append(
            models.PointStruct(
                id=_point_id(record),
                vector={
                    DENSE_VECTOR: list(dense),
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(sparse_vector.indices),
                        values=list(sparse_vector.values),
                    ),
                },
                payload={
                    "preprocessed_data": record["preprocessed_data"],
                    "parent_asin": record["parent_asin"],
                },
            )
        )
    return points


def read_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_progress() -> dict:
    path = Path(PROGRESS_FILE)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {"upserted": 0}


def _save_progress(state: dict) -> None:
    Path(PROGRESS_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")


def upsert(limit: int | None = None, wait: float = SLEEP_BETWEEN_BATCHES, recreate: bool = False) -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, recreate=recreate)

    cache = Path(CACHE_FILE)
    if not cache.exists():
        raise RuntimeError(f"{cache} not found, run --step extract first")

    state = _load_progress()
    start = state["upserted"]
    logger.info("resuming after %s reviews", f"{start:,}")

    batch: List[dict] = []
    pending: List[models.PointStruct] = []
    consumed = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        client.upsert(collection_name=REVIEWS_COLLECTION, points=pending, wait=True)
        pending = []

    def checkpoint() -> None:
        state["upserted"] = consumed
        _save_progress(state)
        if consumed % 1000 < EMBED_BATCH:
            logger.info("  upserted %s of %s reviews", f"{consumed:,}", f"{total:,}")

    total = sum(1 for _ in read_records(cache))
    skipped = 0

    for record in read_records(cache):
        if consumed < start:
            consumed += 1
            skipped += 1
            continue
        if limit is not None and consumed - start >= limit:
            break

        batch.append(record)
        if len(batch) < EMBED_BATCH:
            continue

        vectors = get_embeddings_batch([item["preprocessed_data"] for item in batch])
        pending.extend(_to_points(batch, vectors))
        consumed += len(batch)
        batch.clear()

        if len(pending) >= UPSERT_BATCH:
            flush()
            checkpoint()
            time.sleep(wait)

    if batch:
        vectors = get_embeddings_batch([item["preprocessed_data"] for item in batch])
        pending.extend(_to_points(batch, vectors))
        consumed += len(batch)
    flush()
    checkpoint()
    logger.info(
        "finished: %s reviews upserted in total (%s skipped as already done)",
        f"{consumed:,}",
        f"{skipped:,}",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["extract", "upsert", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="stop after N reviews")
    parser.add_argument(
        "--per-product",
        type=int,
        default=1,
        help="how many reviews each product contributes (0 = all of them)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild the extracted cache")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    if args.step in ("extract", "all"):
        extract(force=args.force, per_product=args.per_product)
    if args.step in ("upsert", "all"):
        upsert(limit=args.limit, recreate=args.recreate)


if __name__ == "__main__":
    main()
