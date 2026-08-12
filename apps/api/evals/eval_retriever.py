"""
eval_retriever.py
-----------------
LangSmith evaluation pipeline for the RAG retrieval-generation pipeline.

Evaluators implemented:
  - ragas_faithfulness : checks whether the answer is grounded in the retrieved context
  - context_recall     : checks whether the retrieved context covers the reference answer

Run via Makefile:
    make run-eval-retriever
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langsmith import Client, evaluate
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.context_recall.metric import ContextRecall

# api.agents is resolvable because PYTHONPATH includes apps/api/src (set in Makefile)
from api.agents.retrieval_generation import rag_pipeline

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("eval_retriever")

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
langsmith_client = Client()

# AsyncOpenAI is required by ragas.metrics.collections (InstructorLLM / agenerate)
async_openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
ragas_llm = llm_factory("gpt-4o-mini", client=async_openai_client)


# ---------------------------------------------------------------------------
# Target function — called once per LangSmith example
# ---------------------------------------------------------------------------
# Production rag_pipeline (retrieval_generation.py) returns:
#   { "answer", "question", "retrieved_context_ids", "retrieved_context",
#     "similarity_scores", "retrieved_context_ratings" }
# All keys match what the RAGAS evaluators expect — no aliasing needed.
# ---------------------------------------------------------------------------
def predict(inputs: dict) -> dict:
    """LangSmith target: receives dataset inputs, returns pipeline output."""
    query = inputs["question"]
    log.info("Running RAG pipeline for query: %r", query)

    result = rag_pipeline(query)
    print(result)
    print(result.keys())

    log.info(
        "Retrieved %d docs — IDs: %s — scores: %s",
        len(result["retrieved_context_ids"]),
        result["retrieved_context_ids"],
        [round(s, 4) for s in result["similarity_scores"]],
    )
    log.info("Generated answer (first 200 chars): %s", result["answer"][:200])

    # Pass through directly — production keys already match evaluator expectations
    return result


# ---------------------------------------------------------------------------
# RAGAS evaluators (async core)
# ---------------------------------------------------------------------------
async def _ragas_faithfulness_async(run, example) -> dict:
    """
    Faithfulness: are all claims in the answer grounded in retrieved context?
    Score 0.0–1.0  (1.0 = fully faithful, no hallucination)
    """
    scorer = Faithfulness(llm=ragas_llm)
    result = await scorer.ascore(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    )
    score = float(result.value) if result.value is not None else float("nan")
    log.info("faithfulness score: %.4f", score)
    return {"key": "faithfulness", "score": score}


async def _ragas_context_recall_async(run, example) -> dict:
    """
    Context Recall: how much of the reference answer does the retrieved context cover?
    Score 0.0–1.0  (1.0 = full recall)
    Requires ground_truths in the LangSmith dataset outputs.
    """
    reference = (example.outputs or {}).get("ground_truths", "")
    if not reference:
        log.warning("No ground_truths in example outputs — skipping context_recall")
        return {"key": "context_recall", "score": float("nan")}

    scorer = ContextRecall(llm=ragas_llm)
    result = await scorer.ascore(
        user_input=run.outputs["question"],
        retrieved_contexts=run.outputs["retrieved_context"],
        reference=reference,
    )
    score = float(result.value) if result.value is not None else float("nan")
    log.info("context_recall score: %.4f", score)
    return {"key": "context_recall", "score": score}


# ---------------------------------------------------------------------------
# Sync wrappers — LangSmith evaluate() calls evaluators synchronously
# ---------------------------------------------------------------------------
def ragas_faithfulness(run, example) -> dict:
    print(run.outputs)
    return asyncio.run(_ragas_faithfulness_async(run, example))


def ragas_context_recall(run, example) -> dict:
    return asyncio.run(_ragas_context_recall_async(run, example))


# ---------------------------------------------------------------------------
# Evaluation entry-point
# ---------------------------------------------------------------------------
def run_evaluation(
    dataset_name: str = "rag-evals",
    experiment_prefix: str = "retriever-eval",
    max_concurrency: int = 1,
) -> None:
    log.info("Starting evaluation on dataset: %r", dataset_name)

    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[ragas_faithfulness, ragas_context_recall],
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        metadata={
            "qdrant_collection": "Amazon-items-collection-01",
            "embedding_model": "embed-v4.0",
            "llm_model": "gpt-oss-120b",
            "eval_llm": "gpt-4o-mini",
        },
    )

    log.info("Evaluation complete. Results URL: %s", results.url)

    # Pretty-print summary to stdout
    print("\n" + "=" * 60)
    print(f"  Experiment : {results.experiment_name}")
    print(f"  Dataset    : {dataset_name}")
    print(f"  Examples   : {len(results)}")
    print("=" * 60)
    for r in results:
        q = r["run"].inputs.get("question", "—")[:60]
        evals = {
            e["key"]: round(e["score"], 4)
            for e in r["evaluation_results"]["results"]
            if e["score"] is not None
        }
        print(f"  Q: {q!r}")
        print(f"     {evals}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_evaluation()



