from typing import Iterator

from api.agents.graph import stream_agent
from api.agents.llm import get_chat_model
from api.agents.prompts import render_prompt
from api.agents.retrieval import hydrate_used_context, retrieve_data
from api.core.tracing import traced
from api.models import RAGGenerationResponse


@traced(name="processing_context", run_type="prompt")
def process_context(context: dict) -> str:
    """One line per retrieved product: its id, its rating and its chunk."""
    return "".join(
        f"-ID: {product_id}, rating: {rating}, context:{chunk}\n"
        for product_id, chunk, rating in zip(
            context["retrieved_context_ids"],
            context["retrieved_context"],
            context["retrieved_context_ratings"],
        )
    )


@traced(name="building_prompt", run_type="prompt")
def build_prompt(preprocessed_data: str, question: str) -> str:
    return render_prompt(
        "shopping_assistant",
        "rag-generation.yaml",
        preprocessed_data=preprocessed_data,
        question=question,
    )


@traced(name="generating_answer", run_type="llm")
def generate_answer(prompt: str) -> RAGGenerationResponse:
    return get_chat_model().with_structured_output(RAGGenerationResponse).invoke(
        [{"role": "user", "content": prompt}]
    )


@traced(name="rag_pipeline")
def rag_pipeline(query: str, top_k: int = 5) -> dict:
    retrieved_data = retrieve_data(query, top_k)
    response = generate_answer(build_prompt(process_context(retrieved_data), query))
    return {
        "datamodel": response,
        "references": response.references,
        "question": query,
        "answer": response.answer,
        "context_ids": retrieved_data["retrieved_context_ids"],
        "retrieved_context": retrieved_data["retrieved_context"],
        "score": retrieved_data["similarity_scores"],
        "rating": retrieved_data["retrieved_context_ratings"],
    }


@traced(name="rag_pipeline_wrapper")
def rag_pipeline_wrapper(question: str, top_k: int = 5) -> dict:
    """The `/rag/` endpoint's shape: an answer plus the products behind it."""
    result = rag_pipeline(question, top_k=top_k)
    used_context = hydrate_used_context(
        [(item.id, item.description) for item in result.get("references", [])]
    )
    return {"answer": result["answer"], "used_context": used_context}


def rag_agent_stream_wrapper(question: str, thread_id: str | None = None) -> Iterator[dict]:
    yield from stream_agent(question, thread_id)
