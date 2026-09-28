"""Retrieval: the search, the prompt, and the answer built from them.

The direct retrieval endpoint's pipeline, kept separate from the agent's tool so
the two can change independently. `catalog` holds the Qdrant and embedding calls
and `prompts` holds the template loading; this module is only the sequence:

    retrieve  ->  format the context  ->  render the prompt  ->  answer
"""

from langsmith import traceable

from api.agents.llm import get_chat_model
from api.agents.retrieval.catalog import retrieve_data
from api.agents.retrieval.prompts import render_prompt
from api.api.models import RAGGenerationResponse


@traceable(name="processing_context", run_type="prompt")
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


@traceable(name="building_prompt", run_type="prompt")
def build_prompt(preprocessed_data: str, question: str) -> str:
    return render_prompt(
        "shopping_assistant",
        "rag-generation.yaml",
        preprocessed_data=preprocessed_data,
        question=question,
    )


@traceable(name="generating_answer", run_type="llm")
def generate_answer(prompt: str) -> RAGGenerationResponse:
    return get_chat_model().with_structured_output(RAGGenerationResponse).invoke(
        [{"role": "user", "content": prompt}]
    )


@traceable(name="rag_pipeline")
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
