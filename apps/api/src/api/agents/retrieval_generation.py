from qdrant_client.http.models import FusionQuery
from qdrant_client.conversions.common_types import Document
from qdrant_client.conversions.common_types import Prefetch
from api.api.models import RAGGenerationResponse
import instructor
from qdrant_client import QdrantClient
import os
import numpy as np
import cohere
from qdrant_client.models import FieldCondition,Filter,MatchValue
from dotenv import load_dotenv
import openai
from langsmith import traceable, get_current_run_tree
load_dotenv()

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY")
# )
client = instructor.from_openai(openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY")
))


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"model":"embed-v4.0","input_type":"classification","output_dimension":1536,"embedding_types":["float"]}
)
def generate_embedding(text):
    response = co.embed(
        model="embed-v4.0",
        inputs=[
            {
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ],
        input_type="classification",
        output_dimension=1536,
        embedding_types=["float"],
    )
    current_run = get_current_run_tree()
    if current_run and hasattr(response, "usage"):
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.embeddings.float[0]

@traceable(
    name="retrieving_data",
    run_type="retriever"
)
def retrieve_data(query,k=5):
    query_embedding = generate_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )
    retrieved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]
    retrieved_context_ratings=[]

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        similarity_scores.append(result.score)
        retrieved_context.append(result.payload['processed_description'])
        retrieved_context_ratings.append(result.payload['average_rating'])
    return {
        "retrieved_context_ids":retrieved_context_ids,
        "retrieved_context":retrieved_context,
        "similarity_scores":similarity_scores,
        "retrieved_context_ratings":retrieved_context_ratings

    }


@traceable(
    name="processing_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context=""
    for id,chunk,rating in zip(context["retrieved_context_ids"],context["retrieved_context"],context["retrieved_context_ratings"]):
        formatted_context += f"-ID: {id}, rating: {rating}, context:{chunk}\n"
    return formatted_context

@traceable(
    name="building_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_data,question):
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.
    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.
    - Never use the word context and refer to it as the available products.

    Context:{preprocessed_data}

    Question:{question}
     """
    return prompt

@traceable(
    name="generating_answer",
    run_type="llm"
)
def generate_answer(prompt):

    response, raw_response = client.chat.completions.create_with_completion(
    model="gpt-oss-120b",
    messages=[{"role":"system", "content":prompt}],
    response_model=RAGGenerationResponse
)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens":raw_response.usage.prompt_tokens,
            "output_tokens":raw_response.usage.completion_tokens,
            "total_tokens":raw_response.usage.total_tokens
        }
    return response


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(query,top_k=5):
    retrieved_data = retrieve_data(query,top_k)
    preprocessed_data = process_context(retrieved_data)
    prompt = build_prompt(preprocessed_data,query)
    response = generate_answer(prompt)
    return {
        "datamodel":response,
        "references":response.references,
        "question":query,
        "answer":response.answer,
        "context_ids":retrieved_data["retrieved_context_ids"],
        "retrieved_context":retrieved_data["retrieved_context"],
        "score":retrieved_data["similarity_scores"],
        "rating":retrieved_data["retrieved_context_ratings"]
    }


def rag_pipeline_wrapper(question,top_k=5):
    result = rag_pipeline(question,top_k=5)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()
    for item in result.get("references",[]):
        payload = qdrant_client.query_points(collection_name="Amazon-items-collection-01-hybrid",with_payload=True,query=dummy_vector,limit=1,query_filter=Filter(must=[FieldCondition(key="parent_asin",match=MatchValue(value=item.id))])).points[0].payload
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append({
                "id": item.id,
                "image_url":image_url,
                "price":price,
                "description":item.description
            })
    return {
        "answer": result["answer"],
        "used_context":used_context
    }

    