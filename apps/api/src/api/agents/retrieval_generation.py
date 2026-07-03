import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Prefetch, Document
from qdrant_client import models
from api.agents.utils.prompt_management import prompt_template_config

qdrant_client = QdrantClient(url='http://qdrant:6333')

class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="The description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

@traceable(
    name='embed_query',
    run_type='embedding',
    metadata={
        'ls_provider': 'openai',
        'ls_model_name': 'text-embedding-3-small',
    },
)
def get_embedding(text, model='text-embedding-3-small'):
    response = openai.embeddings.create(
        model=model,
        input=text,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
        }
    
    return response.data[0].embedding

@traceable(
    name='retrieve_data',
    run_type='retriever',
)
def retrieve_data(query, qdrant_client, collection_name='amazon-items-collection-01-hybrid-search', k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
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
        query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context_scores = []
    retrieved_context_texts = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context_scores.append(result.score)
        retrieved_context_texts.append(result.payload['processed_description'])
        retrieved_context_ratings.append(result.payload['average_rating'])

    return {
        'retrieved_context_ids': retrieved_context_ids,
        'retrieved_context_scores': retrieved_context_scores,
        'retrieved_context_texts': retrieved_context_texts,
        'retrieved_context_ratings': retrieved_context_ratings
    }

@traceable(
    name='format_retrieved_context',
    run_type='prompt',
)
def process_context(retrieve_context):
    formatted_context = ''

    for id, chunk, rating in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts'], retrieve_context['retrieved_context_ratings']):
        formatted_context += f"- Product ID: {id}, Product Rating: {rating}, Product Description: {chunk}\n"

    return formatted_context

@traceable(
    name='build_prompt',
    run_type='prompt',
)
def build_prompt(question, formatted_context):
    template = prompt_template_config('api/agents/prompts/retrieval_generation.yml', 'retrieval_generation')
    prompt = template.render(formatted_context=formatted_context, question=question)
    return prompt

@traceable(
    name='generate_answer',
    run_type='llm',
    metadata={
        'ls_provider': 'openai',
        'ls_model_name': 'gpt-5.4-nano',
    },
)
def generate_answer(prompt):
    client = instructor.from_provider(
        "openai/gpt-5.4-nano",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt}
        ],
        reasoning={'effort': 'none'},
        response_model=RAGGenerationResponse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.input_tokens,
            'output_tokens': raw_response.usage.output_tokens,
            'total_tokens': raw_response.usage.total_tokens,
        }

    return response

@traceable(
    name='rag_pipeline',
)
def rag_pipeline(question, qdrant_client, topk=5):
    retrieved_context = retrieve_data(query=question, qdrant_client=qdrant_client, k=topk)
    formatted_context = process_context(retrieved_context)
    prompt = build_prompt(question, formatted_context)
    answer = generate_answer(prompt)
    
    final_answer = {
        'answer': answer.answer,
        'references': answer.references,
        'question': question,
        'retrieved_context_ids': retrieved_context['retrieved_context_ids'],
        'retrieved_context_texts': retrieved_context['retrieved_context_texts'],
    }

    return final_answer

def rag_pipeline_wrapper(question, topk=5):
    qdrant_client = QdrantClient(url='http://qdrant:6333')

    result = rag_pipeline(question, qdrant_client, topk)

    used_context = []

    for reference in result.get('references', []):
        payload = qdrant_client.scroll(
            collection_name='amazon-items-collection-01-hybrid-search',
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key='parent_asin', match=MatchValue(value=reference.id))
                ]
            ),
        )[0][0].payload

        image_url = payload.get('image', '')
        price = payload.get('price', None)
        
        if image_url:
            used_context.append({
                'image_url': image_url,
                'price': price,
                'description': reference.description,
            })
        
    return {
        'answer': result['answer'],
        'used_context': used_context,
    }