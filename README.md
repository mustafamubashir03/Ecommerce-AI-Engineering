# AI Engineering RAG Application

This repository contains a streamlined, production-focused application for Retrieval-Augmented Generation (RAG) and evaluation.

## Architecture & Engineering Focus

The codebase is built on a clean, API-first architecture:
- **Core RAG Pipeline:** Built a robust, scalable RAG pipeline focused entirely on production readiness.
- **RAG Evaluation Pipeline:** Developed and refined a comprehensive RAG evaluation dataset generation script (`eval_retriever.py`) to systematically assess retriever performance.
- **Prompt Management:** Prompts are managed cleanly via YAML files (e.g., `rag-generation.yaml`) for optimal maintainability and version control.
- **API Architecture:** Implemented custom API middleware and refined FastAPI endpoints for superior request handling and performance.
- **Dependency Management:** Maintained strict lockfiles and fast dependency resolution using `uv`.

## Technologies & Tools Used
- **Backend:** Python, FastAPI
- **AI & RAG:** LangChain / LLM APIs, Qdrant (Vector Database)
- **Tooling & Infrastructure:** `uv` (Dependency Management), Docker & Docker Compose
- **Evaluation:** Custom evaluation scripts using synthetic dataset generation

## Set up
- Clone the repo
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync` to install the dependencies and create the virtual environment under the `.venv` folder
- Create a `.env` file in the root folder with your own API keys and settings based on the `.env.example` file 
- **Run the Application:** Run `make run-docker-compose` from the root folder to start the Streamlit app, FastAPI, and Qdrant containers.
- **Run Evaluations:** Run `make run-eval-retriever` from the root folder to execute the retriever evaluation pipeline.

## Future Roadmap (Aiming for Further)
- **Continuous Evaluation:** Integrate automated RAG evaluation into CI/CD pipelines.
- **Advanced Retrieval Techniques:** Implement and benchmark hybrid search, query rewriting, and reranking natively within the API.
- **Observability:** Integrate comprehensive tracing and logging for LLM calls and API requests.
- **Production Readiness:** Prepare the Dockerized deployment for scalable cloud infrastructure.

## Contact
- Maintained by Mustafa