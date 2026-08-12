run-docker-compose:
	uv sync
	docker-compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/prerequisites/*.ipynb

run-eval-retriever:
	uv sync
	uv run --env-file .env --with python-dotenv \
		python -c "\
import sys; \
sys.path.insert(0, 'apps/api/src'); \
sys.path.insert(0, 'apps/api'); \
from evals.eval_retriever import run_evaluation; \
run_evaluation()"


uv run --env-file .env --with python-dotenv python -c "import sys; sys.path.insert(0, 'apps/api/src'); sys.path.insert(0, 'apps/api'); from evals.eval_retriever import run_evaluation; run_evaluation()"