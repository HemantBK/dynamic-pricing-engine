.PHONY: install train run test lint docker-up docker-down clean data mlflow dashboard

# Install all dependencies
install:
	pip install -r requirements.txt

# Download and ingest data
data:
	python -m src.data.ingestion

# Run MLflow UI
mlflow:
	mlflow ui --backend-store-uri mlruns --port 5000

# Train all models
train:
	python -m src.models.demand_forecaster
	python -m src.models.elasticity_estimator

# Run FastAPI server
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit dashboard
dashboard:
	streamlit run dashboard/app.py --server.port 8501

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Lint with ruff
lint:
	ruff check src/ tests/ dashboard/
	ruff format --check src/ tests/ dashboard/

# Format code
format:
	ruff format src/ tests/ dashboard/

# Docker
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
