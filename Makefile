.PHONY: setup test lint build run down ci-validate

# Setup virtual environment and install dependencies
setup:
	pip install --upgrade pip
	pip install pre-commit
	pre-commit install
	pip install -r requirements.txt
	pip install -e .

# Run the complete test suite with coverage
test:
	pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

# Run linters and formatters manually
lint:
	black src/ app/ tests/
	flake8 src/ app/ tests/ --max-line-length=88 --extend-ignore=E203,E402,E501

# Build the complete production Docker stack
build:
	docker compose build

# Run the production Docker stack detached
run:
	docker compose up -d

# Spin down the Docker stack
down:
	docker compose down

# Force run the exact CI pipeline validation locally
ci-validate: lint test build
	@echo "✅ Local CI validation passed!"
