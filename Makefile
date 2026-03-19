.PHONY: install sync lint infra down run

install:
	@uv sync --dev

sync:
	@uv sync --dev

lint:
	@uv run ruff check --fix src/ && uv run ruff format src/

infra:
	@docker compose up -d

down:
	@docker compose down

run:
	@uv run python -m e_rag
