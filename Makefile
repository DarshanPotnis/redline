.PHONY: setup format lint typecheck test check smoke view puzzle

LEVEL ?= 3
SEED ?= 0

setup:
	uv sync
	uv run pre-commit install

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

smoke:
	uv run inspect eval src/redline/tasks/smoke.py --model mockllm/model

view:
	uv run inspect view

puzzle:
	uv run python -m redline.puzzles.logic_grid --level $(LEVEL) --seed $(SEED)
