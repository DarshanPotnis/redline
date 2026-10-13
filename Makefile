.PHONY: setup format lint typecheck test test-all check smoke eval-mock view puzzle difficulty-report

LEVEL ?= 3
SEED ?= 0

setup:
	uv sync
	uv run pre-commit install

# --exit-zero: an unfixable lint error is still reported, but must not skip the formatter.
format:
	uv run ruff check --fix --exit-zero .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

check: lint typecheck test

smoke:
	uv run inspect eval src/redline/tasks/smoke.py --model mockllm/model

# mockllm's default reply is not JSON, so every sample scores format_ok=0: a pipeline check.
eval-mock:
	uv run inspect eval src/redline/tasks/logic_grid.py --model mockllm/model -T strategy=direct
	uv run inspect eval src/redline/tasks/logic_grid.py --model mockllm/model -T strategy=cot

view:
	uv run inspect view

puzzle:
	uv run python -m redline.puzzles.logic_grid --level $(LEVEL) --seed $(SEED)

difficulty-report:
	uv run python -m redline.puzzles.logic_grid.difficulty_report
