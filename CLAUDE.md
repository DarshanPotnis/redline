# Redline

Redline is an adaptive measurement engine for LLM reasoning. It generates fresh, machine-verifiable puzzles at controllable difficulty, runs several solving strategies (direct answer, code, Z3 translation, tool-using agent) under equal token/dollar budgets, and adaptively picks difficulty to find each model+strategy's breaking point, with error bars, using as few model calls as possible. Stack: Inspect AI (eval harness), Reasoning Gym (existing puzzle generators), z3-solver; later phases export Inspect logs to Parquet and query them with DuckDB. Redline does not claim a new benchmark. Relative to ZebraLogic (ICML 2025) and Reasoning Gym (NeurIPS 2025), it adds adaptive testing, misread-vs-misreasoned diagnosis, budget-matched comparison, and a shared difficulty scale across puzzle types.

## Module map

All code lives in `src/redline/`.

| Package | Responsibility |
|---|---|
| `puzzles` | Seeded generators; each returns puzzle text, a hidden structured instance, and the answer key |
| `strategies` | Async `run(PublicPuzzle, BudgetedModel) -> StrategyResult` (contract v2, `base.py`); they never see `TaskState` or the answer key (AST- and replay-tested) |
| `scoring` | The verifier, as an Inspect scorer: parses the model's grid, grades it, names the broken clues |
| `controller` | The adaptive difficulty loop |
| `stats` | Breaking-point fits and error bars |
| `export` | Inspect logs to Parquet |
| `tasks` | Inspect task definitions (`logic_grid.py`; `smoke.py` is the Phase 0 pipeline check) and `solver.py`, the only code that touches `TaskState` |
| `runtime` | Model profiles (`profiles.py`): per-model budget, context window, thinking and pinned sampling, written into every eval log |

## Commands

| Command | What it runs |
|---|---|
| `make setup` | `uv sync` + `pre-commit install` |
| `make format` | `ruff check --fix` + `ruff format` (auto-fix) |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | `mypy` (strict, on `src/`) |
| `make test` | `pytest -m "not slow"` (skips the level 5-7 tests) |
| `make test-all` | `pytest`, slow tests included (what CI runs) |
| `make check` | lint + typecheck + test (CI runs the same gates, plus the slow tests) |
| `make smoke` | `inspect eval` of the smoke task on `mockllm/model` |
| `make eval-mock` | `inspect eval` of the logic-grid task on `mockllm/model`, once per strategy |
| `make view` | `inspect view` (log viewer) |
| `make puzzle LEVEL=3 SEED=7` | Print a logic-grid puzzle and its answer grid |
| `make difficulty-report` | Regenerate `reports/difficulty_ladder.md` (about a minute) |
| `make ollama-models` | Pull missing base models and (re)create `redline-*` from `models/ollama/*.Modelfile` |

## Models

- **Local, via Ollama:** `ollama/redline-qwen3.5-0.8b`, `ollama/redline-qwen3.5-2b`, `ollama/redline-qwen3.5-4b`, built by `make ollama-models` from `models/ollama/*.Modelfile`: an 8192-token context and every sampling parameter pinned from the model card's non-thinking settings. Thinking is off per request (`reasoning_effort: "none"` in `extra_body`). The dev machine is an 8 GB M2 MacBook Air: max 4B, one model loaded at a time.
- **Cloud, via the Groq free tier** (no card on file): `groq/llama-3.3-70b-versatile` and `groq/openai/gpt-oss-20b` (reasoning on; its reasoning tokens count toward the budget), for regular-vs-reasoning comparisons.
- Every model is registered in `runtime/profiles.py`. Ollama's OpenAI-compatible API resets `temperature` and `top_p` to 1.0 when a request omits them, so local profiles read both from the Modelfile and send them; the Modelfile stays the single source of truth.

## Phase plan

- **Phase 0: setup** (done)
- **Phase 1: first curve (CURRENT).** One logic-grid generator with a brute-force uniqueness check, one Reasoning Gym task, two strategies (direct; Z3-translate via a JSON constraint format we compile ourselves, so no model-written code runs yet), the local Qwen 3.5 sizes plus one Groq model, a fixed difficulty ladder, Parquet export, one accuracy-vs-difficulty chart. Phase 1 is where pandas, pyarrow, duckdb, and jupyter come in.
  - **1a:** logic-grid generator (done).
  - **1b:** Z3 solver + per-puzzle difficulty features; the ladder becomes levels 1-7 (done).
  - **1c:** Inspect task + verifier + direct and cot strategies (done).
  - **1c.1:** pinned model profiles + truncation-aware scoring (done).
  - **1d (CURRENT):** budgeted strategies with budget forcing, then Z3-translate strategy + misread diagnosis.
  - **1e:** Reasoning Gym adapter, ladder runner, Parquet export, first chart, first real run.
  - Generated logic-grid puzzles are fully relational by design: `at` clues are excluded from the generator's candidate pool, but `at` stays in the clue schema for the 1d translations.
  - The generator counts solutions with backtracking up to a search space of 1,000,000 (levels 1-4) and with Z3 above it. Every counter is exact and the generator only branches on "count == 1", so the counter never changes a puzzle.
  - Difficulty features come from `difficulty.measure()`: K=5 Z3 uniqueness proofs, each in a fresh `z3.Context` (a shared context leaks state between runs and makes the statistics irreproducible). Set Z3 options with `solver.set(...)`, never `z3.set_param`.
- **Phase 2: agentic strategies** (code execution in Inspect's Docker sandbox, `react()` agent), budget caps, caching.
- **Phase 3: adaptive controller** (staircase, then IRT), breaking points with bootstrapped error bars.
- **Phase 4:** more generators, a static dashboard (DuckDB-WASM), a write-up.

## Strategy contract (v2)

- A strategy is `async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult`: `final_text` is what the verifier grades, `artifacts` (JSON) go to the scorer.
- `BudgetedModel` (`strategies/base.py`) is the only way a strategy reaches the model. It holds every sample to `TOTAL_BUDGET` output tokens (`redline/budget.py`), sets each call's `max_tokens`, records every call, and refuses calls once the budget is spent.
- Budget forcing (s1): a reasoning call may use at most what remains minus `ANSWER_RESERVE`. If it stops on `max_tokens`, exactly one forcing call follows with the rest, appending the strategy's forcing message.
- `tasks/solver.py` builds the `PublicPuzzle` and the `BudgetedModel`, runs the strategy, and stores the final text as the output and `{artifacts, calls, budget}` in `state.metadata["strategy_run"]` for the scorer.

## Hard rules

- **Zero-cost project: no paid API calls, ever.** Experiments use local models through Ollama or free API tiers. Never run an eval against a real model without asking first.
- Build only what the current phase asks. Ask before adding a dependency.
- Brute force first, optimize second; keep the brute-force version as a test oracle.
- Answer keys never reach a strategy: strategies receive only a `PublicPuzzle` (puzzle text + answer format), and an AST test enforces it.
- Generators are seeded and deterministic: same (difficulty, seed) gives the same puzzle. Use a local `random.Random(seed)`, and never iterate an unsorted set of strings (string hashing varies per process).
- Every generator gets Hypothesis property tests: exactly one solution; the key satisfies every clue.
- Each level strictly increases the search space. Per-puzzle solver effort is recorded (from 1b) and must rise on average across levels; individual puzzles may vary.
- Tests never call real models; use `mockllm/model`.
- All strategies in a comparison run under the same budget, and reasoning tokens count toward it.
- All strategies share one BudgetedModel rule: TOTAL_BUDGET output tokens per sample, with one forcing call from a 512-token reserve.
- Record the exact model ID and config for every experiment run.
- Every real-model run uses a registered profile. Samples with context_limited=1 are harness failures: they are excluded from accuracy and the count is reported.
- `BudgetedModel` sets every call's `max_tokens`; the profile owns sampling and connections. Never override sampling on the command line.
- mypy strict, small pure functions, logging instead of print in library code.
- Run `make check` before every commit; use conventional commit messages.
