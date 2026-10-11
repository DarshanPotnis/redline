# Redline

Redline is an adaptive measurement engine for LLM reasoning. It generates fresh, machine-verifiable puzzles at controllable difficulty, runs several solving strategies (direct answer, code, Z3 translation, tool-using agent) under equal token/dollar budgets, and adaptively picks difficulty to find each model+strategy's breaking point, with error bars, using as few model calls as possible. Stack: Inspect AI (eval harness), Reasoning Gym (existing puzzle generators), z3-solver; later phases export Inspect logs to Parquet and query them with DuckDB. Redline does not claim a new benchmark. Relative to ZebraLogic (ICML 2025) and Reasoning Gym (NeurIPS 2025), it adds adaptive testing, misread-vs-misreasoned diagnosis, budget-matched comparison, and a shared difficulty scale across puzzle types.

## Module map

All code lives in `src/redline/`.

| Package | Responsibility |
|---|---|
| `puzzles` | Seeded generators; each returns puzzle text, a hidden structured instance, and the answer key |
| `strategies` | Inspect solvers; they only ever see puzzle text |
| `scoring` | The verifier, as an Inspect scorer |
| `controller` | The adaptive difficulty loop |
| `stats` | Breaking-point fits and error bars |
| `export` | Inspect logs to Parquet |
| `tasks` | Inspect task definitions (`tasks/smoke.py` is the Phase 0 pipeline check) |

## Commands

| Command | What it runs |
|---|---|
| `make setup` | `uv sync` + `pre-commit install` |
| `make format` | `ruff check --fix` + `ruff format` (auto-fix) |
| `make lint` | `ruff check` + `ruff format --check` |
| `make typecheck` | `mypy` (strict, on `src/`) |
| `make test` | `pytest` |
| `make check` | lint + typecheck + test (same gates as CI) |
| `make smoke` | `inspect eval` of the smoke task on `mockllm/model` |
| `make view` | `inspect view` (log viewer) |

## Models

- **Local, via Ollama** (Inspect model ID `ollama/<tag>`): `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`. The dev machine is an 8 GB M2 MacBook Air: max 4B, one model loaded at a time, thinking mode off for local models.
- **Cloud, via the Groq free tier** (no card on file): `groq/llama-3.3-70b-versatile` and `groq/openai/gpt-oss-20b`, for regular-vs-reasoning comparisons.

## Phase plan

- **Phase 0: setup (CURRENT)**
- **Phase 1: first curve.** One logic-grid generator with a brute-force uniqueness check, one Reasoning Gym task, two strategies (direct; Z3-translate via a JSON constraint format we compile ourselves, so no model-written code runs yet), the local Qwen 3.5 sizes plus one Groq model, a fixed difficulty ladder, Parquet export, one accuracy-vs-difficulty chart. Phase 1 is where pandas, pyarrow, duckdb, and jupyter come in.
- **Phase 2: agentic strategies** (code execution in Inspect's Docker sandbox, `react()` agent), budget caps, caching.
- **Phase 3: adaptive controller** (staircase, then IRT), breaking points with bootstrapped error bars.
- **Phase 4:** more generators, a static dashboard (DuckDB-WASM), a write-up.

## Hard rules

- **Zero-cost project: no paid API calls, ever.** Experiments use local models through Ollama or free API tiers. Never run an eval against a real model without asking first.
- Build only what the current phase asks. Ask before adding a dependency.
- Brute force first, optimize second; keep the brute-force version as a test oracle.
- Answer keys never reach a strategy; strategies receive puzzle text only.
- Generators are seeded and deterministic: same (difficulty, seed) gives the same puzzle.
- Every generator gets Hypothesis property tests: exactly one solution; the key satisfies every clue; raising a difficulty knob never makes puzzles easier.
- Tests never call real models; use `mockllm/model`.
- All strategies in a comparison run under the same budget, and reasoning tokens count toward it.
- Record the exact model ID and config for every experiment run.
- mypy strict, small pure functions, logging instead of print in library code.
- Run `make check` before every commit; use conventional commit messages.
