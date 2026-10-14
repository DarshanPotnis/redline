# Redline

Redline is an adaptive measurement engine for LLM reasoning. It generates fresh, machine-verifiable puzzles at controllable difficulty, runs several solving strategies (direct answer, code, Z3 translation, tool-using agent) under equal token/dollar budgets, and adaptively picks difficulty to find each model+strategy's breaking point, with error bars, using as few model calls as possible. Stack: Inspect AI (eval harness), Reasoning Gym (existing puzzle generators), z3-solver; later phases export Inspect logs to Parquet and query them with DuckDB. Redline does not claim a new benchmark. Relative to ZebraLogic (ICML 2025) and Reasoning Gym (NeurIPS 2025), it adds adaptive testing, misread-vs-misreasoned diagnosis, budget-matched comparison, and a shared difficulty scale across puzzle types.

## Module map

All code lives in `src/redline/`.

| Package | Responsibility |
|---|---|
| `puzzles` | Seeded generators; each returns puzzle text, a hidden structured instance, and the answer key. `logic_grid` is ours; `reasoning_gym` adapts any Reasoning Gym task with a level ladder (knights and knaves so far); `registry.py` maps `--generator` names to generators |
| `strategies` | Async `run(PublicPuzzle, BudgetedModel) -> StrategyResult` (contract v2, `base.py`); they never see `TaskState` or the answer key (AST- and replay-tested) |
| `scoring` | Inspect scorers: `grid_verifier` parses the model's grid, grades it and names the broken clues; `rg_verifier` extracts the last "Answer:" line and scores it with Reasoning Gym's checker; `translation_verifier` diagnoses misreads for translating strategies; `harness.py` holds the truncation, context-limit and forcing flags every answer scorer reports |
| `controller` | The adaptive difficulty loop |
| `stats` | Breaking-point fits and error bars (`wilson_interval` so far) |
| `export` | `to_parquet(run_dir)`: a run's successful logs to `data/runs/<run_id>/samples.parquet`, one row per sample |
| `runner` | The ladder runner: one `eval_set` per (model, strategy, level) cell, high-jump early stop, resumable, `manifest.json` per run |
| `analysis` | Per-level solved rates with Wilson intervals (DuckDB over the Parquet), each generator's secondary score with its chance baseline (cell accuracy, or RG's partial credit), the ladder chart and `reports/<run_id>.md` |
| `tasks` | Inspect task definitions (`logic_grid.py`, `reasoning_gym.py`; `smoke.py` is the Phase 0 pipeline check) and `solver.py`, the only code that touches `TaskState` |
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
| `make pilot` | The 1e pilot: 3 local models x 3 strategies, levels 1-7, 3 dev seeds, early stop 2, run id `pilot-1e` (resumable; under `caffeinate -is` on macOS) |
| `make report RUN=<id>` | Parquet export, chart (`reports/figures/<id>_ladder.png`) and report (`reports/<id>.md`) for one run |
| `make ollama-models` | Pull missing base models and (re)create `redline-*` from `models/ollama/*.Modelfile` |

## Models

- **Local, via Ollama:** `ollama/redline-qwen3.5-0.8b`, `ollama/redline-qwen3.5-2b`, `ollama/redline-qwen3.5-4b`, built by `make ollama-models` from `models/ollama/*.Modelfile`: an 8192-token context and every sampling parameter pinned from the model card's non-thinking settings. Thinking is off per request (`reasoning_effort: "none"` in `extra_body`). The dev machine is an 8 GB M2 MacBook Air: max 4B, one model loaded at a time.
- **Cloud, via the Groq free tier** (no card on file): `groq/llama-3.3-70b-versatile` and `groq/openai/gpt-oss-20b` (reasoning on; its reasoning tokens count toward the budget), for regular-vs-reasoning comparisons.
- Every model is registered in `runtime/profiles.py`. Ollama's OpenAI-compatible API resets `temperature` and `top_p` to 1.0 when a request omits them, so local profiles read both from the Modelfile and send them; the Modelfile stays the single source of truth.

## Phase plan

- **Phase 0: setup** (done)
- **Phase 1: first curve (CURRENT).** One logic-grid generator with a brute-force uniqueness check, one Reasoning Gym task, two strategies (direct; Z3-translate via a JSON constraint format we compile ourselves, so no model-written code runs yet), the local Qwen 3.5 sizes plus one Groq model, a fixed difficulty ladder, Parquet export, one accuracy-vs-difficulty chart. Phase 1 is where pandas, pyarrow, duckdb, and jupyter come in (1e added pandas, pyarrow, duckdb and matplotlib; pandas only at the `samples_df` boundary).
  - **1a:** logic-grid generator (done).
  - **1b:** Z3 solver + per-puzzle difficulty features; the ladder becomes levels 1-7 (done).
  - **1c:** Inspect task + verifier + direct and cot strategies (done).
  - **1c.1:** pinned model profiles + truncation-aware scoring (done).
  - **1d:** budgeted strategies with budget forcing, then Z3-translate strategy + misread diagnosis (done).
  - **1e:** resumable ladder runner, Parquet export, ladder chart and report, local pilot run on dev seeds (done; `reports/pilot-1e.md`).
  - **1e.1:** fixes from the pilot diagnosis: generator text v2, item-level translation parsing with a finer misread diagnosis; checked on fresh dev seeds 3-9 (done). Its prompt field lines were reverted in 1e.2: an A/B showed they made 63% of the 2B's items lose `kind`.
  - **1e.2:** translation format is uniform {kind, a, b}; one worked example; format iteration closed after 1e.2. Remaining translation failures are reported as findings, not tuned further. Checked on fresh dev seeds 10-16 (done).
  - **1f (CURRENT):** Reasoning Gym adapter (knights and knaves) through the same profiles, budget, strategies, runner, export and report; generator text v3 (labels equal attribute keys); dev pilot `pilot-1f`.
  - **1f:** Reasoning Gym adapter.
  - **1g:** full local and Groq runs on test seeds, findings write-up (including the misread-vs-misreasoned verdict).
  - Generator text v3 (1f; `GENERATOR_VERSION` in `model.py`, recorded in every instance, sample and Parquet row): color reads "the person whose favorite color is red" (v2, 1e.1), because v1's "the person in the red house" was read as a house position; every label equals its attribute key ("Color") since v3, because v2's label "Favorite color" made models write "favorite_color". Puzzles are byte-identical across versions (the structure digests in `test_generate.py` are pinned; only the text digests moved). v1 is the text pilot-1e ran on. No template, label or value may use a position word ("house", "left", "right", "next", a number); a vocab test enforces it.
  - Reasoning Gym (1f): `generate(level, seed)` is item 0 of `create_dataset(task, size=1, seed=seed, **ladder[level])` (deterministic; item 0 doesn't depend on size). Knights-and-knaves ladder: (n_people, depth, width) = (2,2,2), (3,2,2), (4,2,2), (4,3,3), (5,3,3), (6,3,3), (7,4,3). RG's metadata holds the solution, so it stays in the hidden instance. RG's checker is lenient (it credits "Let me think... <answer>"), so `rg_verifier` scores only the text after the last "Answer:" line; none means format_ok 0. Its partial credit (0.3 + 0.7 x share right) is high by chance: random roles score 0.575 with 2 people, reported as the chance line.
  - Generated logic-grid puzzles are fully relational by design: `at` clues are excluded from the generator's candidate pool, but `at` stays in the clue schema for the 1d translations.
  - The generator counts solutions with backtracking up to a search space of 1,000,000 (levels 1-4) and with Z3 above it. Every counter is exact and the generator only branches on "count == 1", so the counter never changes a puzzle.
  - Difficulty features come from `difficulty.measure()`: K=5 Z3 uniqueness proofs, each in a fresh `z3.Context` (a shared context leaks state between runs and makes the statistics irreproducible). Set Z3 options with `solver.set(...)`, never `z3.set_param`.
- **Phase 2: agentic strategies** (code execution in Inspect's Docker sandbox, `react()` agent), budget caps, caching.
  - z3_translate_constrained: the same strategy with schema-constrained decoding (feasibility confirmed in 1e.2: inline anyOf + enum, with the schema passed to both BudgetedModel calls), to split translation failures into format vs meaning.
- **Phase 3: adaptive controller** (staircase, then IRT), breaking points with bootstrapped error bars.
- **Phase 4:** more generators, a static dashboard (DuckDB-WASM), a write-up.

## Strategy contract (v2)

- A strategy is `async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult`: `final_text` is what the verifier grades, `artifacts` (JSON) go to the scorer.
- `BudgetedModel` (`strategies/base.py`) is the only way a strategy reaches the model. It holds every sample to `TOTAL_BUDGET` output tokens (`redline/budget.py`), sets each call's `max_tokens`, records every call, and refuses calls once the budget is spent.
- Budget forcing (s1): a reasoning call may use at most what remains minus `ANSWER_RESERVE`. If it stops on `max_tokens`, exactly one forcing call follows with the rest, appending the strategy's forcing message.
- direct and cot word their prompts by the puzzle's family (`PublicPuzzle.generator` before "/"; `strategies/common.py` `PROMPTS`): the logic grid answers in JSON (its prompts unchanged since 1c), Reasoning Gym tasks in a final "Answer:" line. z3_translate runs on logic grids only.
- `tasks/solver.py` builds the `PublicPuzzle` and the `BudgetedModel`, runs the strategy, and stores the final text as the output and `{artifacts, calls, budget}` in `state.metadata["strategy_run"]` for the scorer.
- Strategies: `direct`, `cot`, and `z3_translate`. z3_translate makes the model a translator only: it writes each clue as {"kind", "a", "b"} (`strategies/clue_format.py`; `a` is always a value, `b` a value or {"house": N} for at/not_at, converted to the internal Clue model), taught by one worked example from a made-up two-house puzzle whose names appear in no real puzzle (1e.2), Z3 solves them locally, and only a unique solution becomes an answer (unsat or ambiguous answers nothing; one attempt). `PublicPuzzle.structure` (the spec: houses, attributes, values, all already in the text) gives it the exact spellings. The translation is parsed item by item (1e.1): a malformed item (not a clue, or an unknown value or house) is dropped and recorded in `artifacts["malformed"]`, and the rest go to Z3, so the status is what Z3 sees. It is `parse_error` only when nothing reaches Z3 (no JSON list, or no well-formed item).
- Misread diagnosis (`translation_verifier`, z3_translate runs only; 1e.1). Translated clues: `n_exact` (canonically equal to a stated clue; refs of symmetric kinds sorted), `n_inferred` (true on the solution but not stated: a ref swapped for an equivalent one, or a weakened relation), `n_wrong` (false on the solution: a misread); plus `n_malformed` items dropped. Stated clues: `n_captured` (an exact translation exists), `n_entailed` (the translated set implies it: Z3 finds translated AND NOT clue unsat, and only when the translated set is satisfiable), `n_missed` (neither). `translation_exact` = no wrong, no missed, no malformed. The full misread-vs-misreasoned verdict joins strategies by `sample_id` in the Parquet export (1g): an exact translation plus a wrong direct answer means the model understood the clues but could not reason through them.

## Ladder runner (1e)

- `uv run python -m redline.runner [--generator logic_grid|knights_knaves] --models ... --strategies ... --levels 1-7 --n-seeds N --seed-start S --early-stop 2 --run-id <name>`. The generator defaults to logic_grid, and old manifests (without one) load as logic_grid. Each (model, strategy) climbs the levels in order and stops after `early-stop` consecutive levels with solved = 0 (the high-jump rule); the stop reason (`completed`, `early_stop`, `error`) goes in the manifest.
- Each cell is one Inspect `eval_set` with its own log dir, `logs/runs/<run_id>/<model>/<strategy>/L<level>/`. A single eval_set can't hold the run (the next level depends on the last result), but per cell it gives resume: complete logs are reused without model calls, an interrupted log finishes with its completed samples reused. Rerun the same command to resume. Ctrl-C stops the whole run (exit 130), not just the cell; samples finished before a crash or Ctrl-C are recovered from Inspect's local sample buffer, so only the sample in flight reruns.
- Local models run first, one at a time: every loaded Ollama model is unloaded before a local model starts and after it finishes. API models run after.
- `manifest.json` records the config, seed split, git SHA and dirty flag, every profile, Inspect/z3/Python versions, sessions with start and end times, and per ladder the levels run and the stop reason. A resume must match the config, commit and profiles. A dirty tree is refused unless `--allow-dirty` (recorded).

## Hard rules

- **Zero-cost project: no paid API calls, ever.** Experiments use local models through Ollama or free API tiers. Never run an eval against a real model without asking first.
- Build only what the current phase asks. Ask before adding a dependency.
- Brute force first, optimize second; keep the brute-force version as a test oracle.
- Answer keys never reach a strategy: strategies receive only a `PublicPuzzle` (puzzle text + answer format), and an AST test enforces it.
- Seeds 0-999 are DEV: prompt and strategy changes are only ever tuned on them. Reported results use TEST seeds starting at 1,000,000, which are never looked at while tuning. (`redline/seeds.py`; the runner refuses ranges outside one split, and reports label the split.)
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
