# Phase 1g run plan

Written 2026-11-05 at commit `230b1a3`. Phase 1g produces Redline's first reported results:
- ladder curves for two puzzle families (logic grid, knights and knaves);
- the misread-vs-misreasoned diagnosis on logic grids.

Everything here runs on TEST seeds (from 1,000,000), which no prompt or strategy was tuned on. The strategies are frozen as of 1f.1.

| Step | What | Status |
|---|---|---|
| A | Groq calibration on dev seeds 20-22 | **Blocked: no `GROQ_API_KEY` in `.env`** |
| B | Ladder extension, if A finds a ceiling | Pending A |
| C | Test runs: local and Groq | Planned below |
| 1g.3 | Analysis and findings write-up | Planned below |

## A. Groq calibration

Free tier only; dev seeds 20-22, which no earlier run used.

1. **Reasoning tokens.** One `groq/openai/gpt-oss-20b` call with a small `max_tokens` (e.g. 64) answers two questions, the last open item from 1c.1:
   - Does `usage` report `reasoning_tokens`?
   - Do they count against `max_tokens`? If so, the call stops on `max_tokens` with little or no visible text.

   Redline's budget rule counts reasoning tokens toward the budget either way. This only confirms that the accounting is right.
2. **Limits.** Read the `x-ratelimit-*` headers on the first response of each model; they're returned on every call. Record the requests and tokens per minute and per day.
3. **Ceiling check.** Levels 5-7, 3 seeds, early stop off:

   ```bash
   caffeinate -is uv run python -m redline.runner --generator logic_grid \
     --models groq/llama-3.3-70b-versatile groq/openai/gpt-oss-20b \
     --strategies direct cot z3_translate --levels 5,6,7 --n-seeds 3 --seed-start 20 \
     --early-stop 3 --run-id cal-1g-groq-grid
   caffeinate -is uv run python -m redline.runner --generator knights_knaves \
     --models groq/llama-3.3-70b-versatile groq/openai/gpt-oss-20b \
     --strategies direct cot --levels 5,6,7 --n-seeds 3 --seed-start 20 \
     --early-stop 3 --run-id cal-1g-groq-kk
   ```

   That's 90 samples and about 0.3M tokens, a day or two of free-tier quota. Inspect retries rate-limited (429) requests with exponential backoff and honors `retry-after`, and the Groq profiles keep 2 concurrent requests. If a daily quota runs out mid-run, rerun the same command the next day: the runner resumes.

**Decision rule:** if any Groq model solves at least 2/3 at level 7 of a generator, that generator's ladder gets more levels (B) before the test run.

### Groq limits as published

From [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits), read 2026-11-05. The page has no date, and it labels its table the "base limits for the Developer plan".

| Model | Requests/min | Requests/day | Tokens/min | Tokens/day |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | 30 | 1,000 | 8,000 | 200,000 |
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | not found |

The llama row comes from third-party summaries; that model isn't on Groq's page. Step A2 replaces both rows with the headers' values.

## B. Ladder extension (only if A finds a ceiling)

Planned, not built.

- **Knights and knaves, levels 8-10.** Proposed (n_people, depth, width): (8, 4, 3), (9, 4, 4), (10, 5, 4). Before adopting them:
  - measure generation time; levels 1-7 take at most about 20 ms;
  - check that the gold "Answer:" line stays under about 100 tokens; it's 47 at 7 people.
- **Logic grid, levels 8 (7x7) and 9 (8x8).**
  - The vocabulary grows to 8 attributes with 8 values each. Levels 1-7 keep sampling from the original 6x6 pool, so no existing puzzle changes; the structure golden digests in `tests/puzzles/test_generate.py` stay byte-identical and prove it.
  - A prototype measures Z3 generation time first; level 7 already takes about 3 s per puzzle.
  - **Open issue:** direct's answer-only budget is 512 tokens. A 6x6 grid is already 371 tokens fully indented, so 7x7 and 8x8 grids would overrun it. Extending the grid ladder needs a decision first: a larger answer-only budget for big grids, or keep the grid ladder at 7 and extend only knights and knaves.

## C. Test runs

All runs: TEST seeds from 1,000,000, 10 seeds per level, early stop 2, one generator per run, at the current commit with a clean tree. Run each with `caffeinate -is`, plugged in, lid open, VS Code closed (frees memory for the 4B).

### Local (Ollama: Qwen 3.5 2B and 4B)

```bash
caffeinate -is uv run python -m redline.runner --generator logic_grid \
  --models ollama/redline-qwen3.5-2b ollama/redline-qwen3.5-4b \
  --strategies direct cot z3_translate --levels 1-7 --n-seeds 10 --seed-start 1000000 \
  --early-stop 2 --run-id test-1g-grid-local

caffeinate -is uv run python -m redline.runner --generator knights_knaves \
  --models ollama/redline-qwen3.5-2b ollama/redline-qwen3.5-4b \
  --strategies direct cot --levels 1-7 --n-seeds 10 --seed-start 1000000 \
  --early-stop 2 --run-id test-1g-kk-local
```

| Run | Estimate | Where the time goes |
|---|---|---|
| `test-1g-grid-local` | 7.5-10.5 h | 4B cot about 4-5 h, 2B cot about 2-3 h, 4B z3_translate about 1.2 h, direct under 0.5 h |
| `test-1g-kk-local` | 6.5-8 h | 2B cot about 2.5-3.5 h, 4B cot about 3.5 h, direct about 0.5 h |

The estimates use per-sample times from pilot-1e, the 1e.2 dev check, pilot-1f and dev-1f1, with VS Code closed where measured. The range is how far the climbs go. With 10 seeds a level is all-zero, and stops a climb, less often than with 3, so expect the upper end.

**The logic-grid run may not fit one night.** If it doesn't, split it by model into `test-1g-grid-local-2b` and `test-1g-grid-local-4b`: the same command with one `--models` value each, at the same commit.

### Groq (free tier)

```bash
caffeinate -is uv run python -m redline.runner --generator logic_grid \
  --models groq/llama-3.3-70b-versatile groq/openai/gpt-oss-20b \
  --strategies direct cot z3_translate --levels 1-7 --n-seeds 10 --seed-start 1000000 \
  --early-stop 2 --run-id test-1g-grid-groq

caffeinate -is uv run python -m redline.runner --generator knights_knaves \
  --models groq/llama-3.3-70b-versatile groq/openai/gpt-oss-20b \
  --strategies direct cot --levels 1-7 --n-seeds 10 --seed-start 1000000 \
  --early-stop 2 --run-id test-1g-kk-groq
```

If B adds levels, use `--levels 1-9` (grid) or `1-10` (knights and knaves).

**Sizing.** The tokens-per-day quota is the binding limit. Tokens per sample, input included:

| Strategy | Logic grid | Knights and knaves |
|---|---|---|
| direct | about 1-2K | about 0.5-1K |
| cot | about 4-6K (about 10K when forced) | about 2-3.5K |
| z3_translate | about 3-4K | n/a |

That's about 1.0M tokens per model for the logic-grid run and 0.3M for knights and knaves, before early stop. At gpt-oss-20b's 200K tokens/day, both runs take **about 6-7 days**. Llama's daily cap is unknown and may stretch that. Each model has its own quota, so both run on the same days. When a quota runs out the run stops with an error; rerun the same command after the daily reset and it resumes, reusing every finished sample.

**To confirm in A:** the per-day quotas from the headers, and gpt-oss's reasoning-token accounting. Its reasoning counts toward the 4,096-token budget, so cot samples may be larger than estimated.

## 1g.3 Analysis

Everything is computed from the Parquet exports (`make report RUN=<id>` per run, then joins in DuckDB). Context-limited samples are excluded from every rate and counted separately.

1. **Ladder charts per generator.**
   - Solved rate by level, one panel per model, one line per strategy, with 95% Wilson intervals.
   - The secondary score (cell accuracy for grids, Reasoning Gym's partial credit for knights and knaves) against its chance line.
   - Local and Groq models side by side.
2. **Misread vs misreasoned (logic grid).** Per (model, sample_id), z3_translate's `translation_exact` crossed with direct's and cot's `solved` on the same puzzle:

   | | solved | not solved |
   |---|---|---|
   | **exact translation** | understood and reasoned | **misreasoned**: read the clues right, couldn't reason through them |
   | **inexact translation** | translation was the bottleneck | misread (or both) |

   Reported by level and model, with the translation-error mix: n_wrong, n_missed, n_inferred, n_malformed and n_entailed.
3. **Strategy fidelity and budget pressure.** direct_overrun (direct's one 512-token call cut off) and forced_answer (cot's reasoning hit its cap) by level and model.
4. **Harness health.**
   - context-limited counts, format_ok and truncation;
   - strategy-fidelity flags per (model, generator);
   - errored cells and session counts from each manifest.

**Caveats to state in the write-up:**
- The 4B overrides "no reasoning" on knights and knaves 11% of the time; dev-1f1 flagged it.
- Knights-and-knaves partial credit is high by chance: random roles score 0.575-0.648.
- pilot-1e and pilot-1f are dev-seed pilots and are not reported results. pilot-1f's direct numbers predate the answer-only budget and measure cot, not direct.
