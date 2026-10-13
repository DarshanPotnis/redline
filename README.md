# Redline

Redline finds where LLM reasoning breaks. It generates fresh, machine-verifiable puzzles at a controllable difficulty and runs several solving strategies on them: direct answer, code, Z3 translation and a tool-using agent, all under equal token budgets. It then adaptively picks the next difficulty to locate each model+strategy's breaking point, with error bars, in as few model calls as possible. Every puzzle is new and checked by a verifier, so a score cannot come from a memorized test item.

**Status:** Phase 1 in progress (1e: ladder runner, Parquet export, ladder report). Puzzles run as Inspect evaluations under logged, pinned model profiles, with three budget-matched strategies and a per-clue diagnosis. A resumable runner climbs the difficulty ladder per model and strategy; the first pilot runs on dev seeds, so there are no reported results yet.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and `make`.

```bash
make setup             # uv sync + pre-commit install
cp .env.example .env   # fill in locally; .env is gitignored
make check             # lint, typecheck, tests (the same gates as CI)
make smoke             # run the smoke task on Inspect's mock model
make view              # browse eval logs in Inspect's log viewer
make report RUN=<id>   # Parquet export, ladder chart and report for a runner run
```

The smoke task runs on `mockllm/model`, which always returns the same fixed reply. Its accuracy is therefore 0 by design: the task checks that the pipeline runs, not how good a model is.

## Related work

- **ZebraLogic.** Lin et al., *ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning*, ICML 2025 ([arXiv:2502.01100](https://arxiv.org/abs/2502.01100)). It uses logic-grid puzzles of controllable complexity to show that accuracy collapses as the search space grows.
- **Reasoning Gym.** Stojanovski et al., *Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards*, NeurIPS 2025 ([arXiv:2505.24760](https://arxiv.org/abs/2505.24760)). It is a library of procedural puzzle generators with verifiers and adjustable difficulty. Redline uses it as a puzzle source.

Redline is not a new benchmark. It builds on generators like these and adds four things:

- **Adaptive testing.** Each new difficulty is chosen from the results so far, first with a staircase and later with IRT. Finding a breaking point with error bars then takes far fewer model calls than a fixed sweep.
- **Misread vs. misreasoned.** For each wrong answer, Redline tells apart misreading the puzzle and reasoning badly about a correct reading.
- **Budget-matched comparison.** Every strategy runs under the same token budget, and reasoning tokens count toward it.
- **A shared difficulty scale.** Breaking points are comparable across puzzle types.

## License

MIT. See [LICENSE](LICENSE).
