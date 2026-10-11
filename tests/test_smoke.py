"""Phase 0 toolchain checks. None of these call a real model or a paid API."""

from pathlib import Path

import inspect_ai
import pytest
import reasoning_gym
import z3

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_task_runs_on_mockllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Inspect resolves task files relative to the cwd, so pin it to the repo root.
    monkeypatch.chdir(REPO_ROOT)
    # mockllm returns a fixed reply, so accuracy is 0 by design; this checks the pipeline.
    logs = inspect_ai.eval(
        "src/redline/tasks/smoke.py", model="mockllm/model", log_dir=str(tmp_path), display="none"
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.samples is not None
    assert len(log.samples) == 2


def test_reasoning_gym_scores_its_own_answer() -> None:
    ds = reasoning_gym.create_dataset("zebra_puzzles", size=1, seed=42)
    e = ds[0]
    assert ds.score_answer(answer=e["answer"], entry=e) == 1.0


def test_z3_solves_trivial_constraint() -> None:
    x = z3.Int("x")
    solver = z3.Solver()
    solver.add(x > 2, x < 4)
    assert solver.check() == z3.sat
    assert solver.model()[x].as_long() == 3
