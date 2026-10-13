"""The ladder runner with a scripted cell runner: climb order, early stop, Ollama memory, errors,
and the refusals that keep every result mapped to one commit and one config."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from redline.runner import ladder as ladder_module
from redline.runner.ladder import Cell, CellResult, RunnerError, run, run_order, should_stop
from redline.runner.manifest import GitState, RunConfig, load

CLEAN = GitState(sha="a" * 40, dirty=False)
LOCAL_SMALL = "ollama/redline-qwen3.5-0.8b"
LOCAL_LARGE = "ollama/redline-qwen3.5-2b"
API = "groq/llama-3.3-70b-versatile"


class Script:
    """A cell runner that returns scripted solved counts and logs every event in order."""

    def __init__(
        self, solved: Mapping[tuple[str, str, int], int], broken: frozenset[Cell] = frozenset()
    ) -> None:
        self.solved = solved
        self.broken = broken
        self.events: list[str] = []

    def __call__(self, cell: Cell) -> CellResult:
        self.events.append(f"{cell.model} {cell.strategy} L{cell.level}")
        if any(
            (cell.model, cell.strategy, cell.level) == (b.model, b.strategy, b.level)
            for b in self.broken
        ):
            raise ConnectionError("ollama is not running")
        return CellResult(
            None, "success", 5, self.solved.get((cell.model, cell.strategy, cell.level), 0), 0
        )

    def unload_all(self) -> list[str]:
        self.events.append("unload")
        return []


def config(**changes: object) -> RunConfig:
    base = {
        "run_id": "t",
        "models": (LOCAL_SMALL,),
        "strategies": ("direct",),
        "levels": (1, 2, 3, 4, 5),
        "n_seeds": 5,
        "seed_start": 0,
        "early_stop": 2,
    }
    return RunConfig.model_validate({**base, **changes})


@pytest.mark.parametrize(
    ("solved", "early_stop", "stop"),
    [
        ([], 2, False),
        ([0], 2, False),
        ([0, 0], 2, True),
        ([3, 0], 2, False),
        ([3, 0, 0], 2, True),
        ([0, 1, 0], 2, False),
        ([0], 1, True),
        ([2, 0, 0], 3, False),
    ],
)
def test_the_high_jump_rule(solved: list[int], early_stop: int, stop: bool) -> None:
    assert should_stop(solved, early_stop) is stop


def test_local_models_run_first_in_the_given_order() -> None:
    assert run_order([API, LOCAL_LARGE, LOCAL_SMALL]) == [LOCAL_LARGE, LOCAL_SMALL, API]


def test_climbs_until_two_empty_levels(tmp_path: Path) -> None:
    script = Script({(LOCAL_SMALL, "direct", 1): 4, (LOCAL_SMALL, "direct", 2): 1})
    manifest = run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    [ladder] = manifest.ladders
    assert [level.level for level in ladder.levels] == [1, 2, 3, 4]
    assert [level.solved for level in ladder.levels] == [4, 1, 0, 0]
    assert (ladder.stop_reason, ladder.stop_detail) == (
        "early_stop",
        "nothing solved at levels 3, 4",
    )


def test_a_ladder_that_never_stops_is_completed(tmp_path: Path) -> None:
    script = Script({(LOCAL_SMALL, "direct", level): 1 for level in range(1, 6)})
    manifest = run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    assert manifest.ladders[0].stop_reason == "completed"
    assert len(manifest.ladders[0].levels) == 5


def test_one_local_model_in_memory_at_a_time_then_the_api_models(tmp_path: Path) -> None:
    script = Script({})
    run(
        config(models=(API, LOCAL_SMALL, LOCAL_LARGE), strategies=("direct", "cot"), levels=(1, 2)),
        runs_root=tmp_path,
        run_cell=script,
        memory=script,
        git=CLEAN,
    )

    def climb(model: str) -> list[str]:
        return [
            f"{model} {strategy} L{level}" for strategy in ("direct", "cot") for level in (1, 2)
        ]

    assert script.events == [
        "unload",
        *climb(LOCAL_SMALL),
        "unload",
        "unload",
        *climb(LOCAL_LARGE),
        "unload",
        *climb(API),
    ]


def test_a_failing_cell_ends_its_ladder_and_the_run_goes_on(tmp_path: Path) -> None:
    broken = frozenset({Cell(LOCAL_SMALL, "direct", 2, 5, 0, Path())})
    script = Script({(LOCAL_SMALL, "direct", 1): 2, (LOCAL_SMALL, "cot", 1): 2}, broken)
    manifest = run(
        config(strategies=("direct", "cot"), levels=(1, 2, 3)),
        runs_root=tmp_path,
        run_cell=script,
        memory=script,
        git=CLEAN,
    )
    direct, cot = manifest.ladders
    assert direct.stop_reason == "error"
    assert direct.stop_detail is not None and "ollama is not running" in direct.stop_detail
    assert [level.status for level in direct.levels] == ["success", "error"]
    assert cot.stop_reason == "early_stop"


def test_the_manifest_records_profiles_split_and_sessions(tmp_path: Path) -> None:
    script = Script({})
    run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    manifest = load(tmp_path / "t")
    assert manifest.seed_split == "dev"
    assert manifest.provenance.git_sha == CLEAN.sha
    profile = manifest.profiles[LOCAL_SMALL]
    assert profile.modelfile_sha256 is not None and len(profile.modelfile_sha256) == 64
    assert len(manifest.sessions) == 2
    assert all(session.ended_at is not None for session in manifest.sessions)
    assert manifest.started_at == manifest.sessions[0].started_at
    assert manifest.ended_at == manifest.sessions[-1].ended_at
    assert len(manifest.ladders) == 1  # re-climbed in place, not appended


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path: Path) -> None:
    script = Script({})
    dirty = GitState(sha=CLEAN.sha, dirty=True)
    with pytest.raises(RunnerError, match="uncommitted"):
        run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=dirty)
    assert not (tmp_path / "t").exists()
    manifest = run(
        config(), runs_root=tmp_path, run_cell=script, memory=script, git=dirty, allow_dirty=True
    )
    assert manifest.git_dirty and manifest.sessions[0].git_dirty


@pytest.mark.parametrize(
    ("changed", "git", "match"),
    [
        ({"n_seeds": 6}, CLEAN, "config"),
        ({}, GitState(sha="b" * 40, dirty=False), "commit"),
    ],
)
def test_a_resume_must_match_config_and_commit(
    tmp_path: Path, changed: dict[str, object], git: GitState, match: str
) -> None:
    script = Script({})
    run(config(), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    with pytest.raises(RunnerError, match=match):
        run(config(**changed), runs_root=tmp_path, run_cell=script, memory=script, git=git)


@pytest.mark.parametrize(
    "changes",
    [
        {"models": ("ollama/unregistered",)},
        {"strategies": ("guess",)},
        {"levels": (2, 1)},
        {"levels": (1, 9)},
        {"early_stop": 0},
        {"seed_start": 990, "n_seeds": 20},
        {"strategies": ("direct", "direct")},
    ],
)
def test_bad_configs_are_refused_before_anything_runs(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    script = Script({})
    with pytest.raises(RunnerError):
        run(config(**changes), runs_root=tmp_path, run_cell=script, memory=script, git=CLEAN)
    assert script.events == []
    assert not (tmp_path / "t").exists()


def test_an_interrupt_stops_the_whole_run_and_saves_the_manifest(tmp_path: Path) -> None:
    class Interrupted(Script):
        def __call__(self, cell: Cell) -> CellResult:
            if cell.level == 2:
                raise KeyboardInterrupt
            return super().__call__(cell)

    script = Interrupted({(LOCAL_SMALL, "direct", 1): 3})
    with pytest.raises(KeyboardInterrupt):
        run(
            config(strategies=("direct", "cot")),
            runs_root=tmp_path,
            run_cell=script,
            memory=script,
            git=CLEAN,
        )
    assert script.events == ["unload", f"{LOCAL_SMALL} direct L1"]  # no cot, no final unload
    manifest = load(tmp_path / "t")
    assert manifest.sessions[0].ended_at is not None
    [ladder] = manifest.ladders  # the finished level is kept; cot never started
    assert [level.level for level in ladder.levels] == [1]
    assert ladder.stop_reason is None


def test_an_eval_set_that_returns_no_log_was_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ladder_module, "eval_set", lambda *args, **kwargs: (False, []))
    with pytest.raises(KeyboardInterrupt):
        ladder_module.inspect_cells("none")(Cell("mockllm/model", "direct", 1, 1, 0, Path()))
