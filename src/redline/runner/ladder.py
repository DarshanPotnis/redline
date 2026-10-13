"""The ladder runner: each (model, strategy) climbs the levels until the high-jump rule stops it.

Every (model, strategy, level) cell is its own Inspect `eval_set` with its own log dir. One
eval_set over the whole run can't work, because the next level depends on the last result, so
the task list isn't known up front. Per cell it gives resume for free: a complete log is
returned without calling the model, and an interrupted one is finished, reusing its completed
samples. A restarted run therefore re-climbs the finished levels from their logs in seconds and
makes the same early-stop decisions.
"""

import contextvars
import logging
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import EvalSampleSummary, read_eval_log_sample_summaries
from inspect_ai.model import Model
from inspect_ai.util import DisplayType

from redline.runner.manifest import (
    MANIFEST,
    GitState,
    LadderRecord,
    LevelRecord,
    Manifest,
    RunConfig,
    Session,
    StopReason,
    git_state,
    load,
    now,
    provenance,
    save,
)
from redline.runner.ollama import ModelMemory, Ollama
from redline.runtime.profiles import ModelProfile, profile_for
from redline.seeds import seed_split
from redline.strategies import get_strategy
from redline.tasks.logic_grid import logic_grid, parse_levels

logger = logging.getLogger(__name__)

RUNS_ROOT = Path("logs/runs")
# eval_set retries a failed cell, reusing its completed samples. Its default of 10 would keep a
# dead Ollama server busy for a long time; 3 is enough for a transient error.
RETRY_ATTEMPTS = 3


class RunnerError(RuntimeError):
    """The runner refused to start, and nothing ran."""


@dataclass(frozen=True)
class Cell:
    model: str
    strategy: str
    level: int
    n_seeds: int
    seed_start: int
    log_dir: Path


@dataclass(frozen=True)
class CellResult:
    log: Path | None
    status: str  # the eval log's status
    n: int
    solved: int
    context_limited: int
    error: str | None = None


type CellRunner = Callable[[Cell], CellResult]


def should_stop(solved_per_level: Sequence[int], early_stop: int) -> bool:
    """The high-jump rule: stop after `early_stop` consecutive levels with nothing solved."""
    recent = solved_per_level[-early_stop:]
    return len(recent) == early_stop and not any(recent)


def run_order(models: Sequence[str]) -> list[str]:
    """Local models first, in the order given, then the API models."""
    return sorted(models, key=lambda model: profile_for(model).provider != "ollama")


def cell_dir(run_dir: Path, model: str, strategy: str, level: int) -> Path:
    return run_dir / model.replace("/", "__") / strategy / f"L{level}"


def validate(config: RunConfig) -> None:
    """Every model has a profile, every strategy and level exists, and the seeds are one split."""
    for model in config.models:
        profile_for(model)
    for strategy in config.strategies:
        get_strategy(strategy)
    parse_levels(list(config.levels))
    for name, items in (
        ("models", config.models),
        ("strategies", config.strategies),
        ("levels", config.levels),
    ):
        if len(set(items)) != len(items):
            raise ValueError(f"{name} repeat: {list(items)}")
    if list(config.levels) != sorted(config.levels):
        raise ValueError(f"levels must be in climbing order, got {list(config.levels)}")
    if config.early_stop < 1:
        raise ValueError(f"early_stop must be at least 1, got {config.early_stop}")
    seed_split(config.seed_start, config.n_seeds)


def inspect_cells(
    display: DisplayType = "plain", models: Mapping[str, Model] | None = None
) -> CellRunner:
    """Run each cell as an eval_set of the logic-grid task. `models` swaps in Model objects for
    model IDs (tests pass a mock with scripted outputs)."""

    def run_cell(cell: Cell) -> CellResult:
        # In a copy of the context: eval_set leaves Inspect's context variables (the active
        # model among them) set when it returns, and no cell should see the last one's.
        _, logs = contextvars.copy_context().run(
            eval_set,
            logic_grid,
            log_dir=str(cell.log_dir),
            model=(models or {}).get(cell.model, cell.model),
            task_args={
                "levels": cell.level,
                "n_seeds": cell.n_seeds,
                "seed_start": cell.seed_start,
                "strategy": cell.strategy,
            },
            retry_attempts=RETRY_ATTEMPTS,
            display=display,
            ctl_server=False,  # several eval_sets run in one process
        )
        if not logs:
            return CellResult(None, "error", 0, 0, 0, "eval_set returned no log")
        [log] = logs
        grids = [_grid_scores(summary) for summary in read_eval_log_sample_summaries(log.location)]
        scored = [grid for grid in grids if grid is not None]
        return CellResult(
            log=local_path(log.location),
            status=log.status,
            n=len(scored),
            solved=sum(grid["solved"] == 1 for grid in scored),
            context_limited=sum(grid["context_limited"] == 1 for grid in scored),
            error=log.error.message if log.error else None,
        )

    return run_cell


def run(
    config: RunConfig,
    *,
    runs_root: Path = RUNS_ROOT,
    allow_dirty: bool = False,
    run_cell: CellRunner | None = None,
    memory: ModelMemory | None = None,
    git: GitState | None = None,
) -> Manifest:
    """Climb every (model, strategy) ladder, writing the manifest as it goes."""
    try:
        validate(config)
    except ValueError as error:
        raise RunnerError(str(error)) from error
    git = git or git_state()
    if git.dirty and not allow_dirty:
        raise RunnerError(
            "the git tree has uncommitted changes, so results would map to no commit; commit "
            "first, or pass --allow-dirty (recorded in the manifest)"
        )
    run_dir = runs_root / config.run_id
    manifest = _start(run_dir, config, git)
    run_cell = run_cell or inspect_cells()
    memory = memory or Ollama()
    try:
        for model in run_order(config.models):
            local = profile_for(model).provider == "ollama"
            if local:
                memory.unload_all()  # one local model in memory at a time
            for strategy in config.strategies:
                _climb(model, strategy, config, run_dir, run_cell, manifest)
            if local:
                memory.unload_all()
    finally:
        manifest.sessions[-1].ended_at = manifest.ended_at = now()
        save(manifest, run_dir)
    return manifest


def _climb(
    model: str,
    strategy: str,
    config: RunConfig,
    run_dir: Path,
    run_cell: CellRunner,
    manifest: Manifest,
) -> None:
    ladder = LadderRecord(model=model, strategy=strategy)
    solved: list[int] = []
    for level in config.levels:
        cell = Cell(
            model,
            strategy,
            level,
            config.n_seeds,
            config.seed_start,
            cell_dir(run_dir, model, strategy, level),
        )
        try:
            result = run_cell(cell)
        except Exception as error:  # a broken cell ends this ladder, not the run
            logger.exception("%s %s L%d failed", model, strategy, level)
            result = CellResult(None, "error", 0, 0, 0, f"{type(error).__name__}: {error}")
        logger.info(
            "%s %s L%d: %d/%d solved, %d context-limited (%s)",
            model,
            strategy,
            level,
            result.solved,
            result.n,
            result.context_limited,
            result.status,
        )
        record = LevelRecord(
            level=level,
            log=None if result.log is None else _relative(result.log, run_dir),
            status=result.status,
            n=result.n,
            solved=result.solved,
            context_limited=result.context_limited,
        )
        ladder = ladder.model_copy(update={"levels": (*ladder.levels, record)})
        if result.status != "success":
            ladder = _stopped(ladder, "error", f"level {level}: {result.error or result.status}")
            break
        solved.append(result.solved)
        if should_stop(solved, config.early_stop):
            zero = ", ".join(str(lvl) for lvl in config.levels[: len(solved)][-config.early_stop :])
            ladder = _stopped(ladder, "early_stop", f"nothing solved at levels {zero}")
            break
        manifest.record(ladder)
        save(manifest, run_dir)
    else:
        ladder = _stopped(ladder, "completed", "climbed every level")
    manifest.record(ladder)
    save(manifest, run_dir)
    logger.info("%s %s: %s (%s)", model, strategy, ladder.stop_reason, ladder.stop_detail)


def _start(run_dir: Path, config: RunConfig, git: GitState) -> Manifest:
    profiles = {model: profile_for(model) for model in config.models}
    session = Session(started_at=now(), git_dirty=git.dirty)
    if (run_dir / MANIFEST).exists():
        manifest = load(run_dir)
        problems = _resume_problems(manifest, config, git, profiles)
        if problems:
            raise RunnerError(
                f"can't resume run {config.run_id!r}: {'; '.join(problems)}. Use a new --run-id."
            )
        manifest.sessions.append(session)
        manifest.git_dirty = manifest.git_dirty or git.dirty
        manifest.ended_at = None
        logger.info("resuming run %s (session %d)", config.run_id, len(manifest.sessions))
    else:
        manifest = Manifest(
            config=config,
            seed_split=seed_split(config.seed_start, config.n_seeds),
            provenance=provenance(git.sha),
            git_dirty=git.dirty,
            profiles=profiles,
            started_at=session.started_at,
            sessions=[session],
            ladders=[],
        )
    save(manifest, run_dir)
    return manifest


def _resume_problems(
    manifest: Manifest, config: RunConfig, git: GitState, profiles: Mapping[str, ModelProfile]
) -> list[str]:
    problems = []
    if manifest.config != config:
        problems.append(f"its config was {manifest.config.model_dump()}")
    if manifest.provenance.git_sha != git.sha:
        problems.append(f"it ran at commit {manifest.provenance.git_sha[:12]}, now {git.sha[:12]}")
    if manifest.profiles != profiles:
        problems.append("a model profile changed")
    return problems


def _stopped(ladder: LadderRecord, reason: StopReason, detail: str) -> LadderRecord:
    return ladder.model_copy(update={"stop_reason": reason, "stop_detail": detail})


def local_path(location: str) -> Path:
    """A log location as a path: eval_set reports logs it found on disk as file: URIs."""
    if location.startswith("file:"):
        return Path(urllib.parse.unquote(urllib.parse.urlparse(location).path))
    return Path(location)


def _relative(log: Path, run_dir: Path) -> str:
    return log.resolve().relative_to(run_dir.resolve()).as_posix()


def _grid_scores(summary: EvalSampleSummary) -> Mapping[str, object] | None:
    score = (summary.scores or {}).get("grid_verifier")
    if score is None or not isinstance(score.value, Mapping):
        return None
    return score.value
