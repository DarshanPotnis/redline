"""The run manifest: what was asked, at which commit, with which profiles, how each ladder ended.

It lives at logs/runs/<run_id>/manifest.json and is rewritten (atomically) after every level, so
a crash leaves it describing everything finished so far. Log paths are relative to the run dir.
"""

import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import z3
from pydantic import BaseModel, ConfigDict, field_validator

from redline.runtime.profiles import REPO_ROOT, ModelProfile
from redline.seeds import SeedSplit

MANIFEST = "manifest.json"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

type StopReason = Literal["completed", "early_stop", "error"]


class RunConfig(BaseModel):
    """Everything the runner was asked to do; a resumed run must ask for exactly the same."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    generator: str = "logic_grid"  # a `--generator` name from puzzles/registry.py
    models: tuple[str, ...]
    strategies: tuple[str, ...]
    levels: tuple[int, ...]
    n_seeds: int
    seed_start: int
    early_stop: int  # stop climbing after this many consecutive levels with nothing solved

    @field_validator("run_id")
    @classmethod
    def _path_safe(cls, run_id: str) -> str:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"run_id must be letters, digits, '.', '_' or '-', got {run_id!r}")
        return run_id


class LevelRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int
    log: str | None  # relative to the run dir; None if eval_set produced no log
    status: str  # the eval log's status: success, error, cancelled, started
    n: int  # samples in the log
    solved: int
    context_limited: int


class LadderRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    strategy: str
    levels: tuple[LevelRecord, ...] = ()
    stop_reason: StopReason | None = None  # None while still climbing
    stop_detail: str | None = None


class Session(BaseModel):
    """One invocation of the runner; a resumed run adds a session."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    ended_at: datetime | None = None
    git_dirty: bool


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    git_sha: str
    python: str
    platform: str
    inspect_ai: str
    z3: str
    redline: str
    reasoning_gym: str | None = None  # recorded since 1f


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: RunConfig
    seed_split: SeedSplit
    provenance: Provenance
    git_dirty: bool  # true if any session ran on a dirty tree
    profiles: dict[str, ModelProfile]  # every model in the run
    started_at: datetime
    ended_at: datetime | None = None
    sessions: list[Session]
    ladders: list[LadderRecord]

    def record(self, ladder: LadderRecord) -> None:
        """Replace the (model, strategy) ladder if present, else append it."""
        key = (ladder.model, ladder.strategy)
        for i, existing in enumerate(self.ladders):
            if (existing.model, existing.strategy) == key:
                self.ladders[i] = ladder
                return
        self.ladders.append(ladder)

    def runtime_seconds(self) -> float:
        """Wall time summed over finished sessions."""
        return sum(
            (session.ended_at - session.started_at).total_seconds()
            for session in self.sessions
            if session.ended_at is not None
        )


@dataclass(frozen=True)
class GitState:
    sha: str
    dirty: bool  # uncommitted changes or untracked files


def git_state(repo: Path = REPO_ROOT) -> GitState:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    return GitState(sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))


def provenance(sha: str) -> Provenance:
    return Provenance(
        git_sha=sha,
        python=sys.version.split()[0],
        platform=platform.platform(),
        inspect_ai=version("inspect_ai"),
        z3=z3.get_version_string(),
        redline=version("redline"),
        reasoning_gym=version("reasoning_gym"),
    )


def now() -> datetime:
    return datetime.now(UTC)


def load(run_dir: Path) -> Manifest:
    return Manifest.model_validate_json((run_dir / MANIFEST).read_text())


def save(manifest: Manifest, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / f".{MANIFEST}.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2) + "\n")
    tmp.replace(run_dir / MANIFEST)
