"""Command line for the ladder runner.

    uv run python -m redline.runner --models ollama/redline-qwen3.5-0.8b \\
        --strategies direct cot --levels 1-7 --n-seeds 5 --seed-start 0 --early-stop 2 \\
        --run-id my-run

Exit status: 0 when every ladder completed or stopped early, 1 when a ladder ended in an error
(rerun the same command to resume it), 2 when the runner refused to start.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from inspect_ai.util import DisplayType
from pydantic import ValidationError

from redline.runner.ladder import RUNS_ROOT, RunnerError, inspect_cells, run
from redline.runner.manifest import RunConfig
from redline.tasks.logic_grid import parse_levels

logger = logging.getLogger("redline.runner")


def parse_level_spec(spec: str) -> list[int]:
    """ "1-7", "1,3,5" or a mix like "1-3,5"; checked against the ladder."""
    levels: list[int] = []
    try:
        for part in (part.strip() for part in spec.split(",")):
            if "-" in part:
                low, high = part.split("-", 1)
                levels.extend(range(int(low), int(high) + 1))
            elif part:
                levels.append(int(part))
        return parse_levels(levels)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"bad --levels {spec!r}: {error}") from None


def split_items(items: Sequence[str]) -> tuple[str, ...]:
    """Accept both `--models a b` and `--models a,b`."""
    return tuple(part.strip() for item in items for part in item.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m redline.runner",
        description="Climb the logic-grid ladder for each (model, strategy), resumably.",
    )
    parser.add_argument("--models", nargs="+", required=True, help="Inspect model IDs")
    parser.add_argument("--strategies", nargs="+", required=True)
    parser.add_argument("--levels", type=parse_level_spec, default="1-7", help="e.g. 1-7, 1,3,5")
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--early-stop",
        type=int,
        default=2,
        help="stop a climb after this many consecutive levels with nothing solved",
    )
    parser.add_argument("--run-id", required=True, help="logs go to <runs-root>/<run-id>/")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run on uncommitted changes (recorded)"
    )
    parser.add_argument("--display", choices=get_args(DisplayType), default="plain")
    return parser


def config_from(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        run_id=args.run_id,
        models=split_items(args.models),
        strategies=split_items(args.strategies),
        levels=tuple(args.levels),
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        early_stop=args.early_stop,
    )


def configure_logging() -> None:
    """Progress lines from redline's own loggers. Inspect sets the root logger to WARNING when an
    eval starts, so the package logger gets its own level and handler and doesn't propagate."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    package = logging.getLogger("redline")
    package.handlers = [handler]
    package.setLevel(logging.INFO)
    package.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    try:
        config = config_from(args)
        manifest = run(
            config,
            runs_root=args.runs_root,
            allow_dirty=args.allow_dirty,
            run_cell=inspect_cells(args.display),
        )
    except (RunnerError, ValidationError) as error:
        logger.error("refusing to run: %s", error)
        return 2
    for ladder in manifest.ladders:
        levels = ", ".join(f"L{lvl.level} {lvl.solved}/{lvl.n}" for lvl in ladder.levels)
        logger.info("%s %s: %s; %s", ladder.model, ladder.strategy, ladder.stop_detail, levels)
    logger.info(
        "run %s: %.0f s; manifest at %s",
        config.run_id,
        manifest.runtime_seconds(),
        args.runs_root / config.run_id / "manifest.json",
    )
    return 1 if any(ladder.stop_reason == "error" for ladder in manifest.ladders) else 0


if __name__ == "__main__":
    sys.exit(main())
