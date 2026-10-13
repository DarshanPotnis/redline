"""`make report RUN=<run_id>`: Parquet export, ladder chart and markdown report for one run."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from redline.analysis.report import REPORTS, build_report
from redline.export.parquet import DATA_ROOT
from redline.runner.ladder import RUNS_ROOT

logger = logging.getLogger("redline.analysis")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m redline.analysis", description="Write the ladder report for a run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = args.runs_root / args.run_id
    if not (run_dir / "manifest.json").is_file():
        parser.error(f"no run manifest at {run_dir}/manifest.json")
    paths = build_report(run_dir, args.data_root, args.reports)
    logger.info("parquet: %s\nreport: %s\nchart: %s", paths.parquet, paths.markdown, paths.chart)


if __name__ == "__main__":
    main()
