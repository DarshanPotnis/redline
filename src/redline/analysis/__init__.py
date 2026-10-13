"""Run analysis: per-level solved rates with Wilson intervals, the ladder chart and report.

`make report RUN=<run_id>` exports the run to Parquet and writes reports/<run_id>.md.
"""

from redline.analysis.report import build_report

__all__ = ["build_report"]
