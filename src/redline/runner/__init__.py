"""The ladder runner: resumable climbs of the difficulty ladder, one manifest per run.

`uv run python -m redline.runner --help` lists the options; `make pilot` runs the 1e pilot.
"""

from redline.runner.ladder import RunnerError, run, should_stop
from redline.runner.manifest import Manifest, RunConfig, load

__all__ = ["Manifest", "RunConfig", "RunnerError", "load", "run", "should_stop"]
