"""Hypothesis profiles: derandomized in CI, and no per-example deadline anywhere."""

import os

from hypothesis import settings

settings.register_profile("ci", derandomize=True, max_examples=50, deadline=None)
# Hypothesis's default 200 ms deadline would make the brute-force level-3 examples flaky.
settings.register_profile("dev", deadline=None)
settings.load_profile("ci" if os.environ.get("CI") else "dev")
