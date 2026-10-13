"""The dev/test seed split (the CLAUDE.md hard rule).

Seeds 0-999 are DEV: prompts and strategies are only ever tuned on them. Reported results use
TEST seeds from 1,000,000 up, which are never looked at while tuning. Seeds in between belong to
neither split, so a run can't drift into test seeds by accident.
"""

from typing import Literal

DEV_SEEDS = range(0, 1000)
TEST_SEED_START = 1_000_000

type SeedSplit = Literal["dev", "test"]


def seed_split(seed_start: int, n_seeds: int) -> SeedSplit:
    """The split that seeds seed_start .. seed_start + n_seeds - 1 all belong to."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    last = seed_start + n_seeds - 1
    if seed_start in DEV_SEEDS and last in DEV_SEEDS:
        return "dev"
    if seed_start >= TEST_SEED_START:
        return "test"
    raise ValueError(
        f"seeds {seed_start}-{last} are not all in one split: dev seeds are "
        f"{DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}, test seeds start at {TEST_SEED_START:,}"
    )
