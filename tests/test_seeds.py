"""The dev/test seed split is a locked decision: these boundaries must not move."""

import pytest

from redline.seeds import DEV_SEEDS, TEST_SEED_START, seed_split


def test_the_split_boundaries() -> None:
    assert (DEV_SEEDS.start, DEV_SEEDS.stop, TEST_SEED_START) == (0, 1000, 1_000_000)


@pytest.mark.parametrize(
    ("seed_start", "n_seeds", "split"),
    [
        (0, 5, "dev"),
        (0, 1000, "dev"),
        (999, 1, "dev"),
        (1_000_000, 5, "test"),
        (2_000_000, 1, "test"),
    ],
)
def test_ranges_inside_one_split(seed_start: int, n_seeds: int, split: str) -> None:
    assert seed_split(seed_start, n_seeds) == split


@pytest.mark.parametrize(
    ("seed_start", "n_seeds"),
    [
        (990, 20),  # straddles the end of dev
        (1000, 5),  # the unassigned gap
        (999_999, 2),  # gap into test
        (-1, 2),
        (0, 0),
    ],
)
def test_ranges_outside_one_split_are_refused(seed_start: int, n_seeds: int) -> None:
    with pytest.raises(ValueError):
        seed_split(seed_start, n_seeds)
