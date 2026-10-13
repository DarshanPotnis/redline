"""Wilson intervals: textbook values, and a brute-force oracle from the score test itself."""

from statistics import NormalDist

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from redline.stats import wilson_interval

Z = NormalDist().inv_cdf(0.975)
STEP = 1e-4


def test_textbook_values() -> None:
    # 0/5 and 5/5: the normal approximation would give a zero-width interval at 0 or 1.
    assert wilson_interval(0, 5) == pytest.approx((0.0, 0.4345), abs=1e-4)
    assert wilson_interval(5, 5) == pytest.approx((0.5655, 1.0), abs=1e-4)
    assert wilson_interval(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)


@settings(max_examples=40)
@given(st.integers(1, 60).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_matches_the_score_test_by_brute_force(case: tuple[int, int]) -> None:
    # The Wilson interval is every p the score test does not reject:
    # (k/n - p)^2 <= z^2 p (1 - p) / n. Scan p on a fine grid and compare the ends.
    successes, n = case
    phat = successes / n
    kept = [
        i * STEP
        for i in range(int(1 / STEP) + 1)
        if (phat - i * STEP) ** 2 <= Z * Z * (i * STEP) * (1 - i * STEP) / n
    ]
    low, high = wilson_interval(successes, n)
    assert low == pytest.approx(kept[0], abs=STEP)
    assert high == pytest.approx(kept[-1], abs=STEP)
    assert low <= phat <= high


def test_symmetry() -> None:
    low, high = wilson_interval(2, 7)
    assert wilson_interval(5, 7) == pytest.approx((1 - high, 1 - low))


@pytest.mark.parametrize(("successes", "n"), [(0, 0), (-1, 3), (4, 3)])
def test_rejects_impossible_counts(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)
