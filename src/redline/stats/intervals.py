"""Confidence intervals for a proportion."""

import math
from statistics import NormalDist


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """The Wilson score interval for `successes` out of `n`.

    Unlike the normal approximation it stays inside [0, 1] and keeps a nonzero width at 0/n and
    n/n, which matters at the handful of samples a ladder level gets.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in 0..{n}, got {successes}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    shrink = 1 + z * z / n
    center = (p + z * z / (2 * n)) / shrink
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / shrink
    # At 0/n and n/n one end is exactly 0 or 1; computing it would leave rounding residue.
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return low, high
