"""Difficulty features: repeatable, well formed, and rising with level."""

import itertools
import statistics

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from redline.puzzles.base import PuzzleInstance
from redline.puzzles.logic_grid import LogicGridPuzzle, generate
from redline.puzzles.logic_grid.difficulty import measure
from redline.puzzles.logic_grid.model import CLUE_KINDS

levels = st.sampled_from([1, 2, 3, 4])
seeds = st.integers(min_value=0, max_value=2**32 - 1)
FEATURES = {
    "z3_conflicts_mean",
    "z3_conflicts_std",
    "z3_decisions_mean",
    "n_clues",
    *(f"kind_{kind}" for kind in CLUE_KINDS),
}


@settings(max_examples=10)
@given(levels, seeds)
def test_measure_repeats_exactly_in_one_process(level: int, seed: int) -> None:
    p = generate(level, seed)
    assert measure(p) == measure(p)


@settings(max_examples=20)
@given(levels, seeds)
def test_features_are_well_formed(level: int, seed: int) -> None:
    p = generate(level, seed)
    features = measure(p)
    assert set(features) == FEATURES
    assert all(isinstance(value, float) for value in features.values())
    kind_total = sum(features[f"kind_{kind}"] for kind in CLUE_KINDS)
    assert features["n_clues"] == kind_total == len(p.clues)
    assert features["kind_at"] == 0
    assert features["z3_conflicts_std"] >= 0


def test_measure_rejects_a_puzzle_without_a_unique_solution() -> None:
    p = generate(3, 0)
    # Clue sets are minimal, so dropping any clue leaves more than one solution.
    loose = LogicGridPuzzle(
        spec=p.spec, clues=p.clues[1:], solution=p.solution, level=p.level, seed=p.seed
    )
    with pytest.raises(ValueError, match="exactly one solution"):
        measure(loose)


def test_solver_effort_rises_across_levels_1_to_4() -> None:
    # The CLAUDE.md difficulty rule, checked on the fast levels.
    means = [
        statistics.fmean(measure(generate(level, seed))["z3_conflicts_mean"] for seed in range(20))
        for level in [1, 2, 3, 4]
    ]
    assert all(lower < higher for lower, higher in itertools.pairwise(means)), means


def test_features_default_to_empty() -> None:
    # Generators without difficulty features (e.g. Reasoning Gym) leave the field out.
    instance = PuzzleInstance(
        generator="other", level=1, seed=0, text="", answer={}, instance={}, search_space=1
    )
    assert instance.features == {}
