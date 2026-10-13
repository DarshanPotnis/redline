"""Generator properties: valid, unique, minimal, fully relational, and deterministic."""

import functools
import hashlib
import itertools
import math
import os
import random
import subprocess
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from redline.puzzles.base import PublicPuzzle, PuzzleInstance
from redline.puzzles.logic_grid import GENERATOR, LADDER, LogicGridPuzzle, generate, render
from redline.puzzles.logic_grid.clues import holds
from redline.puzzles.logic_grid.difficulty import method
from redline.puzzles.logic_grid.generate import default_counter, draw_spec
from redline.puzzles.logic_grid.render import answer_format
from redline.puzzles.logic_grid.solve_backtrack import count_solutions
from redline.puzzles.logic_grid.solve_bruteforce import count_solutions_bruteforce
from redline.puzzles.logic_grid.solve_z3 import count_solutions_z3

FAST_LEVELS = [1, 2, 3, 4]  # Hypothesis-driven; levels 5-7 run on fixed seeds, marked slow
UPPER_CASES = [(level, seed) for level in (5, 6, 7) for seed in (0, 1, 2)]
levels = st.sampled_from(FAST_LEVELS)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


@functools.cache
def puzzle(level: int, seed: int) -> LogicGridPuzzle:
    return generate(level, seed)


@given(levels, seeds)
def test_key_satisfies_every_clue(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    assert all(holds(clue, p.solution) for clue in p.clues)


@given(st.sampled_from([1, 2, 3]), seeds)
def test_exactly_one_solution_by_brute_force(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    assert count_solutions_bruteforce(p.spec, p.clues) == 1


@given(seeds)
def test_exactly_one_solution_at_level_4(seed: int) -> None:
    p = puzzle(4, seed)
    assert count_solutions(p.spec, p.clues) == 1


@given(levels, seeds)
def test_every_clue_is_needed(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    for i in range(len(p.clues)):
        without = p.clues[:i] + p.clues[i + 1 :]
        assert count_solutions(p.spec, without, cap=2) == 2


@given(levels, seeds)
def test_puzzles_are_fully_relational(level: int, seed: int) -> None:
    # A locked design decision, not a side effect: see the generate.py docstring.
    assert all(clue.kind != "at" for clue in puzzle(level, seed).clues)


@given(levels, seeds)
def test_same_level_and_seed_give_the_same_puzzle(level: int, seed: int) -> None:
    assert generate(level, seed) == generate(level, seed)


@settings(max_examples=10)  # Z3-counted generation takes 50-300 ms at levels 1-4
@given(levels, seeds)
def test_the_counter_never_changes_the_puzzle(level: int, seed: int) -> None:
    by_backtracking = generate(level, seed, counter=count_solutions)
    assert generate(level, seed, counter=count_solutions_z3) == by_backtracking


@pytest.mark.parametrize("level", sorted(LADDER))
def test_default_counter_is_backtracking_through_level_4(level: int) -> None:
    expected = count_solutions if level <= 4 else count_solutions_z3
    assert default_counter(draw_spec(random.Random(0), level)) is expected


@pytest.mark.slow
@pytest.mark.parametrize(("level", "seed"), UPPER_CASES)
def test_upper_level_key_satisfies_every_clue(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    assert all(holds(clue, p.solution) for clue in p.clues)


@pytest.mark.slow
@pytest.mark.parametrize(("level", "seed"), UPPER_CASES)
def test_upper_level_has_exactly_one_solution(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    assert count_solutions_z3(p.spec, p.clues, cap=2) == 1
    if level == 5:  # an independent check; from level 6 on, backtracking takes ~10 s a puzzle
        assert count_solutions(p.spec, p.clues, cap=2) == 1


@pytest.mark.slow
@pytest.mark.parametrize(("level", "seed"), UPPER_CASES)
def test_upper_level_every_clue_is_needed(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    for i in range(len(p.clues)):
        without = p.clues[:i] + p.clues[i + 1 :]
        assert count_solutions_z3(p.spec, without, cap=2) == 2


@pytest.mark.slow
@pytest.mark.parametrize(("level", "seed"), UPPER_CASES)
def test_upper_level_is_fully_relational(level: int, seed: int) -> None:
    assert all(clue.kind != "at" for clue in puzzle(level, seed).clues)


HASH_SEED_CASES = [(1, 0), (2, 1), (3, 2), (4, 3)]
HASH_SEED_SCRIPT = f"""
from redline.puzzles.logic_grid import generate
print(hash("redline"))
for level, seed in {HASH_SEED_CASES!r}:
    print(generate(level, seed).model_dump_json())
"""


def test_generation_ignores_the_string_hash_seed() -> None:
    runs = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", HASH_SEED_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(result.stdout.splitlines())
    (hash_1, *puzzles_1), (hash_2, *puzzles_2) = runs
    assert hash_1 != hash_2, "string hashing should differ between the two processes"
    assert len(puzzles_1) == len(HASH_SEED_CASES)
    assert puzzles_1 == puzzles_2


def identity_json(instance: PuzzleInstance) -> str:
    """The puzzle's identity: the instance minus its difficulty features (Z3 statistics may
    differ across platforms) and its answer format (pinned on its own in test_render.py)."""
    return instance.model_dump_json(
        exclude={"features": True, "answer_format": True, "instance": {"features_method": True}}
    )


# SHA-256 of each instance's identity JSON (text included). CPython only promises that random()
# stays stable across versions, so a Python upgrade or a code change could silently re-map
# (level, seed) to different puzzles. If a change is intended, update these digests. Levels 1-4
# are unchanged since 1a.
GOLDEN = {
    (1, 0): "703b68dbd37b51a6232ef5feb5eb8f962c0710e569650879db4f70736694d331",
    (2, 1): "0c8263ca0a15dc2ae0ef1309f7fe53a89724be4ae1c24a77ce8c0272691b8ee2",
    (3, 2): "f87fba3e1d3069823b7a8a019783055c36946fa2419ec991d5b79081ad726057",
    (4, 3): "fc7cec3f993b314988efb6eecdb67f5d9cd13a2c3a248fe4837d86778428ba92",
    (5, 0): "a2dfd4635a0f4dbb325b6cea54d3da72836181f32c8808ba5f616e0ea36c1e06",
    (6, 0): "1f233610f616207097c463b463d458c537b44000ea9bb66712b0553d0e15cd77",
    (7, 0): "4045ca5d6d94c38dd5146b9d7c3e7f4af09233b151c1f6df619213fd24f910fb",
}


@pytest.mark.parametrize(
    ("level", "seed"),
    [pytest.param(*case, marks=[pytest.mark.slow] if case[0] >= 5 else []) for case in GOLDEN],
)
def test_puzzle_identity_is_pinned(level: int, seed: int) -> None:
    payload = identity_json(puzzle(level, seed).to_instance())
    assert hashlib.sha256(payload.encode()).hexdigest() == GOLDEN[level, seed]


def test_search_space_strictly_increases_with_level() -> None:
    spaces = [draw_spec(random.Random(0), level).search_space for level in sorted(LADDER)]
    assert spaces == [36, 216, 13_824, 331_776, 120**4, 720**5, 720**6]
    assert all(lower < higher for lower, higher in itertools.pairwise(spaces))


@settings(max_examples=20)  # to_instance() runs five Z3 uniqueness proofs
@given(levels, seeds)
def test_instance_holds_text_key_hidden_puzzle_and_features(level: int, seed: int) -> None:
    p = puzzle(level, seed)
    instance = p.to_instance()
    houses, attributes = LADDER[level]
    assert instance.generator == "logic_grid"
    assert (instance.level, instance.seed) == (level, seed)
    assert instance.text == render(p)
    assert instance.answer_format == answer_format(p.spec)
    assert instance.public() == PublicPuzzle(
        text=instance.text, answer_format=answer_format(p.spec)
    )
    assert instance.answer == p.solution.model_dump(mode="json")
    assert list(instance.instance) == ["spec", "clues", "features_method"]
    assert instance.instance["features_method"] == method()
    assert instance.features["n_clues"] == len(p.clues)
    assert instance.search_space == math.factorial(houses) ** attributes
    assert PuzzleInstance.model_validate_json(instance.model_dump_json()) == instance


def test_generator_follows_the_protocol() -> None:
    assert GENERATOR.name == "logic_grid"
    assert list(GENERATOR.levels) == [1, 2, 3, 4, 5, 6, 7]
    assert GENERATOR.generate(2, 5) == generate(2, 5).to_instance()


@pytest.mark.parametrize("level", [0, 8, -1])
def test_unknown_level_is_rejected(level: int) -> None:
    with pytest.raises(ValueError, match="unknown level"):
        generate(level, 0)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate(1, -1)
