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
from redline.puzzles.logic_grid.model import GENERATOR_VERSION
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


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# SHA-256 of each puzzle's structure: spec, clues and solution. CPython only promises that
# random() stays stable across versions, so a Python upgrade or a code change could silently
# re-map (level, seed) to different puzzles. These must never change: levels 1-4 are unchanged
# since 1a, and generator text v2 (1e.1) left every one byte-identical.
STRUCTURE_GOLDEN = {
    (1, 0): "af3d39c26adedc58a2f243ffbcafea3f505372f5af1a8d31b52160027e0fc77a",
    (2, 1): "0af0305a926139cde7aff8e416951239c390bd708da8e88a67d93c840b1909ca",
    (3, 2): "17a07effe1b8b1724ec67b56bdf093a06d91ee9c6dd2e6b31b79c1d49db3f6c8",
    (4, 3): "1cf757091a59bc9c5e7be8e57844f64a14ff123d952baa3232817b11155e3215",
    (5, 0): "6df386c923710b37b9821337e645a4a8e7c8a11ffc32924df57aecc74ef37b94",
    (6, 0): "eadd59127becc0bf83a7d504de8d5b66c81e57dc38ab9c673d4c496779f2e2e0",
    (7, 0): "c55c84f7084b23bd9716a62bb3d4d6aea53369119ec42207877b98515b728575",
}

# SHA-256 of each puzzle's text. A wording change updates these and nothing above. Last updated
# by "chore(puzzles): labels match attribute keys (text v3)"; (1, 0) and (4, 3) have no color
# attribute, so their text has not changed since v1.
TEXT_GOLDEN = {
    (1, 0): "aac365c72c9f342768615c48208dc93943e673486fabe5e2426251763298a308",
    (2, 1): "c9711fc257b48c7c09d648274bf157fe6c094bf8ae3a9fa2f8ad0a14786e61ea",
    (3, 2): "2ef43b76e0ec7c9c34ecc636e951915664fae0c3d9a5fbc16b5ddfe0240b3637",
    (4, 3): "04e5697f7cee708f567747359bc58ed04e7790a7b7abcf6d0f91d3bc7af814d5",
    (5, 0): "0e6d86f6a92dbeea47a1724e0c1d44e6e6166978da54bdae437d7ed5ac731602",
    (6, 0): "dd38ac87c714c06b9416f5fa5400a81ee987d99cb14cb81b6c66b34015c88ed9",
    (7, 0): "fccce0037080c475823b29541a4e278e32a6394a1c0649c20adacd51d7635a8e",
}
GOLDEN_CASES = [
    pytest.param(*case, marks=[pytest.mark.slow] if case[0] >= 5 else [])
    for case in STRUCTURE_GOLDEN
]


@pytest.mark.parametrize(("level", "seed"), GOLDEN_CASES)
def test_puzzle_structure_is_pinned(level: int, seed: int) -> None:
    structure = puzzle(level, seed).model_dump_json(include={"spec", "clues", "solution"})
    assert sha256(structure) == STRUCTURE_GOLDEN[level, seed]


@pytest.mark.parametrize(("level", "seed"), GOLDEN_CASES)
def test_puzzle_text_is_pinned(level: int, seed: int) -> None:
    assert sha256(render(puzzle(level, seed))) == TEXT_GOLDEN[level, seed]


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
    assert instance.structure == p.spec.model_dump(mode="json")
    assert instance.public() == PublicPuzzle(
        generator="logic_grid",
        text=instance.text,
        answer_format=answer_format(p.spec),
        structure=instance.structure,
    )
    assert instance.answer == p.solution.model_dump(mode="json")
    assert list(instance.instance) == ["spec", "clues", "generator_version", "features_method"]
    assert instance.instance["generator_version"] == GENERATOR_VERSION == 3
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
