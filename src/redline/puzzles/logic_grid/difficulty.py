"""Per-puzzle difficulty features: Z3's effort to prove the solution unique, measured K times.

Each run is one uniqueness proof (count to cap=2) in a fresh `z3.Context`: in a shared
context, identical runs reported different statistics. Runs differ by the solver's
`random_seed` and by the order the clues are asserted in, shuffled by an rng derived from the
puzzle and the run index.
"""

import random
import statistics
from typing import Any

import z3
from pydantic import JsonValue

from redline.puzzles.logic_grid.model import CLUE_KINDS, LogicGridPuzzle
from redline.puzzles.logic_grid.solve_z3 import count_models, encode

RUNS = 5  # K, the number of uniqueness proofs per puzzle


def measure(puzzle: LogicGridPuzzle) -> dict[str, float]:
    proofs = [_prove_unique(puzzle, k) for k in range(RUNS)]
    conflicts = [proof_conflicts for proof_conflicts, _ in proofs]
    decisions = [proof_decisions for _, proof_decisions in proofs]
    kinds: list[str] = [clue.kind for clue in puzzle.clues]
    return {
        "z3_conflicts_mean": statistics.fmean(conflicts),
        "z3_conflicts_std": statistics.pstdev(conflicts),
        "z3_decisions_mean": statistics.fmean(decisions),
        "n_clues": float(len(puzzle.clues)),
        **{f"kind_{kind}": float(kinds.count(kind)) for kind in CLUE_KINDS},
    }


def method() -> dict[str, JsonValue]:
    """How `measure` got its numbers, recorded next to them."""
    return {"z3_version": z3.get_version_string(), "k": RUNS}


def _prove_unique(puzzle: LogicGridPuzzle, k: int) -> tuple[int, int]:
    """One uniqueness proof; returns its (conflicts, decisions)."""
    ctx = z3.Context()
    encoding = encode(puzzle.spec, puzzle.clues, ctx)
    clue_constraints = list(encoding.clues)
    random.Random(f"{puzzle.level}:{puzzle.seed}:{k}").shuffle(clue_constraints)
    solver = z3.Solver(ctx=ctx)
    solver.set(random_seed=k)
    solver.add(*encoding.structure)
    solver.add(*clue_constraints)
    if count_models(solver, list(encoding.variables.values()), cap=2) != 1:
        raise ValueError(
            f"puzzle (level {puzzle.level}, seed {puzzle.seed}) does not have exactly one solution"
        )
    # Statistics accumulate across both checks, so they cover the whole proof.
    stats = solver.statistics()
    return _stat(stats, "conflicts"), _stat(stats, "decisions")


def _stat(stats: Any, key: str) -> int:
    # Z3 leaves out counters that never fired. Statistics is not a mapping (iterating it yields
    # (key, value) pairs), so `key in stats` would always be False: keys() is required.
    return int(stats.get_key_value(key)) if key in stats.keys() else 0  # noqa: SIM118
