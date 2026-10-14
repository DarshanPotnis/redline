"""Misread diagnosis for translating strategies: the model's clues against the stated clues.

Each well-formed translated clue is exact (canonically equal to a stated clue), inferred (true on
the solution but not stated, like "the rabbit owner" rewritten as the person's name, or "directly
left" weakened to "left of") or wrong (false on the solution: a misread). Each stated clue is
captured (an exact translation exists), entailed (the translated set implies it, checked with Z3)
or missed. Malformed items, dropped by the strategy, are counted too. An exact translation has
nothing wrong, missed or malformed; joined with a direct answer by sample id (1g), it separates
misreading a puzzle from misreasoning about it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from pydantic import TypeAdapter

from redline.puzzles.logic_grid.clues import holds
from redline.puzzles.logic_grid.model import (
    Clue,
    NextToClue,
    NotSameHouseClue,
    SameHouseClue,
    Solution,
    Spec,
)
from redline.puzzles.logic_grid.solve_z3 import count_solutions_z3, entails

STATUSES = ("unique", "ambiguous", "unsat", "parse_error")
_CLUES = TypeAdapter(list[Clue])


@dataclass(frozen=True)
class Diagnosis:
    # translated clues, by 1-based position in the translation
    exact: tuple[int, ...]
    inferred: tuple[int, ...]
    wrong: tuple[int, ...]
    # stated clues, by their 1-based number in the puzzle
    captured: tuple[int, ...]
    entailed: tuple[int, ...]
    missed: tuple[int, ...]


def canonical(clue: Clue) -> Clue:
    """Symmetric kinds with their refs sorted, so either order counts as the same clue."""
    if isinstance(clue, SameHouseClue | NotSameHouseClue | NextToClue):
        a, b = sorted((clue.a, clue.b), key=lambda ref: (ref.attribute, ref.value))
        return clue.model_copy(update={"a": a, "b": b})
    return clue


def diagnose(
    spec: Spec, translated: Sequence[Clue], stated: Sequence[Clue], solution: Solution
) -> Diagnosis:
    stated_keys = {_key(clue) for clue in stated}
    written = {_key(clue) for clue in translated}
    exact = tuple(n for n, clue in enumerate(translated, 1) if _key(clue) in stated_keys)
    inferred = tuple(
        n
        for n, clue in enumerate(translated, 1)
        if _key(clue) not in stated_keys and holds(clue, solution)
    )
    wrong = tuple(n for n, clue in enumerate(translated, 1) if not holds(clue, solution))
    # A contradictory translation would entail every clue, so entailment needs a satisfiable
    # one. With nothing wrong, the true solution satisfies it; otherwise Z3 decides.
    consistent = bool(translated) and (not wrong or count_solutions_z3(spec, translated, 1) == 1)
    captured: list[int] = []
    entailed: list[int] = []
    missed: list[int] = []
    for n, clue in enumerate(stated, 1):
        if _key(clue) in written:
            captured.append(n)
        elif consistent and entails(spec, translated, clue):
            entailed.append(n)
        else:
            missed.append(n)
    return Diagnosis(exact, inferred, wrong, tuple(captured), tuple(entailed), tuple(missed))


def _key(clue: Clue) -> str:
    return canonical(clue).model_dump_json()


@scorer(
    metrics={
        "translation_exact": [mean()],
        **{f"status_{status}": [mean()] for status in STATUSES},
    }
)
def translation_verifier() -> Scorer:
    """Grades the translation in `strategy_run.artifacts`, not the final grid."""

    async def score(state: TaskState, target: Target) -> Score:
        artifacts = state.metadata["strategy_run"]["artifacts"]
        hidden = state.metadata["hidden"]
        status = artifacts["translation_status"]
        translated = _CLUES.validate_python(artifacts["translated_clues"] or [])
        malformed = [entry["item"] for entry in artifacts.get("malformed", [])]
        stated = _CLUES.validate_python(hidden["clues"])
        d = diagnose(
            Spec.model_validate(hidden["spec"]),
            translated,
            stated,
            Solution.model_validate_json(target.text),
        )
        exact = not d.wrong and not d.missed and not malformed
        return Score(
            value={
                "translation_exact": float(exact),
                **{f"status_{name}": float(status == name) for name in STATUSES},
            },
            explanation=(
                f"{status}: of {len(translated)} translated clues {len(d.exact)} exact, "
                f"{len(d.inferred)} inferred, {len(d.wrong)} wrong ({len(malformed)} malformed "
                f"items dropped); of {len(stated)} stated clues {len(d.captured)} captured, "
                f"{len(d.entailed)} entailed, {len(d.missed)} missed"
            ),
            metadata={
                "translation_status": status,
                "n_exact": len(d.exact),
                "n_inferred": len(d.inferred),
                "n_wrong": len(d.wrong),
                "n_malformed": len(malformed),
                "n_captured": len(d.captured),
                "n_entailed": len(d.entailed),
                "n_missed": len(d.missed),
                "inferred_clues": list(d.inferred),
                "wrong_clues": list(d.wrong),
                "malformed_items": malformed,
                "entailed_clues": list(d.entailed),
                "missed_clues": list(d.missed),
            },
        )

    return score
