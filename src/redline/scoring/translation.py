"""Misread diagnosis for translating strategies: the model's clues against the true clues.

A translated clue that is false on the true solution is a misread (a flipped direction, say). A
true clue with no equal translation was missed. An exact translation has neither; joined with a
direct answer by sample id (1e), it separates misreading a puzzle from misreasoning about it.
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
)

STATUSES = ("unique", "ambiguous", "unsat", "parse_error")
_CLUES = TypeAdapter(list[Clue])


@dataclass(frozen=True)
class Diagnosis:
    wrong: tuple[int, ...]  # 1-based positions in the translation of clues false on the solution
    missed: tuple[int, ...]  # 1-based numbers of the true clues that no translated clue equals


def canonical(clue: Clue) -> Clue:
    """Symmetric kinds with their refs sorted, so either order counts as the same clue."""
    if isinstance(clue, SameHouseClue | NotSameHouseClue | NextToClue):
        a, b = sorted((clue.a, clue.b), key=lambda ref: (ref.attribute, ref.value))
        return clue.model_copy(update={"a": a, "b": b})
    return clue


def diagnose(translated: Sequence[Clue], true: Sequence[Clue], solution: Solution) -> Diagnosis:
    written = {canonical(clue).model_dump_json() for clue in translated}
    return Diagnosis(
        wrong=tuple(n for n, clue in enumerate(translated, 1) if not holds(clue, solution)),
        missed=tuple(
            n for n, clue in enumerate(true, 1) if canonical(clue).model_dump_json() not in written
        ),
    )


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
        status = artifacts["translation_status"]
        translated = _CLUES.validate_python(artifacts["translated_clues"] or [])
        true = _CLUES.validate_python(state.metadata["hidden"]["clues"])
        diagnosis = diagnose(translated, true, Solution.model_validate_json(target.text))
        exact = status != "parse_error" and not diagnosis.wrong and not diagnosis.missed
        return Score(
            value={
                "translation_exact": float(exact),
                **{f"status_{name}": float(status == name) for name in STATUSES},
            },
            explanation=(
                f"{status}: {len(diagnosis.wrong)} wrong, {len(diagnosis.missed)} missed of "
                f"{len(true)} clues"
            ),
            metadata={
                "translation_status": status,
                "n_wrong": len(diagnosis.wrong),
                "n_missed": len(diagnosis.missed),
                "wrong_clues": list(diagnosis.wrong),
                "missed_clues": list(diagnosis.missed),
            },
        )

    return score
