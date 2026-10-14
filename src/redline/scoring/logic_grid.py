"""Logic-grid verifier: parse the model's grid, grade it, and name the clues it breaks.

Uniqueness makes the two graders agree: a valid grid is solved exactly when it breaks no clue.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState
from pydantic import TypeAdapter

from redline.jsonscan import last_json
from redline.puzzles.logic_grid.clues import holds
from redline.puzzles.logic_grid.model import Clue, Solution, Spec
from redline.puzzles.logic_grid.vocab import ATTRIBUTES
from redline.strategies.base import CallRecord

# house -> attribute -> value, in the spec's own spelling
type Grid = dict[int, dict[str, str]]

ParseReason = Literal[
    "no_json",
    "unknown_house",
    "duplicate_house",
    "missing_house",
    "house_not_object",
    "unknown_attribute",
    "duplicate_attribute",
    "missing_attribute",
    "unknown_value",
]


@dataclass(frozen=True)
class ParseError:
    reason: ParseReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


@dataclass(frozen=True)
class Grade:
    format_ok: bool
    valid_grid: bool  # every attribute is a permutation of its values
    solved: bool
    cell_accuracy: float  # 0 when the format fails
    broken_clues: tuple[int, ...] | None  # 1-based, as numbered in the puzzle; valid grids only
    parse_error: str | None


class _Object(list[tuple[str, Any]]):
    """A decoded JSON object, kept as key/value pairs so duplicate keys stay visible."""


_DECODER = json.JSONDecoder(object_pairs_hook=_Object)
_HOUSE_KEY = re.compile(r"(?:house\s*)?(\d+)", re.IGNORECASE)
_CLUES = TypeAdapter(list[Clue])


def parse_grid(response: str, spec: Spec) -> Grid | ParseError:
    """The last JSON object in `response`, checked against the spec.

    Lenient on form: fences and prose around the JSON, house keys "1" or "House 1", attribute
    names (or their labels, like "Favorite color") and values in any case, with spaces trimmed.
    Strict on content: every house, every attribute, every value from that attribute's list.
    """
    pairs: _Object | None = last_json(response, "{", _DECODER)
    if pairs is None:
        return ParseError("no_json", "no JSON object in the response")
    houses: dict[int, _Object] = {}
    for key, value in pairs:
        house = _house_number(key)
        if house is None or not 1 <= house <= spec.houses:
            return ParseError("unknown_house", f"key {key!r}")
        if house in houses:
            return ParseError("duplicate_house", f"house {house}")
        if not isinstance(value, _Object):
            return ParseError("house_not_object", f"house {house}")
        houses[house] = value
    missing = [house for house in range(1, spec.houses + 1) if house not in houses]
    if missing:
        return ParseError("missing_house", f"houses {missing}")
    grid: Grid = {}
    for house in range(1, spec.houses + 1):
        cells = _parse_house(houses[house], spec, house)
        if isinstance(cells, ParseError):
            return cells
        grid[house] = cells
    return grid


def grade(parsed: Grid | ParseError, solution: Solution, clues: Sequence[Clue]) -> Grade:
    if isinstance(parsed, ParseError):
        return Grade(False, False, False, 0.0, None, str(parsed))
    houses = sorted(parsed)
    expected = {name: tuple(column) for name, column in solution.root.items()}
    columns = {name: tuple(parsed[house][name] for house in houses) for name in expected}
    correct = sum(
        given == wanted
        for name in expected
        for given, wanted in zip(columns[name], expected[name], strict=True)
    )
    valid = all(len(set(column)) == len(column) for column in columns.values())
    broken = None
    if valid:
        as_solution = Solution(columns)
        broken = tuple(n for n, clue in enumerate(clues, start=1) if not holds(clue, as_solution))
    cell_accuracy = correct / (len(houses) * len(expected))
    return Grade(True, valid, columns == expected, cell_accuracy, broken, None)


def truncation(stop_reason: str, output_tokens: int | None, max_tokens: int) -> tuple[bool, bool]:
    """(truncated, context_limited) for one generation.

    Truncated: generation was cut off. Context-limited: it was cut off before spending the
    budget, so the context window stopped it. That is a harness failure, not a model failure.
    """
    truncated = stop_reason in ("max_tokens", "model_length")
    return truncated, truncated and output_tokens is not None and output_tokens < max_tokens


@scorer(
    metrics={
        "solved": [accuracy(), stderr()],
        "cell_accuracy": [mean(), stderr()],
        "format_ok": [mean()],
        "valid_grid": [mean()],
        "truncated": [mean()],
        "context_limited": [mean()],
        "forced_answer": [mean()],
    }
)
def grid_verifier() -> Scorer:
    """Grades the strategy's final text; truncation is judged on the final model call."""

    async def score(state: TaskState, target: Target) -> Score:
        hidden = state.metadata["hidden"]
        spec = Spec.model_validate(hidden["spec"])
        parsed = parse_grid(state.output.completion, spec)
        solution = Solution.model_validate_json(target.text)
        result = grade(parsed, solution, _CLUES.validate_python(hidden["clues"]))
        broken = None if result.broken_clues is None else list(result.broken_clues)
        run = state.metadata.get("strategy_run", {})
        calls = [CallRecord.model_validate(call) for call in run.get("calls", [])]
        final = calls[-1] if calls else None
        truncated, context_limited = (
            truncation(final.stop_reason, final.output_tokens, final.max_tokens)
            if final
            else (False, False)
        )
        explanation = _explain(result)
        if final and context_limited:
            explanation += (
                f"; harness failure: the context window stopped generation at "
                f"{final.output_tokens} of {final.max_tokens} tokens"
            )
        return Score(
            value={
                "format_ok": float(result.format_ok),
                "valid_grid": float(result.valid_grid),
                "solved": float(result.solved),
                "cell_accuracy": result.cell_accuracy,
                "truncated": float(truncated),
                "context_limited": float(context_limited),
                "forced_answer": float(any(call.forcing for call in calls)),
            },
            answer=None if isinstance(parsed, ParseError) else json.dumps(parsed),
            explanation=explanation,
            metadata={
                "broken_clues": broken,
                "n_broken": None if broken is None else len(broken),
                "parse_error": result.parse_error,
                "n_calls": len(calls),
                "budget_used": run.get("budget", {}).get("used"),
                # the final call, whose stop decides truncated and context_limited
                "stop_reason": final.stop_reason if final else None,
                "input_tokens": final.input_tokens if final else None,
                "output_tokens": final.output_tokens if final else None,
                "reasoning_tokens": final.reasoning_tokens if final else None,
                "max_tokens": final.max_tokens if final else None,
            },
        )

    return score


def _house_number(key: str) -> int | None:
    match = _HOUSE_KEY.fullmatch(key.strip())
    return int(match.group(1)) if match else None


def _parse_house(pairs: _Object, spec: Spec, house: int) -> dict[str, str] | ParseError:
    cells: dict[str, str] = {}
    for key, value in pairs:
        name = _attribute_name(key, spec)
        if name is None:
            return ParseError("unknown_attribute", f"house {house}: {key!r}")
        if name in cells:
            return ParseError("duplicate_attribute", f"house {house}: {name!r}")
        canonical = _canonical_value(value, spec.values_of(name))
        if canonical is None:
            return ParseError("unknown_value", f"house {house}: {name}={value!r}")
        cells[name] = canonical
    missing = [name for name in spec.attribute_names if name not in cells]
    if missing:
        return ParseError("missing_attribute", f"house {house}: {missing}")
    return {name: cells[name] for name in spec.attribute_names}


def _attribute_name(key: str, spec: Spec) -> str | None:
    wanted = _fold(key)
    for name in spec.attribute_names:
        label = ATTRIBUTES[name].label if name in ATTRIBUTES else name
        if wanted in (_fold(name), _fold(label)):
            return name
    return None


def _canonical_value(value: object, allowed: Sequence[str]) -> str | None:
    if not isinstance(value, str):
        return None
    wanted = _fold(value)
    return next((option for option in allowed if _fold(option) == wanted), None)


def _fold(text: str) -> str:
    """Case-insensitive, with surrounding and repeated spaces ignored."""
    return " ".join(text.split()).casefold()


def _explain(result: Grade) -> str:
    if result.parse_error is not None:
        return f"format error ({result.parse_error})"
    if not result.valid_grid:
        return "not a valid grid: an attribute repeats a value"
    if result.solved:
        return "solved"
    broken = ", ".join(str(n) for n in result.broken_clues or ())
    return f"wrong: breaks clue {broken}"
