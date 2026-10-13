"""Natural-language rendering, ZebraLogic style.

`render` is the puzzle statement; `answer_format` describes the answer's shape from the spec
alone, so it reveals nothing. Instructions such as "reason step by step" belong to strategies.
"""

from typing import assert_never

from redline.puzzles.logic_grid.model import (
    AtClue,
    Clue,
    ImmediatelyLeftClue,
    LeftOfClue,
    LogicGridPuzzle,
    NextToClue,
    NotAtClue,
    NotSameHouseClue,
    Ref,
    SameHouseClue,
    Spec,
)
from redline.puzzles.logic_grid.vocab import ATTRIBUTES


def render(puzzle: LogicGridPuzzle) -> str:
    spec = puzzle.spec
    return "\n".join(
        [
            f"There are {spec.houses} houses in a row, numbered 1 to {spec.houses} from left to "
            "right. Each house is occupied by a different person. For each attribute below, "
            "every house has a different value.",
            "",
            *(
                f"- {ATTRIBUTES[name].label}: {', '.join(values)}"
                for name, values in spec.attributes
            ),
            "",
            "Clues:",
            *(f"{n}. {clue_sentence(clue)}" for n, clue in enumerate(puzzle.clues, start=1)),
        ]
    )


def answer_format(spec: Spec) -> str:
    """How to write the answer: keys and a template with "..." placeholders, never values."""
    quoted = [f'"{name}"' for name in spec.attribute_names]
    row = "{" + ", ".join(f'{name}: "..."' for name in quoted) + "}"
    template = ",\n".join(f'  "{house}": {row}' for house in range(1, spec.houses + 1))
    return (
        f'Give your answer as a JSON object with one key per house, "1" to "{spec.houses}". '
        f"Each value is an object with every attribute: {', '.join(quoted)}. "
        "Use each value exactly as it is listed in the puzzle. Template:\n"
        "{\n" + template + "\n}"
    )


def clue_sentence(clue: Clue) -> str:
    match clue:
        case AtClue():
            sentence = f"{_phrase(clue.ref)} lives in house {clue.house}."
        case NotAtClue():
            sentence = f"{_phrase(clue.ref)} does not live in house {clue.house}."
        case SameHouseClue():
            sentence = f"{_phrase(clue.a)} is {_phrase(clue.b)}."
        case NotSameHouseClue():
            sentence = f"{_phrase(clue.a)} is not {_phrase(clue.b)}."
        case LeftOfClue():
            sentence = f"{_phrase(clue.a)} lives somewhere to the left of {_phrase(clue.b)}."
        case ImmediatelyLeftClue():
            sentence = f"{_phrase(clue.a)} lives directly left of {_phrase(clue.b)}."
        case NextToClue():
            sentence = f"{_phrase(clue.a)} and {_phrase(clue.b)} live next to each other."
        case _:
            assert_never(clue)
    return sentence[0].upper() + sentence[1:]


def _phrase(ref: Ref) -> str:
    return ATTRIBUTES[ref.attribute].phrase(ref.value)
