"""What the answering strategies share: per-family rules and wording, the user-message layout,
and the forcing message for budget forcing.

A puzzle family (the generator name before "/", e.g. "logic_grid", "reasoning_gym") brings its
own system rules and its own way of naming the answer: the logic grid answers in JSON, Reasoning
Gym tasks in a final "Answer:" line. The logic-grid strings are unchanged since 1c.
"""

from dataclasses import dataclass

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

from redline.puzzles.base import PublicPuzzle

RULES = (
    "You solve logic-grid puzzles. The houses stand in a row, numbered from left to right "
    "starting at 1. Each house has exactly one value for every attribute, and each value "
    'belongs to exactly one house. "Left" means a lower house number, "directly left" means '
    'the next lower number, and "next to" means adjacent houses. Every clue is true, and '
    "together the clues determine exactly one answer."
)
FORCE_ANSWER = (
    "You have run out of reasoning budget. Output only your final answer as JSON now, based on "
    "your reasoning so far."
)


@dataclass(frozen=True)
class Prompts:
    rules: str  # the system message
    direct: str  # the direct strategy's instruction
    cot: str  # the cot strategy's instruction
    force: str  # the forcing message, when reasoning hits its cap


PROMPTS: dict[str, Prompts] = {
    "logic_grid": Prompts(
        rules=RULES,
        direct="Respond with only the JSON.",
        cot="Reason step by step, then give the final answer as JSON at the end.",
        force=FORCE_ANSWER,
    ),
    "reasoning_gym": Prompts(
        rules=(
            "You solve logic puzzles stated in plain language. The question gives every rule you "
            "need, and it has exactly one correct answer."
        ),
        direct='Respond with only the final "Answer:" line.',
        cot='Reason step by step, then end with the final "Answer:" line.',
        force=(
            'You have run out of reasoning budget. Output only your final "Answer:" line now, '
            "based on your reasoning so far."
        ),
    ),
}


def prompts_for(puzzle: PublicPuzzle) -> Prompts:
    if puzzle.family not in PROMPTS:
        raise ValueError(
            f"no prompts for puzzle family {puzzle.family!r}; known: {', '.join(PROMPTS)}"
        )
    return PROMPTS[puzzle.family]


def messages(puzzle: PublicPuzzle, instruction: str) -> list[ChatMessage]:
    """The family's rules, then one user message: puzzle text, answer format, instruction."""
    return [
        ChatMessageSystem(content=prompts_for(puzzle).rules),
        ChatMessageUser(content=f"{puzzle.text}\n\n{puzzle.answer_format}\n\n{instruction}"),
    ]
