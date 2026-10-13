"""What the grid-answering strategies share: the rules, the user-message layout, and the
forcing message for budget forcing."""

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


def messages(puzzle: PublicPuzzle, instruction: str) -> list[ChatMessage]:
    """The rules, then one user message: puzzle text, answer format, instruction."""
    return [
        ChatMessageSystem(content=RULES),
        ChatMessageUser(content=f"{puzzle.text}\n\n{puzzle.answer_format}\n\n{instruction}"),
    ]
