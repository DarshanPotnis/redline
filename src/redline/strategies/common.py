"""What every strategy shares: the rules as a system message, and the user-message layout."""

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

from redline.puzzles.base import PublicPuzzle

RULES = (
    "You solve logic-grid puzzles. The houses stand in a row, numbered from left to right "
    "starting at 1. Each house has exactly one value for every attribute, and each value "
    'belongs to exactly one house. "Left" means a lower house number, "directly left" means '
    'the next lower number, and "next to" means adjacent houses. Every clue is true, and '
    "together the clues determine exactly one answer."
)


def messages(puzzle: PublicPuzzle, instruction: str) -> list[ChatMessage]:
    """The rules, then one user message: puzzle text, answer format, instruction."""
    return [
        ChatMessageSystem(content=RULES),
        ChatMessageUser(content=f"{puzzle.text}\n\n{puzzle.answer_format}\n\n{instruction}"),
    ]
