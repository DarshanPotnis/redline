"""Direct strategy: answer with the JSON grid and nothing else."""

from inspect_ai.model import ChatMessage

from redline.puzzles.base import PublicPuzzle
from redline.strategies.common import messages

INSTRUCTION = "Respond with only the JSON."


def build_messages(puzzle: PublicPuzzle) -> list[ChatMessage]:
    return messages(puzzle, INSTRUCTION)
