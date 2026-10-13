"""Chain-of-thought strategy: reason step by step, then give the JSON grid at the end."""

from inspect_ai.model import ChatMessage

from redline.puzzles.base import PublicPuzzle
from redline.strategies.common import messages

INSTRUCTION = "Reason step by step, then give the final answer as JSON at the end."


def build_messages(puzzle: PublicPuzzle) -> list[ChatMessage]:
    return messages(puzzle, INSTRUCTION)
