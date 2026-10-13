"""Chain-of-thought strategy: reason step by step, then give the JSON grid at the end."""

from inspect_ai.model import ChatMessage

from redline.puzzles.base import PublicPuzzle
from redline.strategies.base import BudgetedModel, StrategyResult
from redline.strategies.common import FORCE_ANSWER, messages

INSTRUCTION = "Reason step by step, then give the final answer as JSON at the end."


def build_messages(puzzle: PublicPuzzle) -> list[ChatMessage]:
    return messages(puzzle, INSTRUCTION)


async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult:
    output = await model.generate(build_messages(puzzle), forcing_message=FORCE_ANSWER)
    return StrategyResult(final_text=output.completion)
