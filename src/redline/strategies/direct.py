"""Direct strategy: answer at once, in the puzzle's answer format and nothing else."""

from inspect_ai.model import ChatMessage

from redline.puzzles.base import PublicPuzzle
from redline.strategies.base import BudgetedModel, StrategyResult
from redline.strategies.common import PROMPTS, messages, prompts_for

INSTRUCTION = PROMPTS["logic_grid"].direct  # the logic-grid wording, unchanged since 1c


def build_messages(puzzle: PublicPuzzle) -> list[ChatMessage]:
    return messages(puzzle, prompts_for(puzzle).direct)


async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult:
    output = await model.generate(build_messages(puzzle), forcing_message=prompts_for(puzzle).force)
    return StrategyResult(final_text=output.completion)
