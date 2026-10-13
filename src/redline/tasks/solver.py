"""The one bridge between Inspect and a strategy: it hands the strategy only the public puzzle.

This is the only code that touches TaskState.
"""

from inspect_ai.solver import Generate, Solver, TaskState, solver

from redline.puzzles.base import PublicPuzzle
from redline.strategies import get_strategy


@solver
def strategy_solver(strategy: str) -> Solver:
    """Replace the conversation with the strategy's messages for this sample, then generate."""
    build_messages = get_strategy(strategy)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = build_messages(PublicPuzzle.model_validate(state.metadata["public"]))
        return await generate(state)

    return solve
