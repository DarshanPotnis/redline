"""The one bridge between Inspect and a strategy, and the only code that touches TaskState.

It hands the strategy the public puzzle and a BudgetedModel over the model under evaluation, then
stores what the strategy produced for the scorer: the final text as the output, and the
artifacts, call records and budget in `state.metadata["strategy_run"]`.
"""

from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from redline.puzzles.base import PublicPuzzle
from redline.strategies import get_strategy
from redline.strategies.base import BudgetedModel


@solver
def strategy_solver(strategy: str) -> Solver:
    run = get_strategy(strategy)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        budgeted = BudgetedModel(lambda messages, config: model.generate(messages, config=config))
        result = await run(PublicPuzzle.model_validate(state.metadata["public"]), budgeted)
        final = budgeted.calls[-1] if budgeted.calls else None
        state.output = ModelOutput.from_content(
            model=str(model),
            content=result.final_text,
            stop_reason=final.stop_reason if final else "stop",
        )
        if budgeted.conversation:
            state.messages = budgeted.conversation
        state.metadata["strategy_run"] = {
            "artifacts": result.artifacts,
            "calls": [call.model_dump(mode="json") for call in budgeted.calls],
            "budget": {
                "total": budgeted.total,
                "reserve": budgeted.reserve,
                "used": budgeted.used,
            },
        }
        return state

    return solve
