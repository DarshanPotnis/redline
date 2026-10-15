"""Strategies: `async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult`.

A strategy never sees Inspect's task state or the answer key, and reaches the model only through
its BudgetedModel (contract v2, strategies/base.py). tasks/solver.py is the only bridge to
Inspect; an AST test keeps this package clear of answer-bearing names and of scoring.
"""

from collections.abc import Mapping

from redline.budget import ANSWER_ONLY, REASONING, Budget
from redline.strategies import cot, direct, z3_translate
from redline.strategies.base import Strategy

STRATEGIES: Mapping[str, Strategy] = {
    "direct": direct.run,
    "cot": cot.run,
    "z3_translate": z3_translate.run,
}


# Each strategy's budget. direct is the answer-only baseline: its whole budget is the answer
# reserve, so "no reasoning" is enforced, not requested (1f.1).
BUDGETS: Mapping[str, Budget] = {
    "direct": ANSWER_ONLY,
    "cot": REASONING,
    "z3_translate": REASONING,
}


def budget_for(name: str) -> Budget:
    get_strategy(name)
    return BUDGETS[name]


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(sorted(STRATEGIES))}")
    return STRATEGIES[name]
