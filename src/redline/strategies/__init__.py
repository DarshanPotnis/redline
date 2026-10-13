"""Strategies: `async def run(puzzle: PublicPuzzle, model: BudgetedModel) -> StrategyResult`.

A strategy never sees Inspect's task state or the answer key, and reaches the model only through
its BudgetedModel (contract v2, strategies/base.py). tasks/solver.py is the only bridge to
Inspect; an AST test keeps this package clear of answer-bearing names and of scoring.
"""

from collections.abc import Mapping

from redline.strategies import cot, direct
from redline.strategies.base import Strategy

STRATEGIES: Mapping[str, Strategy] = {
    "direct": direct.run,
    "cot": cot.run,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(sorted(STRATEGIES))}")
    return STRATEGIES[name]
