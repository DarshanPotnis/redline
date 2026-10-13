"""Strategies: pure functions from a PublicPuzzle to chat messages.

A strategy never sees Inspect's task state or the answer key. tasks/solver.py is the only bridge
to Inspect, and an AST test keeps this package clear of answer-bearing names and of scoring.
"""

from collections.abc import Callable, Mapping

from inspect_ai.model import ChatMessage

from redline.puzzles.base import PublicPuzzle
from redline.strategies import cot, direct

type Strategy = Callable[[PublicPuzzle], list[ChatMessage]]

STRATEGIES: Mapping[str, Strategy] = {
    "direct": direct.build_messages,
    "cot": cot.build_messages,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(sorted(STRATEGIES))}")
    return STRATEGIES[name]
