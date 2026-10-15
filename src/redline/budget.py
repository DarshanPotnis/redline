"""The per-sample generation budgets (the CLAUDE.md budget rule).

Reasoning strategies share TOTAL_BUDGET output tokens, reasoning included, with ANSWER_RESERVE held
back for one forced final answer. direct is the answer-only baseline: its whole budget is the
reserve, in a single call with no forcing, so "no reasoning" is enforced rather than requested.
The largest gold answer fits well inside the reserve: a 6x6 grid is 255-275 tokens as the answer
format shows it, 371 fully indented; a 7-person knights-and-knaves answer line is 47 (Qwen 3.5).
"""

from dataclasses import dataclass

TOTAL_BUDGET = 4096  # output tokens per sample, reasoning tokens included
ANSWER_RESERVE = 512  # held back from reasoning calls for one forced final answer


@dataclass(frozen=True)
class Budget:
    total: int  # output tokens for the whole sample
    reserve: int  # held back from the first call for a forced answer
    forcing: bool  # whether a call that stops on max_tokens is followed by a forcing call


REASONING = Budget(total=TOTAL_BUDGET, reserve=ANSWER_RESERVE, forcing=True)
ANSWER_ONLY = Budget(total=ANSWER_RESERVE, reserve=0, forcing=False)
