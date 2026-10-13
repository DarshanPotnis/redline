"""The per-sample generation budget every strategy shares (the CLAUDE.md budget rule)."""

TOTAL_BUDGET = 4096  # output tokens per sample, reasoning tokens included
ANSWER_RESERVE = 512  # held back from reasoning calls for one forced final answer
