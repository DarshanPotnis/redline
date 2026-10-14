"""What every answer scorer reports about the run itself, whatever the puzzle: was the final
generation cut off, was that the context window's doing (a harness failure, excluded from
accuracy), and was the answer forced from the reserve. Judged on the final model call."""

from dataclasses import dataclass

from inspect_ai.solver import TaskState
from pydantic import JsonValue

from redline.strategies.base import CallRecord

# Each generator's answer scorer (logic grid, Reasoning Gym). Every one reports format_ok, solved,
# truncated, context_limited and forced_answer, plus the final-call metadata below.
ANSWER_SCORERS = ("grid_verifier", "rg_verifier")


def truncation(stop_reason: str, output_tokens: int | None, max_tokens: int) -> tuple[bool, bool]:
    """(truncated, context_limited) for one generation.

    Truncated: generation was cut off. Context-limited: it was cut off before spending the
    budget, so the context window stopped it. That is a harness failure, not a model failure.
    """
    truncated = stop_reason in ("max_tokens", "model_length")
    return truncated, truncated and output_tokens is not None and output_tokens < max_tokens


@dataclass(frozen=True)
class RunFlags:
    truncated: bool
    context_limited: bool
    forced_answer: bool
    calls: list[CallRecord]
    budget_used: int | None

    @property
    def final(self) -> CallRecord | None:
        return self.calls[-1] if self.calls else None

    def values(self) -> dict[str, float]:
        """The score values every answer scorer carries."""
        return {
            "truncated": float(self.truncated),
            "context_limited": float(self.context_limited),
            "forced_answer": float(self.forced_answer),
        }

    def metadata(self) -> dict[str, JsonValue]:
        final = self.final
        return {
            "n_calls": len(self.calls),
            "budget_used": self.budget_used,
            # the final call, whose stop decides truncated and context_limited
            "stop_reason": final.stop_reason if final else None,
            "input_tokens": final.input_tokens if final else None,
            "output_tokens": final.output_tokens if final else None,
            "reasoning_tokens": final.reasoning_tokens if final else None,
            "max_tokens": final.max_tokens if final else None,
        }

    def explanation_suffix(self) -> str:
        final = self.final
        if not (final and self.context_limited):
            return ""
        return (
            f"; harness failure: the context window stopped generation at "
            f"{final.output_tokens} of {final.max_tokens} tokens"
        )


def run_flags(state: TaskState) -> RunFlags:
    run = state.metadata.get("strategy_run", {})
    calls = [CallRecord.model_validate(call) for call in run.get("calls", [])]
    final = calls[-1] if calls else None
    truncated, context_limited = (
        truncation(final.stop_reason, final.output_tokens, final.max_tokens)
        if final
        else (False, False)
    )
    return RunFlags(
        truncated=truncated,
        context_limited=context_limited,
        forced_answer=any(call.forcing for call in calls),
        calls=calls,
        budget_used=run.get("budget", {}).get("used"),
    )
