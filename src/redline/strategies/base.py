"""Strategy contract v2: a strategy is `async def run(puzzle, model) -> StrategyResult`.

`BudgetedModel` is a strategy's only way to reach the model. It holds a sample to its budget
(redline/budget.py), records every call, and does budget forcing as in s1: a call may spend at
most what remains minus the reserve; if it stops on max_tokens, exactly one forcing call follows,
with the rest of the budget and a strategy-supplied message asking for the final answer.
Reasoning strategies get TOTAL_BUDGET with a 512-token reserve. direct gets only the reserve, in
one call with forcing off: a reply cut off there is an overrun, not something to force.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, ModelOutput, StopReason
from pydantic import BaseModel, ConfigDict, JsonValue

from redline.budget import ANSWER_RESERVE, TOTAL_BUDGET
from redline.puzzles.base import PublicPuzzle

type GenerateFn = Callable[[list[ChatMessage], GenerateConfig], Awaitable[ModelOutput]]


@dataclass(frozen=True)
class StrategyResult:
    final_text: str  # what the verifier grades
    artifacts: dict[str, JsonValue] = field(default_factory=dict)  # for the scorer


class BudgetExhausted(RuntimeError):
    """A strategy asked for a call that the remaining budget cannot cover."""


class CallRecord(BaseModel):
    """One model call as the strategy made it."""

    model_config = ConfigDict(frozen=True)

    messages: list[tuple[str, str]]  # (role, text) of every input message
    max_tokens: int  # this call's cap
    forcing: bool
    stop_reason: StopReason
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    text: str  # the completion


class BudgetedModel:
    def __init__(
        self,
        generate: GenerateFn,
        total: int = TOTAL_BUDGET,
        reserve: int = ANSWER_RESERVE,
        forcing: bool = True,
    ) -> None:
        self._generate = generate
        self.total = total
        self.reserve = reserve
        self.forcing = forcing
        self.used = 0
        self.calls: list[CallRecord] = []
        self.conversation: list[ChatMessage] = []  # the last call's input plus its output

    @property
    def remaining(self) -> int:
        return max(self.total - self.used, 0)

    async def generate(
        self, messages: Sequence[ChatMessage], *, forcing_message: str
    ) -> ModelOutput:
        """One call, plus one forcing call if it stops on max_tokens and forcing is on."""
        if self.remaining <= self.reserve:
            raise BudgetExhausted(f"{self.remaining} tokens left, all held for a forced answer")
        output = await self._call(list(messages), self.remaining - self.reserve, forcing=False)
        if not self.forcing or output.stop_reason != "max_tokens" or self.remaining == 0:
            return output
        forced = [*messages, output.message, ChatMessageUser(content=forcing_message)]
        return await self._call(forced, self.remaining, forcing=True)

    async def _call(
        self, messages: list[ChatMessage], max_tokens: int, *, forcing: bool
    ) -> ModelOutput:
        output = await self._generate(messages, GenerateConfig(max_tokens=max_tokens))
        usage = output.usage
        # A call that reports no usage is charged its whole cap, so the budget can never overrun.
        self.used += usage.output_tokens if usage else max_tokens
        self.calls.append(
            CallRecord(
                messages=[(message.role, message.text) for message in messages],
                max_tokens=max_tokens,
                forcing=forcing,
                stop_reason=output.stop_reason,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                reasoning_tokens=usage.reasoning_tokens if usage else None,
                text=output.completion,
            )
        )
        self.conversation = [*messages, output.message]
        return output


type Strategy = Callable[[PublicPuzzle, BudgetedModel], Awaitable[StrategyResult]]
