"""The Reasoning Gym task end to end on mockllm: scored by our extraction and RG's checker,
budget-forced like the logic grid, and leak-free."""

import asyncio
from dataclasses import dataclass

import inspect_ai
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    StopReason,
    get_model,
)

from redline.puzzles.base import PublicPuzzle
from redline.puzzles.reasoning_gym import KNIGHTS_KNAVES
from redline.strategies import get_strategy
from redline.strategies.base import BudgetedModel, CallRecord
from redline.strategies.common import PROMPTS
from redline.tasks.reasoning_gym import reasoning_gym, rg_samples

FORCE = PROMPTS["reasoning_gym"].force
SAMPLES = rg_samples(KNIGHTS_KNAVES, [1, 2], 2)
CORRECT, FLIPPED, NO_LINE, LAST_WINS = (str(sample.id) for sample in SAMPLES)


def flip_first(sample: Sample) -> str:
    """The answer with the first person's role swapped."""
    terms = (sample.metadata or {})["hidden"]["metadata"]["knight_knave_terms"]
    first, *rest = str(sample.target).rstrip(".").split(", ")
    hero, villain = terms["a_knight"], terms["a_knave"]
    flipped = first.replace(hero, "@").replace(villain, hero).replace("@", villain)
    return ", ".join([flipped, *rest]) + "."


def reply_for(sample: Sample) -> str:
    answer = str(sample.target)
    return {
        CORRECT: f"Answer: {answer}",
        FLIPPED: f"Answer: {flip_first(sample)}",
        NO_LINE: answer,  # right, but not on an Answer: line
        LAST_WINS: f"Answer: {flip_first(sample)}\nWait, let me recheck.\n**Answer:** {answer}",
    }[str(sample.id)]


def output(text: str, stop_reason: StopReason, tokens: int) -> ModelOutput:
    reply = ModelOutput.from_content("mockllm/model", text, stop_reason=stop_reason)
    usage = ModelUsage(input_tokens=200, output_tokens=tokens, total_tokens=200 + tokens)
    return reply.model_copy(update={"usage": usage})


@dataclass
class Run:
    strategy: str
    log: EvalLog
    inputs: list[str]  # every message text the model received


@pytest.fixture(scope="module", params=["direct", "cot"])
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Run:
    strategy: str = request.param
    inputs: list[str] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        inputs.extend(message.text for message in messages)
        first_user = next(message.text for message in messages if message.role == "user")
        [sample] = [s for s in SAMPLES if str(s.input) in first_user]
        if strategy == "cot" and str(sample.id) == CORRECT and messages[-1].text != FORCE:
            assert config.max_tokens is not None
            return output("Let me consider each person...", "max_tokens", config.max_tokens)
        return output(reply_for(sample), "stop", 40)

    [log] = inspect_ai.eval(
        reasoning_gym,
        task_args={"task": "knights_knaves", "levels": "1,2", "n_seeds": 2, "strategy": strategy},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path_factory.mktemp(f"rg-{strategy}")),
        display="none",
    )
    assert log.status == "success"
    return Run(strategy, read_eval_log(log.location, resolve_attachments=True), inputs)


def scores(run: Run) -> dict[str, dict[str, float]]:
    assert run.log.samples is not None
    return {str(s.id): s.scores["rg_verifier"].value for s in run.log.samples if s.scores}


def test_samples_are_scored_from_the_last_answer_line(run: Run) -> None:
    by_id = scores(run)
    assert by_id[CORRECT]["solved"] == 1.0 and by_id[CORRECT]["rg_score"] == 1.0
    assert by_id[FLIPPED]["solved"] == 0.0 and 0 < by_id[FLIPPED]["rg_score"] < 1
    assert by_id[NO_LINE]["format_ok"] == 0.0 and by_id[NO_LINE]["rg_score"] == 0.0
    assert by_id[LAST_WINS]["solved"] == 1.0  # an earlier, wrong Answer: line is ignored
    assert all(value["context_limited"] == 0.0 for value in by_id.values())


def test_a_cut_off_cot_answer_is_forced(run: Run) -> None:
    forced = {sample_id: value["forced_answer"] for sample_id, value in scores(run).items()}
    expected = {CORRECT: 1.0} if run.strategy == "cot" else {}
    assert {k: v for k, v in forced.items() if v} == expected


def test_the_metrics_follow_the_logic_grid_conventions(run: Run) -> None:
    assert run.log.results is not None
    names = {score.name: set(score.metrics) for score in run.log.results.scores}
    assert names["solved"] == {"accuracy", "stderr"}
    assert names["rg_score"] == {"mean", "stderr"}
    assert set(names) == {
        "solved",
        "rg_score",
        "format_ok",
        "truncated",
        "context_limited",
        "forced_answer",
        "direct_overrun",
    }


def replay(strategy: str, puzzle: PublicPuzzle, calls: list[CallRecord]) -> list[CallRecord]:
    """Re-run the strategy on the public puzzle alone, fed the recorded outputs in order."""
    replies = iter(calls)

    async def generate(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        call = next(replies)
        return output(call.text, call.stop_reason, call.output_tokens or 0)

    model = BudgetedModel(generate)
    asyncio.run(get_strategy(strategy)(puzzle, model))
    return model.calls


def test_every_call_comes_from_the_public_puzzle_and_no_answer_reaches_the_model(run: Run) -> None:
    # The leak test: re-running the strategy on the public puzzle alone, fed the recorded
    # outputs, makes the same calls; and no answer key appears in anything the model received.
    assert run.log.samples is not None
    for sample in run.log.samples:
        calls = [CallRecord.model_validate(c) for c in sample.metadata["strategy_run"]["calls"]]
        events = [event for event in sample.events if isinstance(event, ModelEvent)]
        assert [[(m.role, m.text) for m in event.input] for event in events] == [
            call.messages for call in calls
        ]
        public = PublicPuzzle.model_validate(sample.metadata["public"])
        replayed = replay(run.strategy, public, calls)
        assert [call.messages for call in replayed] == [call.messages for call in calls]
    received = "\n".join(run.inputs)
    assert all(str(sample.target) not in received for sample in SAMPLES)


def test_z3_translate_is_refused() -> None:
    with pytest.raises(ValueError, match="translates logic-grid clues"):
        reasoning_gym(task="knights_knaves", strategy="z3_translate")
