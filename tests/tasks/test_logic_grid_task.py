"""The logic-grid task end to end on mockllm: solvable, leak-free, budget-matched, reproducible."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

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

from conftest import as_translation
from redline.budget import ANSWER_RESERVE, TOTAL_BUDGET
from redline.puzzles.base import PublicPuzzle
from redline.puzzles.logic_grid.model import GENERATOR_VERSION
from redline.runtime.profiles import profile_for
from redline.strategies import STRATEGIES, get_strategy
from redline.strategies.base import BudgetedModel, CallRecord
from redline.strategies.common import FORCE_ANSWER
from redline.tasks.logic_grid import logic_grid, logic_grid_samples, parse_levels

MOCK = profile_for("mockllm/model")
FIRST_CAP = TOTAL_BUDGET - ANSWER_RESERVE


def public(sample: Sample) -> PublicPuzzle:
    return PublicPuzzle.model_validate((sample.metadata or {})["public"])


def answer(target: str) -> str:
    columns = json.loads(target)
    houses = len(next(iter(columns.values())))
    return json.dumps(
        {
            str(h): {name: values[h - 1] for name, values in columns.items()}
            for h in range(1, houses + 1)
        }
    )


def puzzle_of(messages: list[ChatMessage], samples: list[Sample]) -> Sample:
    """The sample whose puzzle text is in the conversation's first user message."""
    first_user = next(message.text for message in messages if message.role == "user")
    [sample] = [sample for sample in samples if str(sample.input) in first_user]
    return sample


def output(text: str, stop_reason: StopReason, tokens: int) -> ModelOutput:
    usage = ModelUsage(input_tokens=300, output_tokens=tokens, total_tokens=300 + tokens)
    reply = ModelOutput.from_content("mockllm/model", text, stop_reason=stop_reason)
    return reply.model_copy(update={"usage": usage})


@dataclass
class Run:
    strategy: str
    samples: dict[str, Sample]
    configs: list[GenerateConfig]  # what each model call was given
    log: EvalLog  # read back from disk


@pytest.fixture(scope="module", params=sorted(STRATEGIES))
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One eval per strategy: a mock model that answers every puzzle from its own target."""
    strategy: str = request.param
    samples = logic_grid_samples([1, 2, 3], 2)
    configs: list[GenerateConfig] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        configs.append(config)
        sample = puzzle_of(messages, samples)
        if strategy == "z3_translate":  # a perfect translator: the true clues, as a JSON list
            clues = as_translation((sample.metadata or {})["hidden"]["clues"])
            return output(f"Here are the clues:\n```json\n{clues}\n```", "stop", 120)
        grid = answer(sample.target)
        reply = (
            grid
            if strategy == "direct"
            else f"Working through the clues...\n\n```json\n{grid}\n```"
        )
        return output(reply, "stop", 120)

    # The task is built inside eval, as in a real run, so it resolves the mock model's profile.
    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "1,2,3", "n_seeds": 2, "strategy": strategy},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path_factory.mktemp(strategy)),
        display="none",
    )
    assert log.status == "success"
    # Logs store long message text as attachment:// references; resolve them to compare text.
    log = read_eval_log(log.location, resolve_attachments=True)
    return Run(strategy, {str(sample.id): sample for sample in samples}, configs, log)


def recorded_calls(sample_metadata: dict[str, object]) -> list[CallRecord]:
    strategy_run = sample_metadata["strategy_run"]
    assert isinstance(strategy_run, dict)
    return [CallRecord.model_validate(call) for call in strategy_run["calls"]]


def replay(strategy: str, puzzle: PublicPuzzle, calls: list[CallRecord]) -> list[CallRecord]:
    """Re-run the strategy on the public puzzle alone, fed the recorded outputs in order."""
    replies = iter(calls)

    async def generate(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        call = next(replies)
        return output(call.text, call.stop_reason, call.output_tokens or 0)

    model = BudgetedModel(generate)
    asyncio.run(get_strategy(strategy)(puzzle, model))
    return model.calls


def test_every_sample_is_solved_and_scored(run: Run) -> None:
    assert run.log.samples is not None
    assert {sample.id for sample in run.log.samples} == set(run.samples)
    for sample in run.log.samples:
        assert sample.scores is not None
        assert sample.scores["grid_verifier"].value == {
            "format_ok": 1.0,
            "valid_grid": 1.0,
            "solved": 1.0,
            "cell_accuracy": 1.0,
            "truncated": 0.0,
            "context_limited": 0.0,
            "forced_answer": 0.0,
        }
    assert run.log.results is not None
    solved = next(score for score in run.log.results.scores if score.name == "solved")
    assert solved.metrics["accuracy"].value == 1.0


def test_every_call_comes_from_the_public_puzzle_and_earlier_outputs(run: Run) -> None:
    # The leak test. What reached the model is what the strategy recorded, and re-running the
    # strategy on the public puzzle alone, fed the same outputs, makes exactly the same calls:
    # nothing but the PublicPuzzle and the model's own earlier outputs shaped any message.
    assert run.log.samples is not None
    for sample in run.log.samples:
        calls = recorded_calls(sample.metadata)
        events = [event for event in sample.events if isinstance(event, ModelEvent)]
        assert len(events) == len(calls) >= 1
        for event, call in zip(events, calls, strict=True):
            assert event.input, "the log should hold the full model input"
            assert [(message.role, message.text) for message in event.input] == call.messages
        replayed = replay(run.strategy, public(run.samples[str(sample.id)]), calls)
        assert [call.messages for call in replayed] == [call.messages for call in calls]


def test_every_call_gets_the_budget_and_the_profile_sampling(run: Run) -> None:
    expected = MOCK.generate_config()
    assert run.configs
    for config in run.configs:
        assert config.max_tokens == FIRST_CAP  # one uncut call: the reasoning cap
        assert (config.temperature, config.top_p) == (expected.temperature, expected.top_p)
    assert run.log.plan.config.max_tokens == TOTAL_BUDGET  # the profile records the budget


def test_the_log_records_the_full_profile(run: Run) -> None:
    assert run.log.eval.metadata is not None
    assert run.log.eval.metadata["model_profile"] == MOCK.model_dump(mode="json")


def test_a_cut_off_answer_is_forced_from_the_reserve(tmp_path: Path) -> None:
    samples = logic_grid_samples([1], 2)
    caps: list[int] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        assert config.max_tokens is not None
        caps.append(config.max_tokens)
        if messages[-1].text == FORCE_ANSWER:
            return output(answer(puzzle_of(messages, samples).target), "stop", 50)
        return output("Let me check clue 3 again...", "max_tokens", config.max_tokens)

    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "1", "n_seeds": 2, "strategy": "cot"},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success" and log.samples is not None
    assert sorted(caps) == [ANSWER_RESERVE, ANSWER_RESERVE, FIRST_CAP, FIRST_CAP]
    for sample in log.samples:
        assert sample.scores is not None
        score = sample.scores["grid_verifier"]
        assert isinstance(score.value, dict) and score.metadata is not None
        assert (score.value["solved"], score.value["forced_answer"]) == (1.0, 1.0)
        assert score.value["truncated"] == 0.0  # the forced answer finished
        assert (score.metadata["n_calls"], score.metadata["budget_used"]) == (2, FIRST_CAP + 50)
        assert [call.forcing for call in recorded_calls(sample.metadata)] == [False, True]


def test_the_task_needs_a_model_under_evaluation() -> None:
    with pytest.raises(RuntimeError, match="no model is being evaluated"):
        logic_grid()


def test_samples_carry_ids_public_hidden_and_features() -> None:
    [sample] = logic_grid_samples([2], 1, seed_start=5)
    assert sample.id == "logic_grid/L2/s5"
    metadata = sample.metadata or {}
    assert {"public", "hidden", "level", "seed", "generator", "generator_version"} <= set(metadata)
    assert (metadata["level"], metadata["seed"], metadata["generator"]) == (2, 5, "logic_grid")
    assert metadata["generator_version"] == GENERATOR_VERSION
    assert set(metadata["hidden"]) == {"spec", "clues", "generator_version", "features_method"}
    assert public(sample).text == sample.input
    assert metadata["feature_n_clues"] == len(metadata["hidden"]["clues"])
    assert set(json.loads(sample.target)) == {
        name for name, _ in metadata["hidden"]["spec"]["attributes"]
    }


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_the_public_structure_only_restates_the_text(level: int) -> None:
    [sample] = logic_grid_samples([level], 1)
    structure = public(sample).structure
    assert set(structure) == {"houses", "attributes"}  # never the clues, never the answer
    assert structure == (sample.metadata or {})["hidden"]["spec"]
    text = str(sample.input)
    assert f"There are {structure['houses']} houses" in text
    attributes = structure["attributes"]
    assert isinstance(attributes, list)
    for _, values in attributes:
        assert all(value in text for value in values)


def test_the_dataset_is_identical_across_calls() -> None:
    assert logic_grid_samples([1, 2], 2) == logic_grid_samples([1, 2], 2)


def test_levels_parse_from_the_cli_forms() -> None:
    assert parse_levels("1,2,3") == [1, 2, 3]
    assert parse_levels(" 4 , 7 ") == [4, 7]
    assert parse_levels(2) == [2]  # `-T levels=2` arrives as an int
    assert parse_levels(["1", "2"]) == [1, 2]  # `-T levels=1,2` arrives as a list of strings
    assert parse_levels([1, 3]) == [1, 3]


@pytest.mark.parametrize("levels", ["0", "1,9", "", "two", ["1", "x"]])
def test_unknown_levels_are_rejected(levels: str | list[str]) -> None:
    with pytest.raises(ValueError, match="levels"):
        parse_levels(levels)


def test_an_unknown_strategy_lists_the_valid_ones() -> None:
    with pytest.raises(
        ValueError, match="unknown strategy 'oracle'; choose from cot, direct, z3_translate"
    ):
        logic_grid(strategy="oracle")
