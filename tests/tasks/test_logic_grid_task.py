"""The logic-grid task end to end on mockllm: solvable, leak-free, budget-matched, reproducible."""

import json
from dataclasses import dataclass

import inspect_ai
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model

from redline.puzzles.base import PublicPuzzle
from redline.runtime.profiles import profile_for
from redline.strategies import STRATEGIES, get_strategy
from redline.tasks.logic_grid import logic_grid, logic_grid_samples, parse_levels

MOCK = profile_for("mockllm/model")


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
    samples = {str(sample.id): sample for sample in logic_grid_samples([1, 2, 3], 2)}
    build = get_strategy(strategy)
    answers = {build(public(s))[-1].text: answer(s.target) for s in samples.values()}
    configs: list[GenerateConfig] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        configs.append(config)
        grid = answers[messages[-1].text]
        reply = (
            grid
            if strategy == "direct"
            else f"Working through the clues...\n\n```json\n{grid}\n```"
        )
        return ModelOutput.from_content(model="mockllm/model", content=reply)

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
    return Run(strategy, samples, configs, read_eval_log(log.location, resolve_attachments=True))


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
        }
    assert run.log.results is not None
    solved = next(score for score in run.log.results.scores if score.name == "solved")
    assert solved.metrics["accuracy"].value == 1.0


def test_the_model_saw_only_the_public_puzzle(run: Run) -> None:
    # The leak test: what reached the model is exactly build_messages(PublicPuzzle).
    assert run.log.samples is not None
    build = get_strategy(run.strategy)
    for sample in run.log.samples:
        [event] = [event for event in sample.events if isinstance(event, ModelEvent)]
        assert event.input, "the log should hold the full model input"
        expected = build(public(run.samples[str(sample.id)]))
        assert [(m.role, m.content) for m in event.input] == [(m.role, m.content) for m in expected]


def test_every_call_gets_the_profile_config(run: Run) -> None:
    # The budget rule: every strategy gets the same profile-defined config, nothing else.
    expected = MOCK.generate_config()
    assert run.configs
    for config in [*run.configs, run.log.plan.config]:
        assert config.max_tokens == expected.max_tokens == 4096
        assert (config.temperature, config.top_p) == (expected.temperature, expected.top_p)


def test_the_log_records_the_full_profile(run: Run) -> None:
    assert run.log.eval.metadata is not None
    assert run.log.eval.metadata["model_profile"] == MOCK.model_dump(mode="json")


def test_the_task_needs_a_model_under_evaluation() -> None:
    with pytest.raises(RuntimeError, match="no model is being evaluated"):
        logic_grid()


def test_samples_carry_ids_public_hidden_and_features() -> None:
    [sample] = logic_grid_samples([2], 1, seed_start=5)
    assert sample.id == "logic_grid/L2/s5"
    metadata = sample.metadata or {}
    assert {"public", "hidden", "level", "seed", "generator"} <= set(metadata)
    assert (metadata["level"], metadata["seed"], metadata["generator"]) == (2, 5, "logic_grid")
    assert set(metadata["hidden"]) == {"spec", "clues", "features_method"}
    assert public(sample).text == sample.input
    assert metadata["feature_n_clues"] == len(metadata["hidden"]["clues"])
    assert set(json.loads(sample.target)) == {
        name for name, _ in metadata["hidden"]["spec"]["attributes"]
    }


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
    with pytest.raises(ValueError, match="unknown strategy 'oracle'; choose from cot, direct"):
        logic_grid(strategy="oracle")
