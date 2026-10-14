"""Hypothesis profiles, and one small ladder run on mockllm shared by the runner, export and
analysis tests."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import settings
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, ModelUsage, get_model
from pydantic import TypeAdapter

from redline.puzzles.logic_grid.model import Clue
from redline.runner.ladder import inspect_cells, run
from redline.runner.manifest import GitState, Manifest, RunConfig
from redline.strategies.clue_format import translation_json
from redline.tasks.logic_grid import logic_grid_samples

settings.register_profile("ci", derandomize=True, max_examples=50, deadline=None)
# Hypothesis's default 200 ms deadline would make the brute-force level-3 examples flaky.
settings.register_profile("dev", deadline=None)
settings.load_profile("ci" if os.environ.get("CI") else "dev")

MOCK_CONFIG = RunConfig(
    run_id="mock",
    models=("mockllm/model",),
    strategies=("direct", "z3_translate"),
    levels=(1, 2, 3, 4),
    n_seeds=2,
    seed_start=0,
    early_stop=2,
)
MOCK_GIT = GitState(sha="0123456789abcdef0123456789abcdef01234567", dirty=False)
CONTEXT_LIMITED = "logic_grid/L2/s1"  # direct only: cut off below its cap, every call


@dataclass
class MockRun:
    runs_root: Path
    run_dir: Path
    manifest: Manifest
    calls: list[str]  # the sample id of every model call, across both runs
    calls_first_run: int
    resumed: Manifest


def _grid(target: str) -> str:
    columns = json.loads(target)
    houses = len(next(iter(columns.values())))
    return json.dumps(
        {
            str(h): {name: values[h - 1] for name, values in columns.items()}
            for h in range(1, houses + 1)
        }
    )


def _usage(output: ModelOutput, tokens: int) -> ModelOutput:
    usage = ModelUsage(input_tokens=300, output_tokens=tokens, total_tokens=300 + tokens)
    return output.model_copy(update={"usage": usage})


@pytest.fixture(scope="session")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> MockRun:
    """Levels 1-4, early stop 2, on a mock that solves level 1 and nothing above it.

    Both ladders climb L1-L3 and stop; L4 never runs. One direct sample at L2 is
    context-limited. The run is then started again, which must find every log complete.
    """
    samples: dict[str, Sample] = {
        str(sample.id): sample for sample in logic_grid_samples([1, 2, 3, 4], 2)
    }
    calls: list[str] = []

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        first_user = next(message.text for message in messages if message.role == "user")
        [sample_id] = [id for id, sample in samples.items() if str(sample.input) in first_user]
        calls.append(sample_id)
        sample = samples[sample_id]
        if sample_id.startswith("logic_grid/L1/"):
            hidden = (sample.metadata or {})["hidden"]
            text = (
                as_translation(hidden["clues"])
                if _is_translation(messages)
                else _grid(sample.target)
            )
            return _usage(ModelOutput.from_content("mockllm/model", text), 120)
        if sample_id == CONTEXT_LIMITED and not _is_translation(messages):
            out = ModelOutput.from_content("mockllm/model", "Let me", stop_reason="max_tokens")
            return _usage(out, 100)
        return _usage(ModelOutput.from_content("mockllm/model", "no idea"), 120)

    runs_root = tmp_path_factory.mktemp("runs")
    model = get_model("mockllm/model", custom_outputs=respond)
    cells = inspect_cells("none", {"mockllm/model": model})
    manifest = run(MOCK_CONFIG, runs_root=runs_root, run_cell=cells, git=MOCK_GIT)
    first = len(calls)
    resumed = run(MOCK_CONFIG, runs_root=runs_root, run_cell=cells, git=MOCK_GIT)
    return MockRun(runs_root, runs_root / "mock", manifest, calls, first, resumed)


def as_translation(clues: list[dict[str, object]]) -> str:
    """A perfect translator's reply: stored clues, rewritten in the translation format."""
    return translation_json(TypeAdapter(list[Clue]).validate_python(clues))


def _is_translation(messages: list[ChatMessage]) -> bool:
    """z3_translate asks for the clues as JSON; the grid strategies ask for the grid."""
    return any('"kind"' in message.text for message in messages)
