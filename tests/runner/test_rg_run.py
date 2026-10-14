"""A knights-and-knaves ladder through the runner, export and report on mockllm: any generator
plugs in."""

from pathlib import Path

import duckdb
import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model

from redline.analysis import build_report
from redline.puzzles.reasoning_gym import KNIGHTS_KNAVES
from redline.runner.ladder import inspect_cells, run
from redline.runner.manifest import GitState, Manifest, RunConfig
from redline.tasks.reasoning_gym import rg_samples

CONFIG = RunConfig(
    run_id="rg",
    generator="knights_knaves",
    models=("mockllm/model",),
    strategies=("direct",),
    levels=(1, 2, 3, 4),
    n_seeds=2,
    seed_start=0,
    early_stop=2,
)


@pytest.fixture(scope="module")
def rg_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    """Answers level 1 right and every later level with the roles of nobody in the puzzle."""
    samples = rg_samples(KNIGHTS_KNAVES, [1, 2, 3, 4], 2)

    def respond(
        messages: list[ChatMessage], _tools: object, _choice: object, config: GenerateConfig
    ) -> ModelOutput:
        first_user = next(message.text for message in messages if message.role == "user")
        [sample] = [s for s in samples if str(s.input) in first_user]
        level = (sample.metadata or {})["level"]
        reply = f"Answer: {sample.target}" if level == 1 else "Answer: Nobody is a knight."
        return ModelOutput.from_content("mockllm/model", reply)

    root = tmp_path_factory.mktemp("rg-runs")
    model = get_model("mockllm/model", custom_outputs=respond)
    manifest = run(
        CONFIG,
        runs_root=root,
        run_cell=inspect_cells("none", {"mockllm/model": model}),
        git=GitState(sha="f" * 40, dirty=False),
    )
    return root / "rg", manifest


def test_the_ladder_climbs_and_stops_early(rg_run: tuple[Path, Manifest]) -> None:
    _, manifest = rg_run
    [ladder] = manifest.ladders
    assert [(level.level, level.solved, level.n) for level in ladder.levels] == [
        (1, 2, 2),
        (2, 0, 2),
        (3, 0, 2),
    ]
    assert ladder.stop_reason == "early_stop"
    assert manifest.config.generator == "knights_knaves"
    assert manifest.provenance.reasoning_gym is not None


def test_export_report_and_chart(rg_run: tuple[Path, Manifest], tmp_path: Path) -> None:
    run_dir, _ = rg_run
    paths = build_report(run_dir, tmp_path / "data", tmp_path / "reports")
    rows = duckdb.sql(
        f"""
        SELECT generator, level, count_if(solved), avg(rg_score), count(cell_accuracy),
               count_if(valid_grid IS NULL), any_value(sample_id)
        FROM '{paths.parquet}' GROUP BY ALL ORDER BY level
        """
    ).fetchall()
    assert [(g, level, solved, cells, nulls) for g, level, solved, _, cells, nulls, _ in rows] == [
        ("reasoning_gym/knights_knaves", 1, 2, 0, 2),
        ("reasoning_gym/knights_knaves", 2, 0, 0, 2),
        ("reasoning_gym/knights_knaves", 3, 0, 0, 2),
    ]
    assert rows[0][3] == 1.0 and rows[1][3] == 0.0
    assert rows[0][6].startswith("knights_knaves/L1/")
    markdown = paths.markdown.read_text()
    assert "| Generator | reasoning_gym/knights_knaves |" in markdown
    assert "| RG score | Chance |" in markdown
    assert "| knights_knaves | model | direct | 1 | 2p d2 w2 | 2 | 0 | 2/2 |" in markdown
    assert "random roles" in markdown
    assert paths.chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
