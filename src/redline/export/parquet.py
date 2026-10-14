"""A run's Inspect logs to one Parquet file, one row per sample.

Only the logs the run manifest lists as successful are read, so a crashed or retried attempt
never double-counts. Samples come from `inspect_ai.analysis.samples_df` over the sample
summaries, which carry every score and score metadata field this needs; the rows are written
through an explicit schema so types are stable across runs (translation columns are null for
strategies that don't translate).
"""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from inspect_ai.analysis import EvalColumn, SampleColumn, samples_df
from inspect_ai.log import EvalSampleSummary
from inspect_ai.scorer import Score

from redline.runner.manifest import Manifest, load
from redline.scoring.harness import ANSWER_SCORERS

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/runs")
SAMPLES = "samples.parquet"

# Answer-scorer flags; valid_grid is the logic grid's alone (null for other generators).
FLAGS = ("solved", "format_ok", "valid_grid", "truncated", "context_limited", "forced_answer")
ANSWER = "answer"  # whichever answer scorer the sample has (scoring.harness.ANSWER_SCORERS)
SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("generator", pa.string()),  # e.g. logic_grid, reasoning_gym/knights_knaves
        ("model", pa.string()),
        ("strategy", pa.string()),
        ("level", pa.int64()),
        ("seed", pa.int64()),
        ("sample_id", pa.string()),  # the dataset id, e.g. logic_grid/L3/s7: joins strategies
        ("generator_version", pa.int64()),  # puzzle text version; null before 1e.1 (= 1)
        *((flag, pa.bool_()) for flag in FLAGS),
        ("cell_accuracy", pa.float64()),  # logic grid only
        ("rg_score", pa.float64()),  # Reasoning Gym only: RG's partial credit
        ("translation_exact", pa.bool_()),  # null unless the strategy translates
        ("translation_status", pa.string()),
        # translated clues: exact, inferred (true, not stated), wrong; items dropped as malformed
        ("n_exact", pa.int64()),
        ("n_inferred", pa.int64()),
        ("n_wrong", pa.int64()),
        ("n_malformed", pa.int64()),
        # stated clues: captured (translated exactly), entailed (implied), missed
        ("n_captured", pa.int64()),
        ("n_entailed", pa.int64()),
        ("n_missed", pa.int64()),
        ("input_tokens", pa.int64()),  # summed over every call of the sample
        ("output_tokens", pa.int64()),
        ("reasoning_tokens", pa.int64()),  # null when the provider reports none
        ("n_calls", pa.int64()),
        ("seconds", pa.float64()),  # Inspect's working time for the sample
        ("stop_reason", pa.string()),  # the final model call's
        ("ladder_stop_reason", pa.string()),  # why this (model, strategy) stopped climbing
    ]
)
FEATURE_PREFIX = "feature_"
TRANSLATION_COUNTS = (
    "n_exact",
    "n_inferred",
    "n_wrong",
    "n_malformed",
    "n_captured",
    "n_entailed",
    "n_missed",
)


def to_parquet(run_dir: Path, out_root: Path = DATA_ROOT) -> Path:
    """Write <out_root>/<run_id>/samples.parquet from the run in `run_dir`; return its path."""
    manifest = load(run_dir)
    run_id = manifest.config.run_id
    logs = [
        run_dir / level.log
        for ladder in manifest.ladders
        for level in ladder.levels
        if level.status == "success" and level.log is not None
    ]
    table = samples_table(manifest, logs)
    out = out_root / run_id / SAMPLES
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    logger.info("wrote %d samples from %d logs to %s", table.num_rows, len(logs), out)
    return out


def samples_table(manifest: Manifest, logs: list[Path]) -> pa.Table:
    if not logs:
        return SCHEMA.empty_table()
    frame = samples_df([str(log) for log in logs], columns=COLUMNS, quiet=True)
    frame = frame.drop(columns=["sample_id"]).rename(columns={"id": "sample_id"})
    frame["run_id"] = manifest.config.run_id
    stops = {(ladder.model, ladder.strategy): ladder.stop_reason for ladder in manifest.ladders}
    frame["ladder_stop_reason"] = [
        stops.get((model, strategy))
        for model, strategy in zip(frame["model"], frame["strategy"], strict=True)
    ]
    features = sorted(name for name in frame.columns if name.startswith(FEATURE_PREFIX))
    schema = pa.schema([*SCHEMA, *((name, pa.float64()) for name in features)])
    frame = frame.sort_values(["model", "strategy", "level", "seed"], kind="stable")
    return pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)


def _score(summary: EvalSampleSummary, scorer: str) -> Score | None:
    scores = summary.scores or {}
    if scorer == ANSWER:
        return next((scores[name] for name in ANSWER_SCORERS if name in scores), None)
    return scores.get(scorer)


def _value(scorer: str, key: str) -> Callable[[EvalSampleSummary], Any]:
    def extract(summary: EvalSampleSummary) -> Any:
        score = _score(summary, scorer)
        if score is None or not isinstance(score.value, Mapping):
            return None
        return score.value.get(key)

    return extract


def _metadata(scorer: str, key: str) -> Callable[[EvalSampleSummary], Any]:
    def extract(summary: EvalSampleSummary) -> Any:
        score = _score(summary, scorer)
        return None if score is None else (score.metadata or {}).get(key)

    return extract


def _flag(scorer: str, key: str) -> Callable[[EvalSampleSummary], bool | None]:
    value = _value(scorer, key)

    def extract(summary: EvalSampleSummary) -> bool | None:
        number = value(summary)
        return None if number is None else number == 1

    return extract


def _usage(field: str) -> Callable[[EvalSampleSummary], int | None]:
    def extract(summary: EvalSampleSummary) -> int | None:
        counts = [getattr(usage, field) for usage in summary.model_usage.values()]
        known = [count for count in counts if count is not None]
        return sum(known) if known else None

    return extract


def _features(summary: EvalSampleSummary) -> dict[str, Any]:
    return {
        name.removeprefix(FEATURE_PREFIX): value
        for name, value in summary.metadata.items()
        if name.startswith(FEATURE_PREFIX)
    }


COLUMNS = [
    EvalColumn("model", path="eval.model", required=True),
    EvalColumn("strategy", path="eval.task_args.strategy", required=True),
    SampleColumn("id", path="id", required=True),
    SampleColumn("level", path="metadata.level", required=True),
    SampleColumn("seed", path="metadata.seed", required=True),
    SampleColumn("generator", path="metadata.generator", required=True),
    SampleColumn("generator_version", path="metadata.generator_version"),
    *(SampleColumn(flag, path=_flag(ANSWER, flag)) for flag in FLAGS),
    SampleColumn("cell_accuracy", path=_value("grid_verifier", "cell_accuracy")),
    SampleColumn("rg_score", path=_value("rg_verifier", "rg_score")),
    SampleColumn("translation_exact", path=_flag("translation_verifier", "translation_exact")),
    SampleColumn(
        "translation_status", path=_metadata("translation_verifier", "translation_status")
    ),
    *(
        SampleColumn(count, path=_metadata("translation_verifier", count))
        for count in TRANSLATION_COUNTS
    ),
    SampleColumn("input_tokens", path=_usage("input_tokens")),
    SampleColumn("output_tokens", path=_usage("output_tokens")),
    SampleColumn("reasoning_tokens", path=_usage("reasoning_tokens")),
    SampleColumn("n_calls", path=_metadata(ANSWER, "n_calls")),
    SampleColumn("seconds", path="working_time"),
    SampleColumn("stop_reason", path=_metadata(ANSWER, "stop_reason")),
    SampleColumn(f"{FEATURE_PREFIX}*", path=_features),
]
