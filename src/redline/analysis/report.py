"""The ladder report: reports/<run_id>.md with its chart in reports/figures/."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from redline.analysis.chart import ladder_chart, level_label
from redline.analysis.ladder import HealthRow, LevelRow, health_rows, level_rows
from redline.budget import ANSWER_RESERVE, TOTAL_BUDGET
from redline.export import to_parquet
from redline.export.parquet import DATA_ROOT
from redline.runner.manifest import Manifest, load

logger = logging.getLogger(__name__)

REPORTS = Path("reports")


@dataclass(frozen=True)
class ReportPaths:
    parquet: Path
    markdown: Path
    chart: Path


def build_report(
    run_dir: Path, data_root: Path = DATA_ROOT, reports: Path = REPORTS
) -> ReportPaths:
    """Export the run to Parquet, then write its chart and markdown report."""
    manifest = load(run_dir)
    config = manifest.config
    parquet = to_parquet(run_dir, data_root)
    levels = level_rows(parquet, config.models, config.strategies)
    health = health_rows(parquet, config.models, config.strategies)
    chart = ladder_chart(
        levels,
        config.models,
        config.strategies,
        config.levels,
        title=f"{config.run_id} ({manifest.seed_split} seeds): solved rate and cell accuracy",
        out=reports / "figures" / f"{config.run_id}_ladder.png",
    )
    markdown = reports / f"{config.run_id}.md"
    markdown.write_text(render(manifest, levels, health, chart.relative_to(reports).as_posix()))
    logger.info("wrote %s and %s", markdown, chart)
    return ReportPaths(parquet, markdown, chart)


def render(
    manifest: Manifest, levels: Sequence[LevelRow], health: Sequence[HealthRow], chart: str
) -> str:
    config = manifest.config
    return "\n".join(
        [
            f"# Ladder report: {config.run_id}",
            "",
            _split_note(manifest),
            "",
            *_run_table(manifest),
            "",
            f"![Solved rate and cell accuracy by level, one column per model]({chart})",
            "",
            "## Solved rate by level",
            "",
            "Context-limited samples are harness failures (the context window cut the model off "
            "before its budget ran out). They are excluded from every column but their own. "
            "Intervals are 95% Wilson. Levels a ladder never reached are absent, not zero. "
            "Chance is the cell accuracy a random valid grid would get (1/houses); an answer "
            "that fails to parse scores 0 cells, so a strategy can land below chance.",
            "",
            *_level_table(levels),
            "",
            "## Ladders",
            "",
            *_ladder_table(manifest),
            "",
            "## Harness health",
            "",
            *_health_section(manifest, health),
            "",
        ]
    )


def _split_note(manifest: Manifest) -> str:
    config = manifest.config
    seeds = f"seeds {config.seed_start}-{config.seed_start + config.n_seeds - 1}"
    if manifest.seed_split == "dev":
        return f"**DEV {seeds}: for tuning and pipeline checks, not a reported result.**"
    return f"**TEST {seeds}.**"


def _run_table(manifest: Manifest) -> list[str]:
    config = manifest.config
    git = manifest.provenance.git_sha[:12] + (" (dirty tree)" if manifest.git_dirty else "")
    models = "<br>".join(
        f"`{model}`"
        + (
            f" (Modelfile sha256 `{profile.modelfile_sha256[:12]}`)"
            if profile.modelfile_sha256
            else ""
        )
        for model, profile in manifest.profiles.items()
    )
    sessions = len(manifest.sessions)
    rows = [
        ("Commit", f"`{git}`"),
        ("Models", models),
        ("Strategies", ", ".join(config.strategies)),
        (
            "Ladder",
            f"levels {_span(config.levels)}, {config.n_seeds} seeds per level; a climb stops "
            f"after {config.early_stop} consecutive levels with nothing solved",
        ),
        ("Budget", f"{TOTAL_BUDGET:,} output tokens per sample ({ANSWER_RESERVE} reserve)"),
        (
            "Runtime",
            f"{_duration(manifest.runtime_seconds())} over {sessions} "
            f"session{'s' if sessions != 1 else ''}",
        ),
        (
            "Software",
            f"inspect_ai {manifest.provenance.inspect_ai}, z3 {manifest.provenance.z3}, "
            f"Python {manifest.provenance.python}",
        ),
    ]
    return ["| | |", "|---|---|", *(f"| {name} | {value} |" for name, value in rows)]


def _level_table(levels: Sequence[LevelRow]) -> list[str]:
    header = (
        "| Model | Strategy | Level | Grid | n | Ctx-lim. | Solved | 95% CI | Cell acc. | Chance "
        "| Format OK | Forced | Out tokens | Seconds |"
    )
    lines = [header, _rule(header)]
    for row in levels:
        interval = row.interval
        lines.append(
            "| "
            + " | ".join(
                [
                    _short(row.model),
                    row.strategy,
                    str(row.level),
                    level_label(row.level).split("\n")[1],
                    str(row.n),
                    str(row.context_limited),
                    f"{row.solved}/{row.n}",
                    "n/a" if interval is None else f"{interval[0]:.0%}-{interval[1]:.0%}",
                    _percent(row.cell_accuracy),
                    _percent(row.chance),
                    _percent(row.format_ok),
                    _percent(row.forced_answer),
                    _number(row.output_tokens),
                    _number(row.seconds),
                ]
            )
            + " |"
        )
    return lines


def _ladder_table(manifest: Manifest) -> list[str]:
    header = "| Model | Strategy | Levels run | Stop | Detail |"
    lines = [header, _rule(header)]
    for ladder in manifest.ladders:
        run = ", ".join(f"L{level.level} {level.solved}/{level.n}" for level in ladder.levels)
        lines.append(
            f"| {_short(ladder.model)} | {ladder.strategy} | {run} | "
            f"{ladder.stop_reason or 'unfinished'} | {ladder.stop_detail or ''} |"
        )
    return lines


def _health_section(manifest: Manifest, health: Sequence[HealthRow]) -> list[str]:
    samples = sum(row.samples for row in health)
    limited = sum(row.context_limited for row in health)
    errored = [
        f"{_short(ladder.model)} {ladder.strategy} L{level.level}: {level.status}"
        for ladder in manifest.ladders
        for level in ladder.levels
        if level.status != "success"
    ]
    header = (
        "| Model | Strategy | Samples | Ctx-lim. | Format OK | Forced answer | Truncated "
        "| Sample time |"
    )
    lines = [
        f"- Context-limited samples: **{limited} of {samples}** (excluded from accuracy).",
        "- Errored cells: " + ("; ".join(errored) if errored else "none."),
        "- Format OK, forced answer and truncated are shares of the samples that were not "
        "context-limited. A forced answer means the reasoning call hit its cap and the "
        "reserve forced a final answer.",
        "",
        header,
        _rule(header),
    ]
    for row in health:
        lines.append(
            f"| {_short(row.model)} | {row.strategy} | {row.samples} | {row.context_limited} | "
            f"{_percent(row.format_ok)} | {_percent(row.forced_answer)} | "
            f"{_percent(row.truncated)} | {_duration(row.seconds)} |"
        )
    return lines


def _rule(header: str) -> str:
    return "|" + "---|" * (header.count("|") - 1)


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def _span(levels: Sequence[int]) -> str:
    contiguous = list(levels) == list(range(levels[0], levels[-1] + 1))
    return f"{levels[0]}-{levels[-1]}" if contiguous else ", ".join(map(str, levels))


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"
