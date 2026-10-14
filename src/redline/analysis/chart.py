"""The ladder chart: one column per model, one line per strategy, and two rows per generator. The
first row is the solved rate with Wilson intervals; the second is the generator's secondary score
(the logic grid's cell accuracy, Reasoning Gym's partial credit) with a dashed line at chance.

A level a ladder never reached (it stopped early) has no point: absence, not a zero. Colors are
the first three categorical slots of the dataviz reference palette, the ones that stay
distinguishable for colorblind readers when every pair is on screen; more strategies than that
fall back to the remaining slots, in a fixed order.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from redline.analysis.ladder import LevelRow, Secondary, secondary
from redline.puzzles.registry import get_generator

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DODGE = 0.12  # x offset between strategies, so their error bars don't overlap
ROW_HEIGHT = 3.3  # inches per panel row


def points(rows: Sequence[LevelRow], model: str, strategy: str) -> list[LevelRow]:
    """The levels one line draws: those that ran and scored at least one sample."""
    return [row for row in rows if row.model == model and row.strategy == strategy and row.n > 0]


def level_label(generator: str, level: int) -> str:
    return f"L{level}\n{get_generator(generator).describe(level)}"


def ladder_chart(
    rows: Sequence[LevelRow],
    models: Sequence[str],
    strategies: Sequence[str],
    levels: Sequence[int],
    title: str,
    out: Path,
) -> Path:
    colors: Mapping[str, str] = {s: SERIES[i % len(SERIES)] for i, s in enumerate(strategies)}
    generators = sorted({row.generator for row in rows})
    figure = Figure(
        figsize=(3.6 * len(models) + 0.6, ROW_HEIGHT * 2 * len(generators) + 0.4),
        dpi=150,
        facecolor=SURFACE,
    )
    grid = figure.subplots(2 * len(generators), len(models), sharey=True, squeeze=False)
    for g, generator in enumerate(generators):
        own = [row for row in rows if row.generator == generator]
        second = secondary(generator)
        solved_axes, score_axes = grid[2 * g], grid[2 * g + 1]
        for column, model in enumerate(models):
            ax, scores = solved_axes[column], score_axes[column]
            _style(ax, generator, levels)
            _style(scores, generator, levels)
            heading = model.rsplit("/", 1)[-1]
            if len(generators) > 1:
                heading += f" · {generator.removeprefix('reasoning_gym/')}"
            ax.set_title(heading, color=INK, fontsize=10, loc="left")
            _chance_line(scores, levels, second)
            for i, strategy in enumerate(strategies):
                _draw(
                    ax,
                    scores,
                    points(own, model, strategy),
                    i,
                    len(strategies),
                    colors[strategy],
                    strategy,
                )
        solved_axes[0].set_ylabel("solved (95% Wilson)", color=INK_SECONDARY, fontsize=9)
        score_axes[0].set_ylabel(second.axis, color=INK_SECONDARY, fontsize=9)
    handles, labels = _legend_entries([ax for row in grid for ax in row])
    figure.legend(
        handles,
        labels,
        loc="upper right",
        ncols=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.3 / figure.get_figheight()))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    return out


def _draw(
    ax: Axes,
    scores: Axes,
    drawn: Sequence[LevelRow],
    index: int,
    count: int,
    color: str,
    strategy: str,
) -> None:
    """One strategy's line on a model's two panels."""
    if not drawn:
        return
    offset = (index - (count - 1) / 2) * DODGE
    xs = [row.level + offset for row in drawn]
    ys = [row.solved / row.n for row in drawn]
    intervals = [row.interval or (0.0, 0.0) for row in drawn]
    style: dict[str, Any] = {
        "color": color,
        "linewidth": 2,
        "marker": "o",
        "markersize": 6,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 1.5,
        "zorder": 3,
    }
    scores.plot(xs, [math.nan if row.score is None else row.score for row in drawn], **style)
    ax.errorbar(
        xs,
        ys,
        yerr=[
            [y - low for y, (low, _) in zip(ys, intervals, strict=True)],
            [high - y for y, (_, high) in zip(ys, intervals, strict=True)],
        ],
        elinewidth=1,
        capsize=0,
        label=strategy,
        **style,
    )


def _chance_line(ax: Axes, levels: Sequence[int], second: Secondary) -> None:
    """A dashed step at chance across each level's slot (none where chance is unknown)."""
    known = [(level, second.chance(level)) for level in levels]
    steps = [(level, value) for level, value in known if value is not None]
    if not steps:
        return
    ax.hlines(
        [value for _, value in steps],
        [level - 0.5 for level, _ in steps],
        [level + 0.5 for level, _ in steps],
        colors=MUTED,
        linestyles="dashed",
        linewidth=1.2,
        label=second.chance_label,
        zorder=2,
    )


def _style(ax: Axes, generator: str, levels: Sequence[int]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(min(levels) - 0.5, max(levels) + 0.5)
    ax.set_xticks(list(levels), [level_label(generator, level) for level in levels])
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def _legend_entries(axes: Sequence[Axes]) -> tuple[list[Artist], list[str]]:
    """Each strategy (and the chance line) once, from whichever panel drew it."""
    entries: dict[str, Artist] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    return list(entries.values()), list(entries)
