"""The ladder chart: solved rate by level, one panel per model, one line per strategy.

A level a ladder never reached (it stopped early) has no point: absence, not a zero. Colors are
the first three categorical slots of the dataviz reference palette, the ones that stay
distinguishable for colorblind readers when every pair is on screen; more strategies than that
fall back to the remaining slots, in a fixed order.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from redline.analysis.ladder import LevelRow
from redline.puzzles.logic_grid import LADDER

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DODGE = 0.12  # x offset between strategies, so their error bars don't overlap


def points(rows: Sequence[LevelRow], model: str, strategy: str) -> list[LevelRow]:
    """The levels one line draws: those that ran and scored at least one sample."""
    return [row for row in rows if row.model == model and row.strategy == strategy and row.n > 0]


def level_label(level: int) -> str:
    houses, attributes = LADDER[level]
    return f"L{level}\n{houses}x{attributes}"


def ladder_chart(
    rows: Sequence[LevelRow],
    models: Sequence[str],
    strategies: Sequence[str],
    levels: Sequence[int],
    title: str,
    out: Path,
) -> Path:
    colors: Mapping[str, str] = {s: SERIES[i % len(SERIES)] for i, s in enumerate(strategies)}
    figure = Figure(figsize=(3.6 * len(models) + 0.6, 3.9), dpi=150, facecolor=SURFACE)
    axes = figure.subplots(1, len(models), sharey=True, squeeze=False)[0]
    for ax, model in zip(axes, models, strict=True):
        _style(ax, levels)
        ax.set_title(model.rsplit("/", 1)[-1], color=INK, fontsize=10, loc="left")
        for i, strategy in enumerate(strategies):
            drawn = points(rows, model, strategy)
            if not drawn:
                continue
            offset = (i - (len(strategies) - 1) / 2) * DODGE
            xs = [row.level + offset for row in drawn]
            ys = [row.solved / row.n for row in drawn]
            intervals = [row.interval or (0.0, 0.0) for row in drawn]
            ax.errorbar(
                xs,
                ys,
                yerr=[
                    [y - low for y, (low, _) in zip(ys, intervals, strict=True)],
                    [high - y for y, (_, high) in zip(ys, intervals, strict=True)],
                ],
                color=colors[strategy],
                linewidth=2,
                elinewidth=1,
                capsize=0,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=strategy,
                zorder=3,
            )
    axes[0].set_ylabel("solved (95% Wilson)", color=INK_SECONDARY, fontsize=9)
    handles, labels = _legend_entries(axes)
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
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    return out


def _style(ax: Axes, levels: Sequence[int]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(min(levels) - 0.5, max(levels) + 0.5)
    ax.set_xticks(list(levels), [level_label(level) for level in levels])
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
    """Each strategy once, from whichever panel drew it."""
    entries: dict[str, Artist] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    return list(entries.values()), list(entries)
