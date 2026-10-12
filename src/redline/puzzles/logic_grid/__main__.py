"""Preview CLI: `python -m redline.puzzles.logic_grid --level 3 --seed 7` prints a puzzle and
its answer grid. This is the only module where print() is fine."""

import argparse
from collections.abc import Sequence

from redline.puzzles.logic_grid.generate import LADDER, generate
from redline.puzzles.logic_grid.model import LogicGridPuzzle
from redline.puzzles.logic_grid.render import render
from redline.puzzles.logic_grid.vocab import ATTRIBUTES


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m redline.puzzles.logic_grid",
        description="Print a logic-grid puzzle, then its answer grid.",
    )
    parser.add_argument("--level", type=int, required=True, choices=sorted(LADDER))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    puzzle = generate(args.level, args.seed)
    print(render(puzzle))
    print()
    print("Answer:")
    print(answer_grid(puzzle))


def answer_grid(puzzle: LogicGridPuzzle) -> str:
    names = puzzle.spec.attribute_names
    header = ["House", *(ATTRIBUTES[name].label for name in names)]
    rows = [
        [str(house), *(puzzle.solution.root[name][house - 1] for name in names)]
        for house in range(1, puzzle.spec.houses + 1)
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    )


if __name__ == "__main__":
    main()
