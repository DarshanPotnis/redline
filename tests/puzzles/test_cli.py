"""The preview CLI prints the rendered puzzle, then its answer grid."""

import pytest

from redline.puzzles.logic_grid import generate, render
from redline.puzzles.logic_grid.__main__ import main


def test_preview_prints_puzzle_then_answer(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--level", "1", "--seed", "0"])
    out = capsys.readouterr().out
    puzzle = generate(1, 0)
    assert out.startswith(render(puzzle) + "\n\nAnswer:\nHouse")
    for house, name in enumerate(puzzle.solution.root["name"], start=1):
        assert any(line.split()[:2] == [str(house), name] for line in out.splitlines())
