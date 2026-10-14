"""The runner's command line: level specs, list arguments, and refusals before any run."""

import argparse
from pathlib import Path

import pytest

from redline.runner.__main__ import build_parser, config_from, main, parse_level_spec


@pytest.mark.parametrize(
    ("spec", "levels"),
    [("1-7", [1, 2, 3, 4, 5, 6, 7]), ("1,3,5", [1, 3, 5]), ("1-3,5", [1, 2, 3, 5]), ("4", [4])],
)
def test_level_specs(spec: str, levels: list[int]) -> None:
    assert parse_level_spec(spec) == levels


@pytest.mark.parametrize("spec", ["0-2", "a-b", "1-", ""])
def test_bad_level_specs(spec: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_level_spec(spec)


def test_models_and_strategies_accept_spaces_or_commas() -> None:
    args = build_parser().parse_args(
        ["--models", "a,b", "c", "--strategies", "direct", "--n-seeds", "5", "--run-id", "x"]
    )
    config = config_from(args)
    assert config.models == ("a", "b", "c")
    assert config.levels == (1, 2, 3, 4, 5, 6, 7)
    assert (config.seed_start, config.early_stop) == (0, 2)


@pytest.mark.parametrize(
    "extra",
    [
        ["--run-id", "../escape"],
        ["--run-id", "ok", "--seed-start", "1000"],
        ["--run-id", "ok", "--strategies", "guess"],
        ["--run-id", "ok", "--levels", "8"],  # past the logic-grid ladder
        ["--run-id", "ok", "--generator", "knights_knaves", "--strategies", "z3_translate"],
    ],
)
def test_refusals_exit_with_2_and_run_nothing(tmp_path: Path, extra: list[str]) -> None:
    argv = ["--models", "mockllm/model", "--n-seeds", "1", "--runs-root", str(tmp_path)]
    if "--strategies" not in extra:
        argv += ["--strategies", "direct"]
    assert main([*argv, *extra]) == 2
    assert list(tmp_path.iterdir()) == []
