"""The Reasoning Gym adapter: deterministic items, the answer key only in the hidden instance."""

import json

import pytest

from redline.puzzles.reasoning_gym import (
    ANSWER_FORMAT,
    KNIGHTS_KNAVES,
    ReasoningGymGenerator,
    rg_item,
)
from redline.puzzles.registry import get_generator


def test_generation_is_deterministic() -> None:
    assert KNIGHTS_KNAVES.generate(3, 7) == KNIGHTS_KNAVES.generate(3, 7)
    assert KNIGHTS_KNAVES.generate(3, 7).text != KNIGHTS_KNAVES.generate(3, 8).text


def test_the_ladder_has_seven_levels_of_rising_size() -> None:
    assert isinstance(KNIGHTS_KNAVES, ReasoningGymGenerator)
    assert KNIGHTS_KNAVES.levels == (1, 2, 3, 4, 5, 6, 7)
    people = [KNIGHTS_KNAVES.ladder[level]["n_people"] for level in KNIGHTS_KNAVES.levels]
    assert people == [2, 3, 4, 4, 5, 6, 7]
    assert [KNIGHTS_KNAVES.describe(level) for level in (1, 4, 7)] == [
        "2p d2 w2",
        "4p d3 w3",
        "7p d4 w3",
    ]


def test_the_instance_fields() -> None:
    instance = KNIGHTS_KNAVES.generate(4, 2)
    item = rg_item("knights_knaves", KNIGHTS_KNAVES.ladder[4], 2)
    assert instance.generator == "reasoning_gym/knights_knaves"
    assert (instance.level, instance.seed) == (4, 2)
    assert instance.text == item["question"]
    assert instance.answer == item["answer"]
    assert instance.answer_format == ANSWER_FORMAT
    assert instance.structure == {} and instance.search_space is None
    assert instance.instance["task"] == "knights_knaves"
    assert instance.instance["config"] == {
        "n_people": 4,
        "depth_constraint": 3,
        "width_constraint": 3,
    }
    assert instance.instance["seed"] == 2
    assert instance.features == {
        "n_people": 4.0,
        "depth_constraint": 3.0,
        "width_constraint": 3.0,
        "n_chars": float(len(item["question"])),
    }


def test_the_answer_key_stays_hidden() -> None:
    instance = KNIGHTS_KNAVES.generate(3, 0)
    assert "solution" in instance.instance["metadata"]  # RG's metadata holds the answer key
    public = instance.public()
    assert public.generator == "reasoning_gym/knights_knaves" and public.family == "reasoning_gym"
    dumped = public.model_dump_json()
    assert "solution" not in dumped and str(instance.answer) not in dumped


@pytest.mark.parametrize("name", ["knights_knaves", "reasoning_gym/knights_knaves"])
def test_the_registry_finds_it_by_either_name(name: str) -> None:
    assert get_generator(name) is KNIGHTS_KNAVES


def test_an_unknown_generator_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown generator"):
        get_generator("sudoku")


def test_metadata_is_plain_json() -> None:
    instance = KNIGHTS_KNAVES.generate(7, 0)
    assert json.loads(json.dumps(instance.instance)) == instance.instance
