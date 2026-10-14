"""Reasoning Gym scoring: our extraction of the last "Answer:" line, then RG's own checker."""

import itertools

import pytest

from redline.puzzles.reasoning_gym import KNIGHTS_KNAVES, ReasoningGymGenerator, rg_dataset
from redline.scoring.reasoning_gym import chance_knights_knaves, extract_answer


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("Answer: Alice is a knight.", "Alice is a knight."),
        ("Thinking...\nAnswer: first try\nmore thought\nAnswer: final one", "final one"),
        ("Alice must be a knave because...\n\nAnswer: Alice is a knight.", "Alice is a knight."),
        ("**Answer:** Alice is a knave.", "Alice is a knave."),
        ("## Answer: Alice is a knave", "Alice is a knave"),
        ("answer: lower case works", "lower case works"),
    ],
)
def test_the_last_answer_line_is_extracted(text: str, answer: str) -> None:
    assert extract_answer(text) == answer


@pytest.mark.parametrize(
    "text",
    [
        "Alice is a knight, and Bob is a knave.",  # no Answer: line at all
        "Answer:",  # empty
        "The final Answer: Alice is a knight.",  # not at the start of a line
    ],
)
def test_no_answer_line_is_a_format_failure(text: str) -> None:
    assert extract_answer(text) is None


def item_and_checker(level: int, seed: int) -> tuple[dict[str, object], object]:
    assert isinstance(KNIGHTS_KNAVES, ReasoningGymGenerator)
    dataset = rg_dataset("knights_knaves", KNIGHTS_KNAVES.ladder[level], seed)
    return dataset[0], dataset


def test_a_correct_answer_scores_one_and_one_flip_scores_partial() -> None:
    item, dataset = item_and_checker(5, 0)
    answer = str(item["answer"])
    assert dataset.score_answer(answer, item) == 1.0
    terms = item["metadata"]["knight_knave_terms"]
    first, *rest = answer.rstrip(".").split(", ")
    hero, villain = terms["a_knight"], terms["a_knave"]
    flipped = first.replace(hero, "@").replace(villain, hero).replace("@", villain)
    partial = dataset.score_answer(", ".join([flipped, *rest]) + ".", item)
    assert 0 < partial < 1


@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_chance_matches_every_random_assignment_scored_by_rg(level: int) -> None:
    item, dataset = item_and_checker(level, 0)
    metadata = item["metadata"]
    names, terms = metadata["names"], metadata["knight_knave_terms"]
    roles = (terms["a_knight"], terms["a_knave"])
    scores = [
        dataset.score_answer(
            ", ".join(f"{name} is {role}" for name, role in zip(names, guess, strict=True)) + ".",
            item,
        )
        for guess in itertools.product(roles, repeat=len(names))
    ]
    assert sum(scores) / len(scores) == pytest.approx(chance_knights_knaves(len(names)))


def test_chance_at_two_people() -> None:
    # Both right 1/4, one right 1/2 scoring 0.65, none right 1/4 scoring 0.
    assert chance_knights_knaves(2) == pytest.approx(0.25 + 0.5 * 0.65)
