"""Vocabulary invariants that keep rendering and answer parsing unambiguous."""

import itertools

from redline.puzzles.logic_grid.vocab import VOCAB

VALUES = [value for attribute in VOCAB for value in attribute.values]


def test_six_attributes_of_six_values_with_name_first() -> None:
    assert len(VOCAB) == 6
    assert VOCAB[0].name == "name"
    assert all(len(attribute.values) == 6 for attribute in VOCAB)


def test_values_are_unique_across_attributes() -> None:
    assert len(set(VALUES)) == len(VALUES)


def test_no_value_is_a_substring_of_another() -> None:
    for a, b in itertools.permutations(VALUES, 2):
        assert a.lower() not in b.lower(), f"{a!r} is inside {b!r}"


def test_each_template_has_one_placeholder() -> None:
    assert all(attribute.template.count("{}") == 1 for attribute in VOCAB)
