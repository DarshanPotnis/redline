"""Vocabulary invariants that keep rendering and answer parsing unambiguous."""

import itertools
import re

import pytest

from redline.puzzles.logic_grid.vocab import VOCAB, Attribute

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


# Words that name or imply a house position. They belong to the position clues alone; v1's
# "the person in the blue house" was read as a house position (pilot-1e).
POSITION_WORDS = re.compile(r"\b(house|houses|left|right|next)\b|\d", re.IGNORECASE)


@pytest.mark.parametrize("attribute", VOCAB, ids=lambda attribute: attribute.name)
def test_no_template_label_or_value_names_a_position(attribute: Attribute) -> None:
    for text in (attribute.template, attribute.label, *attribute.values):
        assert not POSITION_WORDS.search(text), f"{attribute.name}: {text!r}"


@pytest.mark.parametrize("attribute", VOCAB, ids=lambda attribute: attribute.name)
def test_each_label_is_its_attribute_key(attribute: Attribute) -> None:
    # Models build answer keys from labels: text v2's "Favorite color" became "favorite_color".
    assert attribute.label.lower() == attribute.name
