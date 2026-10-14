"""Misread diagnosis: translated clues are exact, inferred or wrong; stated clues are captured,
entailed or missed; malformed items are dropped and counted."""

import json
from pathlib import Path

import inspect_ai
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ModelOutput, get_model

from conftest import as_translation
from redline.puzzles.logic_grid import generate
from redline.puzzles.logic_grid.model import (
    AtClue,
    Clue,
    ImmediatelyLeftClue,
    LeftOfClue,
    NextToClue,
    NotAtClue,
    NotSameHouseClue,
    Ref,
    SameHouseClue,
    Solution,
    Spec,
)
from redline.scoring.translation import canonical, diagnose
from redline.tasks.logic_grid import logic_grid, logic_grid_samples


def with_kind(kind: type, seeds: range = range(50)) -> tuple[int, int]:
    """A level-3 (level, seed) whose puzzle has at least one clue of `kind`."""
    for seed in seeds:
        if any(isinstance(clue, kind) for clue in generate(3, seed).clues):
            return 3, seed
    raise AssertionError(f"no puzzle with a {kind.__name__} clue")


def test_the_exact_translation_has_nothing_wrong_or_missed() -> None:
    p = generate(3, 0)
    diagnosis = diagnose(p.spec, list(p.clues), list(p.clues), p.solution)
    every = tuple(range(1, len(p.clues) + 1))
    assert (diagnosis.exact, diagnosis.captured) == (every, every)
    assert diagnosis.inferred == diagnosis.wrong == diagnosis.entailed == diagnosis.missed == ()


def test_a_flipped_left_of_is_wrong_and_its_clue_missed() -> None:
    p = generate(*with_kind(LeftOfClue))
    i = next(i for i, clue in enumerate(p.clues) if isinstance(clue, LeftOfClue))
    clue = p.clues[i]
    assert isinstance(clue, LeftOfClue)
    misread = [*p.clues[:i], LeftOfClue(a=clue.b, b=clue.a), *p.clues[i + 1 :]]
    diagnosis = diagnose(p.spec, misread, list(p.clues), p.solution)
    assert diagnosis.wrong == (i + 1,)
    assert diagnosis.missed == (i + 1,)


@pytest.mark.parametrize("kind", [NextToClue, SameHouseClue, NotSameHouseClue])
def test_symmetric_clues_in_either_order_are_not_missed(kind: type) -> None:
    p = generate(*with_kind(kind))
    swapped: list[Clue] = [
        clue.model_copy(update={"a": clue.b, "b": clue.a}) if isinstance(clue, kind) else clue
        for clue in p.clues
    ]
    assert swapped != list(p.clues)
    diagnosis = diagnose(p.spec, swapped, list(p.clues), p.solution)
    assert (diagnosis.wrong, diagnosis.missed) == ((), ())


def test_left_of_is_not_symmetric() -> None:
    p = generate(*with_kind(LeftOfClue))
    clue = next(clue for clue in p.clues if isinstance(clue, LeftOfClue))
    assert canonical(LeftOfClue(a=clue.b, b=clue.a)) != canonical(clue)


def test_an_omission_is_missed_but_not_wrong() -> None:
    # Clue sets are minimal, so the other clues never entail the one left out.
    p = generate(2, 0)
    diagnosis = diagnose(p.spec, list(p.clues[1:]), list(p.clues), p.solution)
    assert (diagnosis.wrong, diagnosis.entailed, diagnosis.missed) == ((), (), (1,))


# A three-house street with one attribute, for hand-made translations.
STREET = Spec(houses=3, attributes=(("name", ("Alice", "Bob", "Carol")),))
IN_ORDER = Solution({"name": ("Alice", "Bob", "Carol")})
ALICE, BOB, CAROL = (Ref(attribute="name", value=name) for name in ("Alice", "Bob", "Carol"))


def test_an_inferred_but_true_clue_is_inferred_not_wrong() -> None:
    # The pilot's case: "the rabbit owner" rewritten as the name of the person who keeps it.
    p = generate(3, 0)
    i, clue = next(
        (i, clue)
        for i, clue in enumerate(p.clues)
        if isinstance(clue, SameHouseClue | NotSameHouseClue | LeftOfClue | NextToClue)
        and clue.a.attribute != "name"
    )
    assert isinstance(clue, SameHouseClue | NotSameHouseClue | LeftOfClue | NextToClue)
    column = p.solution.root[clue.a.attribute]
    holder = p.solution.root["name"][column.index(clue.a.value)]
    rewritten = clue.model_copy(update={"a": Ref(attribute="name", value=holder)})
    translated = [*p.clues[:i], rewritten, *p.clues[i + 1 :]]
    diagnosis = diagnose(p.spec, translated, list(p.clues), p.solution)
    assert diagnosis.inferred == (i + 1,)
    assert diagnosis.wrong == ()


def test_a_weakened_clue_is_inferred_and_its_clue_missed() -> None:
    # "directly left" written as "somewhere left": true, but it says less.
    stated = [ImmediatelyLeftClue(a=ALICE, b=BOB)]
    diagnosis = diagnose(STREET, [LeftOfClue(a=ALICE, b=BOB)], stated, IN_ORDER)
    assert (diagnosis.inferred, diagnosis.wrong) == ((1,), ())
    assert (diagnosis.captured, diagnosis.entailed, diagnosis.missed) == ((), (), (1,))


@pytest.mark.parametrize(
    ("translated", "stated"),
    [
        pytest.param(
            [AtClue(ref=CAROL, house=3)],
            [NotAtClue(ref=CAROL, house=1), NotAtClue(ref=CAROL, house=2)],
            id="at-for-two-not-ats",
        ),
        pytest.param(
            [LeftOfClue(a=ALICE, b=BOB), NextToClue(a=ALICE, b=BOB)],
            [ImmediatelyLeftClue(a=ALICE, b=BOB)],
            id="left-of-and-next-to-for-directly-left",
        ),
    ],
)
def test_an_equivalent_paraphrase_is_entailed(translated: list[Clue], stated: list[Clue]) -> None:
    diagnosis = diagnose(STREET, translated, stated, IN_ORDER)
    assert diagnosis.inferred == tuple(range(1, len(translated) + 1))
    assert diagnosis.entailed == tuple(range(1, len(stated) + 1))
    assert diagnosis.missed == diagnosis.wrong == ()


def test_a_contradictory_translation_entails_nothing() -> None:
    translated = [AtClue(ref=ALICE, house=1), AtClue(ref=ALICE, house=2)]
    diagnosis = diagnose(STREET, translated, [NotAtClue(ref=BOB, house=1)], IN_ORDER)
    assert diagnosis.wrong == (2,)
    assert (diagnosis.entailed, diagnosis.missed) == ((), (1,))


def test_translation_verifier_in_an_eval(tmp_path: Path) -> None:
    samples = logic_grid_samples([3], 4)

    def clues_of(sample: Sample) -> list[dict[str, object]]:
        clues: list[dict[str, object]] = (sample.metadata or {})["hidden"]["clues"]
        return clues

    def flipped(sample: Sample) -> list[dict[str, object]]:
        out = []
        for clue in clues_of(sample):
            if clue["kind"] in ("left_of", "immediately_left"):
                clue = {**clue, "a": clue["b"], "b": clue["a"]}
            out.append(clue)
        return out

    exact, misread, garbage, padded = samples
    # A malformed item: not_at in the internal {ref, house} shape, which the translation
    # format ({kind, a, b}) rejects.
    bad_item = {"kind": "not_at", "ref": {"attribute": "name", "value": "Bob"}, "house": 3}
    replies = {
        str(exact.input): as_translation(clues_of(exact)),
        str(misread.input): as_translation(flipped(misread)),
        str(garbage.input): "The answer is obviously house 2.",
        str(padded.input): json.dumps([*json.loads(as_translation(clues_of(padded))), bad_item]),
    }

    def respond(messages: list[ChatMessage], *_: object) -> ModelOutput:
        first_user = next(message.text for message in messages if message.role == "user")
        [reply] = [reply for text, reply in replies.items() if text in first_user]
        return ModelOutput.from_content("mockllm/model", reply)

    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "3", "n_seeds": 4, "strategy": "z3_translate"},
        model=get_model("mockllm/model", custom_outputs=respond),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success" and log.samples is not None
    by_id = {sample.id: sample.scores for sample in log.samples if sample.scores}

    exact_scores = by_id[exact.id]
    assert exact_scores["grid_verifier"].value["solved"] == 1.0
    assert exact_scores["translation_verifier"].value == {
        "translation_exact": 1.0,
        "status_unique": 1.0,
        "status_ambiguous": 0.0,
        "status_unsat": 0.0,
        "status_parse_error": 0.0,
    }

    misread_scores = by_id[misread.id]["translation_verifier"]
    assert misread_scores.metadata is not None
    assert misread_scores.metadata["n_wrong"] >= 1
    assert misread_scores.value["translation_exact"] == 0.0

    garbage_scores = by_id[garbage.id]
    assert garbage_scores["translation_verifier"].value["status_parse_error"] == 1.0
    assert garbage_scores["grid_verifier"].value["format_ok"] == 0.0

    padded_scores = by_id[padded.id]
    padded_translation = padded_scores["translation_verifier"]
    assert padded_scores["grid_verifier"].value["solved"] == 1.0  # the real clues still solve it
    assert padded_translation.value["status_unique"] == 1.0
    assert padded_translation.value["translation_exact"] == 0.0
    assert padded_translation.metadata is not None
    n_clues = len(clues_of(padded))
    assert padded_translation.metadata["malformed_items"] == [n_clues + 1]
    assert {key: padded_translation.metadata[key] for key in COUNTS} == {
        "n_exact": n_clues,
        "n_inferred": 0,
        "n_wrong": 0,
        "n_malformed": 1,
        "n_captured": n_clues,
        "n_entailed": 0,
        "n_missed": 0,
    }

    assert log.results is not None
    exact_mean = next(s for s in log.results.scores if s.name == "translation_exact")
    assert exact_mean.metrics["mean"].value == pytest.approx(1 / 4)


COUNTS = ("n_exact", "n_inferred", "n_wrong", "n_malformed", "n_captured", "n_entailed", "n_missed")
