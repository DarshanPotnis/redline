"""Misread diagnosis: wrong (false on the solution) and missed (true clue not translated)."""

import json
from pathlib import Path

import inspect_ai
import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ModelOutput, get_model

from redline.puzzles.logic_grid import generate
from redline.puzzles.logic_grid.model import (
    Clue,
    LeftOfClue,
    NextToClue,
    NotSameHouseClue,
    SameHouseClue,
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
    diagnosis = diagnose(list(p.clues), list(p.clues), p.solution)
    assert (diagnosis.wrong, diagnosis.missed) == ((), ())


def test_a_flipped_left_of_is_wrong_and_its_clue_missed() -> None:
    p = generate(*with_kind(LeftOfClue))
    i = next(i for i, clue in enumerate(p.clues) if isinstance(clue, LeftOfClue))
    clue = p.clues[i]
    assert isinstance(clue, LeftOfClue)
    misread = [*p.clues[:i], LeftOfClue(a=clue.b, b=clue.a), *p.clues[i + 1 :]]
    diagnosis = diagnose(misread, list(p.clues), p.solution)
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
    diagnosis = diagnose(swapped, list(p.clues), p.solution)
    assert (diagnosis.wrong, diagnosis.missed) == ((), ())


def test_left_of_is_not_symmetric() -> None:
    p = generate(*with_kind(LeftOfClue))
    clue = next(clue for clue in p.clues if isinstance(clue, LeftOfClue))
    assert canonical(LeftOfClue(a=clue.b, b=clue.a)) != canonical(clue)


def test_an_omission_is_missed_but_not_wrong() -> None:
    p = generate(2, 0)
    diagnosis = diagnose(list(p.clues[1:]), list(p.clues), p.solution)
    assert (diagnosis.wrong, diagnosis.missed) == ((), (1,))


def test_translation_verifier_in_an_eval(tmp_path: Path) -> None:
    samples = logic_grid_samples([3], 3)

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

    exact, misread, garbage = samples
    replies = {
        str(exact.input): json.dumps(clues_of(exact)),
        str(misread.input): json.dumps(flipped(misread)),
        str(garbage.input): "The answer is obviously house 2.",
    }

    def respond(messages: list[ChatMessage], *_: object) -> ModelOutput:
        first_user = next(message.text for message in messages if message.role == "user")
        [reply] = [reply for text, reply in replies.items() if text in first_user]
        return ModelOutput.from_content("mockllm/model", reply)

    [log] = inspect_ai.eval(
        logic_grid,
        task_args={"levels": "3", "n_seeds": 3, "strategy": "z3_translate"},
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

    assert log.results is not None
    exact_mean = next(s for s in log.results.scores if s.name == "translation_exact")
    assert exact_mean.metrics["mean"].value == pytest.approx(1 / 3)
