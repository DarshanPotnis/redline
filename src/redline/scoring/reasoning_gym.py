"""Reasoning Gym verifier: our own answer extraction, then RG's checker on what we extracted.

RG's checkers are lenient: knights-and-knaves gives full credit to "Let me think... <answer>",
and a wrong claim anywhere in the reasoning changes the score. So only the text after the last
"Answer:" line is scored, and a response without one fails the format. The item is rebuilt from
(task, config, seed), which the generator guarantees is the item the model saw.
"""

import math
import re

from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

from redline.puzzles.reasoning_gym import rg_dataset
from redline.scoring.harness import run_flags

# A line that starts with "Answer:", markdown around the label allowed ("**Answer:**", "# Answer:").
_ANSWER_LINE = re.compile(r"^[\s>#*_`]*answer[*_`]*\s*:(.*)$", re.IGNORECASE | re.MULTILINE)


def extract_answer(text: str) -> str | None:
    """The text after the last "Answer:" line, or None if there is no such line with content."""
    matches = _ANSWER_LINE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip().strip("*_`").strip()
    return answer or None


def chance_knights_knaves(n_people: int) -> float:
    """Expected RG score of a uniformly random role assignment that names everyone.

    RG scores knights-and-knaves as 1 when every role is right, else 0.3 + 0.7 * matching / n when
    at least one is, else 0. With each role a fair coin, the number right is Binomial(n, 1/2).
    """
    if n_people < 1:
        raise ValueError(f"n_people must be at least 1, got {n_people}")
    partial = sum(
        math.comb(n_people, right) * (0.3 + 0.7 * right / n_people) for right in range(1, n_people)
    )
    return (1 + partial) / float(2**n_people)


@scorer(
    metrics={
        "solved": [accuracy(), stderr()],
        "rg_score": [mean(), stderr()],
        "format_ok": [mean()],
        "truncated": [mean()],
        "context_limited": [mean()],
        "forced_answer": [mean()],
        "direct_overrun": [mean()],
    }
)
def rg_verifier() -> Scorer:
    """Scores the strategy's final text with RG's checker; truncation is judged on the final
    model call, as for the logic grid."""

    async def score(state: TaskState, target: Target) -> Score:
        hidden = state.metadata["hidden"]
        dataset = rg_dataset(hidden["task"], hidden["config"], hidden["seed"])
        item = dataset[0]
        if item["question"] != state.metadata["public"]["text"]:
            raise RuntimeError(
                f"{hidden['task']} seed {hidden['seed']} no longer rebuilds the question the model "
                f"saw (reasoning_gym {hidden.get('rg_version')} at generation)"
            )
        extracted = extract_answer(state.output.completion)
        rg_score = 0.0 if extracted is None else float(dataset.score_answer(extracted, item))
        flags = run_flags(state)
        if extracted is None:
            explanation = 'format error: no "Answer:" line'
        elif rg_score == 1.0:
            explanation = "solved"
        else:
            explanation = f"wrong: RG score {rg_score:.3f}"
        return Score(
            value={
                "format_ok": float(extracted is not None),
                "solved": float(rg_score == 1.0),
                "rg_score": rg_score,
                **flags.values(),
            },
            answer=extracted,
            explanation=explanation + flags.explanation_suffix(),
            metadata=flags.metadata(),
        )

    return score
