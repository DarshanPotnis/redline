"""Strategies see only the public puzzle: enforced on the source (AST) and on their output."""

import ast
import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput

import redline.strategies
from redline.puzzles.base import PublicPuzzle
from redline.puzzles.logic_grid import generate
from redline.puzzles.reasoning_gym import KNIGHTS_KNAVES
from redline.strategies import STRATEGIES, cot, direct, get_strategy
from redline.strategies.base import BudgetedModel
from redline.strategies.common import FORCE_ANSWER, PROMPTS, RULES
from redline.strategies.cot import INSTRUCTION as COT_INSTRUCTION
from redline.strategies.direct import INSTRUCTION as DIRECT_INSTRUCTION

PACKAGE = Path(redline.strategies.__file__).parent
BANNED = {"target", "metadata", "hidden", "solution", "TaskState"}


def offenses(source: str, filename: str) -> list[str]:
    """Answer-bearing names (as identifiers, or as exact string keys) and scoring imports."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        names: list[str] = []
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name) | ast.arg(arg=name):
                names = [name]
            case ast.keyword(arg=str() as name):
                names = [name]
            case ast.FunctionDef(name=name) | ast.ClassDef(name=name):
                names = [name]
            case ast.alias(name=name, asname=asname):
                names = [*name.split("."), *([asname] if asname else [])]
            case ast.Constant(value=str() as text):
                names = [text]
        found += [f"{filename}:{getattr(node, 'lineno', '?')}: {n}" for n in names if n in BANNED]
        if isinstance(node, ast.ImportFrom) and _imports_scoring(node):
            found.append(f"{filename}:{node.lineno}: imports scoring")
        if isinstance(node, ast.Import) and any(
            a.name.startswith("redline.scoring") for a in node.names
        ):
            found.append(f"{filename}:{node.lineno}: imports scoring")
    return found


def _imports_scoring(node: ast.ImportFrom) -> bool:
    if node.level == 0:
        return (node.module or "").startswith("redline.scoring")
    # Relative imports resolve against redline.strategies: `from ..scoring import x`.
    base = ["redline", "strategies"][: 2 - (node.level - 1)]
    return ".".join([*base, *(node.module or "").split(".")]).startswith("redline.scoring")


def test_strategy_modules_never_touch_answer_bearing_names() -> None:
    modules = sorted(PACKAGE.glob("*.py"))
    assert {"__init__.py", "common.py", "cot.py", "direct.py"} <= {m.name for m in modules}
    found = [o for module in modules for o in offenses(module.read_text(), module.name)]
    assert found == []


def test_the_ast_check_catches_violations() -> None:
    source = (
        "from redline.scoring.logic_grid import grade\n"
        "from ..scoring import grid_verifier\n"
        "import redline.scoring\n"
        "def build(state):\n"
        "    return state.metadata['public'], getattr(state, 'target')\n"
        "def peek(task_state: TaskState, hidden=None): ...\n"
    )
    found = offenses(source, "bad.py")
    assert sum("imports scoring" in o for o in found) == 3
    assert {o.rsplit(": ", 1)[1] for o in found} >= {"metadata", "target", "TaskState", "hidden"}


@pytest.mark.parametrize(
    ("build", "instruction"),
    [(direct.build_messages, DIRECT_INSTRUCTION), (cot.build_messages, COT_INSTRUCTION)],
)
def test_messages_are_the_rules_then_the_puzzle(
    build: Callable[[PublicPuzzle], list[ChatMessage]], instruction: str
) -> None:
    public = generate(2, 3).to_instance().public()
    system, user = build(public)
    assert (system.role, system.content) == ("system", RULES)
    assert user.role == "user"
    assert user.content == f"{public.text}\n\n{public.answer_format}\n\n{instruction}"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategies_are_pure(name: str) -> None:
    # Same public puzzle and same model replies give the same calls and the same result.
    public = generate(1, 0).to_instance().public()

    async def reply(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        return ModelOutput.from_content("fake", "[]")

    runs = []
    for _ in range(2):
        model = BudgetedModel(reply)
        result = asyncio.run(get_strategy(name)(public, model))
        runs.append((result, [call.messages for call in model.calls]))
    assert runs[0] == runs[1]


def test_unknown_strategy_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="choose from cot, direct, z3_translate"):
        get_strategy("peek")


def test_the_logic_grid_prompts_are_unchanged_since_1c() -> None:
    grid = PROMPTS["logic_grid"]
    assert grid.direct == DIRECT_INSTRUCTION == "Respond with only the JSON."
    assert (
        grid.cot
        == COT_INSTRUCTION
        == "Reason step by step, then give the final answer as JSON at the end."
    )
    assert grid.force == FORCE_ANSWER and grid.rules == RULES
    assert FORCE_ANSWER.startswith("You have run out of reasoning budget.")


@pytest.mark.parametrize(
    ("build", "field"), [(direct.build_messages, "direct"), (cot.build_messages, "cot")]
)
def test_reasoning_gym_puzzles_get_their_own_wording(
    build: Callable[[PublicPuzzle], list[ChatMessage]], field: str
) -> None:
    public = KNIGHTS_KNAVES.generate(2, 0).public()
    system, user = build(public)
    rg = PROMPTS["reasoning_gym"]
    assert system.content == rg.rules
    assert user.content == f"{public.text}\n\n{public.answer_format}\n\n{getattr(rg, field)}"
    for text in (rg.rules, rg.direct, rg.cot, rg.force):
        assert "JSON" not in text and "house" not in text


# What each family's answer format is called. A new family must add itself here, so the checks
# below cover it.
ANSWER_MARKERS = {"logic_grid": "JSON", "reasoning_gym": "Answer:"}


def test_every_family_has_an_answer_marker() -> None:
    assert set(PROMPTS) == set(ANSWER_MARKERS)


@pytest.mark.parametrize("family", sorted(PROMPTS))
def test_direct_means_no_reasoning_in_every_family(family: str) -> None:
    # A strategy must mean the same thing on every family, or cross-family comparisons are void:
    # pilot-1f's knights-and-knaves direct reasoned for a median of 2,000-3,600 tokens.
    assert "Respond with only" in PROMPTS[family].direct


def test_the_reasoning_gym_direct_prompt_rules_out_reasoning() -> None:
    assert PROMPTS["reasoning_gym"].direct == "Respond with only that line, no reasoning."


@pytest.mark.parametrize("family", sorted(PROMPTS))
def test_every_forcing_message_names_its_answer_format(family: str) -> None:
    assert ANSWER_MARKERS[family] in PROMPTS[family].force


def test_an_unknown_family_is_an_error() -> None:
    public = PublicPuzzle(generator="sudoku/9x9", text="?", answer_format="?", structure={})
    with pytest.raises(ValueError, match="no prompts for puzzle family 'sudoku'"):
        direct.build_messages(public)


def test_z3_translate_refuses_other_families() -> None:
    public = KNIGHTS_KNAVES.generate(1, 0).public()

    async def reply(messages: list[ChatMessage], config: GenerateConfig) -> ModelOutput:
        raise AssertionError("no model call should happen")

    with pytest.raises(ValueError, match="z3_translate translates logic-grid clues"):
        asyncio.run(get_strategy("z3_translate")(public, BudgetedModel(reply)))
