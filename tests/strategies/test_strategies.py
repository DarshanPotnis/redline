"""Strategies see only the public puzzle: enforced on the source (AST) and on their output."""

import ast
from pathlib import Path

import pytest

import redline.strategies
from redline.puzzles.logic_grid import generate
from redline.strategies import STRATEGIES, get_strategy
from redline.strategies.common import RULES
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
    ("name", "instruction"), [("direct", DIRECT_INSTRUCTION), ("cot", COT_INSTRUCTION)]
)
def test_messages_are_the_rules_then_the_puzzle(name: str, instruction: str) -> None:
    public = generate(2, 3).to_instance().public()
    system, user = get_strategy(name)(public)
    assert (system.role, system.content) == ("system", RULES)
    assert user.role == "user"
    assert user.content == f"{public.text}\n\n{public.answer_format}\n\n{instruction}"


def test_strategies_are_pure() -> None:
    public = generate(1, 0).to_instance().public()
    for build in STRATEGIES.values():
        first, second = build(public), build(public)
        assert [(m.role, m.content) for m in first] == [(m.role, m.content) for m in second]


def test_unknown_strategy_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="choose from cot, direct"):
        get_strategy("peek")
