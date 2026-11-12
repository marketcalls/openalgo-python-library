"""What a user reads: every message says plainly what is wrong or not supported and what
to do, never an internal planning reference the user cannot look up (a requirement,
decision or rule id, a phase number or a section of a design document), and a count of
one never reads as a plural.

The not-built errors are reached through the public API, one row each; the string
literals of the package and of the kernel (Rust code that reaches Python) are scanned
so that no new message brings a reference back.
"""

import ast
import datetime as dt
import re
from pathlib import Path

import pytest
from strategy_samples import daily_bars, run

import openalgo.backtest as bt
from openalgo.backtest._messages import counted

ROOT = Path(__file__).resolve().parents[2]
# Internal references: requirement ids (FR-MKT-11, NFR-PERF-04), decision and rule ids
# (SPEC-27, LEAD-04, GTT-2, TIMES-1, D18), phase numbers, "work in progress", and
# sections of the design documents ("14-strategy-api 4.1", "PRD 06").
PLANNING = re.compile(
    r"\bN?FR-[A-Z]+-\d+|\bSPEC-\d+|\bLEAD-\d+|\bGTT-\d{1,2}\b|\bTIMES-\d+|\bD\d{1,2}\b"
    r"|[Pp]hase \d|work in progress|\b\d{2}-[a-z]+(?:-[a-z]+)* \d|\bPRD\b"
)
NOT_SUPPORTED = " is not supported in this version: "
DAYS = [dt.date(2025, 7, day) for day in (1, 2, 3, 4, 7)]


class Quarterly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"


class Weekly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "W"


class Universe(bt.Strategy):
    universe, exchange, interval = "NIFTY50", "NSE", "15m"


class MixedFutures(bt.Strategy):
    symbols, exchange, interval = ("NIFTY", "NIFTY24NOV26FUT"), "NFO", "D"


def _with(**attributes) -> type:
    return type("Variant", (Quarterly,), attributes)


def _result() -> bt.Result:
    return run(Quarterly)


NOT_BUILT = {
    "bt.live": lambda: bt.live(Quarterly, None, capital=100_000, max_daily_loss=1_000),
    "bt.walk_forward": lambda: bt.walk_forward(Quarterly),
    "bt.rebalance": lambda: bt.rebalance(),
    "several strategies": lambda: bt.backtest([Quarterly, Quarterly], capital=100_000),
    "Result.strategy": lambda: _result().strategy("Quarterly"),
    "Result.combined": lambda: _result().combined,
    "Result.correlation": lambda: _result().correlation,
    "pair_long": lambda: Quarterly().pair_long(),
    "pair_short": lambda: Quarterly().pair_short(),
    "universe": lambda: run(Universe),
    "cooldown as a time": lambda: run(_with(cooldown="30m")),
    "combined rule": lambda: run(_with(lock_profit=bt.rupees(5_000))),
    "after_risk_exit": lambda: run(Quarterly, after_risk_exit="allow_reentry"),
    "fill close": lambda: run(Quarterly, fill="close"),
    "weekly strategy interval": lambda: run(Weekly, data=daily_bars(DAYS)),
    "mixed futures symbols": lambda: run(MixedFutures, start=DAYS[0], end=DAYS[-1]),
}


@pytest.mark.parametrize("call", NOT_BUILT.values(), ids=NOT_BUILT.keys())
def test_a_part_not_built_says_what_to_do_instead_without_planning_references(call):
    with pytest.raises(NotImplementedError) as caught:
        call()
    message = str(caught.value)
    feature, _, instead = message.partition(NOT_SUPPORTED)
    assert feature and instead, message
    assert PLANNING.search(message) is None, message


def _python_literals(path: Path) -> list[tuple[int, str]]:
    """The string constants of a module that are not docstrings, with their lines."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_message_string_of_the_package_cites_a_planning_reference():
    found = [
        f"{path.relative_to(ROOT)}:{line}: {text!r}"
        for path in sorted((ROOT / "openalgo" / "backtest").rglob("*.py"))
        for line, text in _python_literals(path)
        if PLANNING.search(text)
    ]
    assert found == []


# A Rust string literal: a quoted run of escapes and other characters. Comments are
# removed first (`//` to the end of the line, outside a literal).
RUST_LITERAL = re.compile(r'"((?:\\.|[^"\\])*)"', re.DOTALL)
RUST_COMMENT = re.compile(r'//[^\n]*|("(?:\\.|[^"\\])*")', re.DOTALL)


def _rust_literals(path: Path) -> list[str]:
    code = RUST_COMMENT.sub(lambda match: match.group(1) or "", path.read_text(encoding="utf-8"))
    # A line continuation inside a literal joins the two lines with no break.
    return [re.sub(r"\\\n\s*", "", text) for text in RUST_LITERAL.findall(code)]


def _kernel_sources() -> list[Path]:
    crates = (ROOT / "rust" / "oa_bt" / "src", ROOT / "rust" / "oa_py" / "src")
    return [
        path
        for crate in crates
        for path in sorted(crate.rglob("*.rs"))
        if "tests" not in path.relative_to(crate).parts and path.name != "tests.rs"
    ]


def test_no_message_string_of_the_kernel_cites_a_planning_reference():
    found = [
        f"{path.relative_to(ROOT)}: {text!r}"
        for path in _kernel_sources()
        for text in _rust_literals(path)
        if PLANNING.search(text)
    ]
    assert found == []


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "0 rows have"), (1, "1 row has"), (2, "2 rows have")],
)
def test_a_count_takes_the_singular_only_for_one(count, expected):
    assert counted(count, "row has", "rows have") == expected


def test_a_count_takes_an_s_when_no_plural_is_given():
    assert (counted(1, "session"), counted(3, "session")) == ("1 session", "3 sessions")
