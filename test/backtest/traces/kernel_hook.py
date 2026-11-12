"""Where the kernel plugs into the worked traces, and the comparison with a fixture.

The kernel does not exist yet. When it does, it provides one function:

    openalgo.backtest.trace_runner.run_trace_case(case, strategy_form) -> dict

`case` is one case of a fixture file as `fixture_loader.load_cases` returns it (derived
bars resolved). `strategy_form` is "script" (run `strategy.decisions` as an on_bar
strategy) or "conditions" (run `strategy.conditions` as a condition strategy; only for
cases that carry conditions). The function runs the case with the declared instruments,
calendar, settings and bars, and returns a dict with the keys of the case's `expected`
section, in the same shapes and units. `compare_with_expected` then lists every
difference.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any

from . import vocabulary as vocab
from .timegrid import Case

RUNNER_MODULE = "openalgo.backtest.trace_runner"
RUNNER_FUNCTION = "run_trace_case"
KERNEL_MISSING_REASON = "kernel not implemented"

# Prices and amounts are rupees with at most two decimals; anything closer than this is
# the same number.
RUPEE_TOLERANCE = 1e-6

SCRIPT_FORM = "script"
CONDITIONS_FORM = "conditions"

Runner = Callable[[Case, str], dict[str, Any]]
Differences = list[str]


def find_kernel_runner() -> Runner | None:
    """The kernel's trace runner, or None while the runner module does not exist.

    Only the absence of the runner module itself counts as "no kernel"; an import error
    raised inside an existing runner module propagates, so a broken kernel cannot pass
    as a missing one.
    """
    try:
        module = importlib.import_module(RUNNER_MODULE)
    except ModuleNotFoundError as error:
        if error.name == RUNNER_MODULE:
            return None
        raise
    return getattr(module, RUNNER_FUNCTION)


def strategy_forms(case: Case) -> tuple[str, ...]:
    """The forms a case must be run in: always the script, and the conditions if present."""
    if case["strategy"]["conditions"] is None:
        return (SCRIPT_FORM,)
    return (SCRIPT_FORM, CONDITIONS_FORM)


def comparable_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kernel events restricted to the names the fixtures compare (SPEC-21)."""
    return [
        event
        for event in events
        if event.get("name") in vocab.EVENT_NAMES.get(event.get("type"), ())
    ]


def compare_with_expected(case: Case, actual: dict[str, Any], strategy_form: str) -> Differences:
    """Every difference between a kernel result and the expected section of a case.

    Scripted decisions are compared only for the script form, strategy bars only when
    the fixture states them, and kernel events only for the compared names. A difference
    on an expectation that depends on an open specification point names that point.
    """
    expected = case["expected"]
    differences: Differences = []
    for section, wanted in expected.items():
        if section == "strategy_bars" and wanted is None:
            continue
        if section == "decisions" and strategy_form != SCRIPT_FORM:
            continue
        got = actual.get(section)
        if section == "events" and got is not None:
            got = comparable_events(got)
        differences += _differences(wanted, got, f"{case['id']} {section}")
    return differences


def _differences(wanted: Any, got: Any, where: str) -> Differences:
    if isinstance(wanted, dict):
        return _dict_differences(wanted, got, where)
    if isinstance(wanted, list):
        return _list_differences(wanted, got, where)
    if _same_scalar(wanted, got):
        return []
    return [f"{where}: expected {wanted!r}, the kernel gave {got!r}"]


def _same_scalar(wanted: Any, got: Any) -> bool:
    numbers = (int, float)
    if isinstance(wanted, bool) or isinstance(got, bool):
        return wanted is got
    if isinstance(wanted, numbers) and isinstance(got, numbers):
        return math.isclose(wanted, got, rel_tol=0.0, abs_tol=RUPEE_TOLERANCE)
    return wanted == got


def _dict_differences(wanted: dict[str, Any], got: Any, where: str) -> Differences:
    if not isinstance(got, dict):
        return [f"{where}: expected an object, the kernel gave {got!r}"]
    tags = wanted.get(vocab.DEPENDS_ON, [])
    slice_ist = wanted.get("slice_ist")
    label = where if slice_ist is None else f"{where} at {slice_ist}"
    differences: Differences = []
    for key, value in wanted.items():
        if key != vocab.DEPENDS_ON:
            differences += _differences(value, got.get(key), f"{label} {key}")
    for key in sorted(set(got) - set(wanted)):
        differences.append(f"{label}: the kernel gave the unknown key {key!r}")
    if tags:
        differences = [f"{text} (depends on {', '.join(tags)})" for text in differences]
    return differences


def _list_differences(wanted: list[Any], got: Any, where: str) -> Differences:
    if not isinstance(got, list):
        return [f"{where}: expected a list, the kernel gave {got!r}"]
    differences: Differences = []
    if len(wanted) != len(got):
        differences.append(f"{where}: expected {len(wanted)} entries, the kernel gave {len(got)}")
    for index, (wanted_item, got_item) in enumerate(zip(wanted, got, strict=False)):
        differences += _differences(wanted_item, got_item, f"{where}[{index}]")
    return differences
