"""Check the shape of a case's expected section against the fixture schema.

The expected section is what the kernel is compared with: the resolved assumptions, the
strategy bars delivered to on_bar, the outcome of every scripted decision, the fills,
the level timeline, the events, stale marks, final positions and the gross P&L.
"""

from __future__ import annotations

from typing import Any

from . import vocabulary as vocab
from .schema import Violations, each, key_violations, member_violation
from .timegrid import Case


def validate_expected(case: Case) -> Violations:
    """Shape of the expected section of one case."""
    name = case["id"]
    expected = case["expected"]
    problems = key_violations(expected, "expected", f"{name} expected")
    if problems:
        return problems
    problems += key_violations(expected["assumptions"], "assumptions", f"{name} assumptions")
    if expected["strategy_bars"] is not None:
        problems += each(expected["strategy_bars"], "strategy_bar", f"{name} strategy_bars")
    problems += _decision_violations(case, name)
    problems += _fill_violations(expected["fills"], name)
    problems += _level_violations(expected["levels"], name)
    problems += _event_violations(expected["events"], name)
    problems += each(expected["stale_marks"], "stale_mark", f"{name} stale_marks")
    problems += _closing_state_violations(case, name)
    return problems


def _decision_violations(case: Case, name: str) -> Violations:
    outcomes = case["expected"]["decisions"]
    problems = each(outcomes, "expected_decision", f"{name} expected.decisions")
    if problems:
        return problems
    scripted = len(case["strategy"]["decisions"])
    if [outcome["decision"] for outcome in outcomes] != list(range(scripted)):
        problems.append(f"{name}: expected.decisions must list the {scripted} decisions in order")
    for outcome in outcomes:
        where = f"{name}: expected decision {outcome['decision']}"
        problems += member_violation(outcome["outcome"], vocab.DECISION_OUTCOMES, "outcome", where)
        if (outcome["outcome"] == "no_order") != (outcome["order_id"] is None):
            problems.append(f"{where}: an order id is given exactly when an order was created")
    return problems


def _fill_violations(fills: list[dict[str, Any]], name: str) -> Violations:
    problems = each(fills, "fill", f"{name} fills")
    if problems:
        return problems
    for fill in fills:
        where = f"{name} {fill['slice_ist']}: fill"
        problems += member_violation(fill["side"], vocab.SIDES, "side", where)
        problems += member_violation(fill["fill_at"], vocab.FILL_AT, "fill_at", where)
        problems += member_violation(fill["reason"], vocab.FILL_REASONS, "reason", where)
        problems += _step_violation(fill["step"], where)
        for flag in fill["flags"]:
            problems += member_violation(flag, vocab.FILL_FLAGS, "flag", where)
        quantity = fill["quantity_units"]
        if not isinstance(quantity, int) or quantity <= 0:
            problems.append(f"{where}: quantity_units must be a positive whole number of units")
        if (fill["reason"] == "leg_target") != (fill["target_index"] is not None):
            problems.append(f"{where}: target_index is given exactly for a leg_target fill")
    return problems


def _step_violation(step: Any, where: str) -> Violations:
    if isinstance(step, int) and vocab.FIRST_STEP <= step <= vocab.LAST_STEP:
        return []
    return [f"{where}: step is {step!r}; the sequence has steps 1 to 8"]


def _level_violations(rows: list[dict[str, Any]], name: str) -> Violations:
    problems = each(rows, "level_row", f"{name} levels")
    if problems:
        return problems
    for row in rows:
        where = f"{name} {row['slice_ist']}: level row"
        for key, kind in (("stop", "stop"), ("trail", "trail"), ("breakeven", "breakeven")):
            if row[key] is not None:
                problems += key_violations(row[key], kind, f"{where} {key}")
        problems += each(row["targets"], "target", f"{where} targets")
        for target in row["targets"]:
            state = target.get("state")
            problems += member_violation(state, vocab.TARGET_STATES, "target state", where)
    return problems


def _event_violations(events: list[dict[str, Any]], name: str) -> Violations:
    problems: Violations = []
    for event in events:
        where = f"{name} {event.get('slice_ist')}: event {event.get('name')!r}"
        kind = vocab.EVENT_KEYS.get(event.get("type"))
        if kind is None:
            problems.append(f"{where}: type must be one of {sorted(vocab.EVENT_KEYS)}")
            continue
        found = key_violations(event, kind, where)
        problems += found
        if found:
            continue
        names = vocab.EVENT_NAMES[event["type"]]
        problems += member_violation(event["name"], names, "name", where)
        problems += _step_violation(event["step"], where)
        problems += _event_field_violations(event, where)
    return problems


def _event_field_violations(event: dict[str, Any], where: str) -> Violations:
    problems: Violations = []
    if event["type"] == "risk":
        problems += member_violation(event["scope"], vocab.RISK_SCOPES, "scope", where)
        if (event["scope"] == "leg") != (event["instrument"] is not None):
            problems.append(f"{where}: a leg event names its instrument, a session event none")
    if event["type"] == "order":
        rejected = event["name"] == "Rejected"
        if rejected != (event["reject_code"] is not None):
            problems.append(f"{where}: reject_code is given exactly for a Rejected event")
        if rejected:
            code = event["reject_code"]
            problems += member_violation(code, vocab.REJECT_CODES, "reject_code", where)
        if (event["name"] == "Filled") != (event["price_rupees"] is not None):
            problems.append(f"{where}: quantity and price are given exactly for a Filled event")
    return problems


def _closing_state_violations(case: Case, name: str) -> Violations:
    expected = case["expected"]
    problems = each(expected["final_positions"], "final_position", f"{name} final_positions")
    problems += key_violations(expected["gross_pnl"], "gross_pnl", f"{name} gross_pnl")
    if problems:
        return problems
    declared = [instrument["symbol"] for instrument in case["instruments"]]
    listed = [position["instrument"] for position in expected["final_positions"]]
    if listed != declared:
        problems.append(f"{name}: final_positions must list every instrument in declared order")
    return problems
