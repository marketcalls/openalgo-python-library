"""Consistency rules for the strategy script of a case. One function per rule.

A case scripts its strategy as decisions: at the close of a strategy bar, if the position
is in a given state, make one order call. A condition strategy also carries its boolean
arrays, and its decisions are the equivalent on_bar of 14-strategy-api section 4.1.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken.
"""

from __future__ import annotations

from typing import Any

from . import ledger
from . import vocabulary as vocab
from .bars import resample, rupees
from .timegrid import Case, bar_close, decision_slice, format_ist, parse_ist

Violations = list[str]


def _exchange(case: Case) -> str:
    """The strategy's exchange. Every Phase 1 trace trades one exchange."""
    return case["instruments"][0]["exchange"]


def render_value(value: Any) -> str:
    """Python source text of one call argument of a fixture.

    A {"bt": name, "args": [...]} object is a helper call such as bt.points(4); a list is
    a list of (level, fraction) tuples, the shape of targets=.
    """
    if isinstance(value, dict):
        arguments = ", ".join(render_value(argument) for argument in value["args"])
        return f"bt.{value['bt']}({arguments})"
    if isinstance(value, list):
        pairs = ("(" + ", ".join(render_value(part) for part in pair) + ")" for pair in value)
        return "[" + ", ".join(pairs) + "]"
    return repr(value)


def render_arguments(arguments: dict[str, Any]) -> str:
    """Keyword arguments of a call as Python source text."""
    return ", ".join(f"{name}={render_value(value)}" for name, value in arguments.items())


def python_text_matches_the_call(case: Case) -> Violations:
    """Rule: the readable `python` text of a decision is exactly its structured call."""
    problems = []
    for index, decision in enumerate(case["strategy"]["decisions"]):
        call = decision["call"]
        rendered = f"self.{call['method']}({render_arguments(call['arguments'])})"
        if rendered != decision["python"]:
            problems.append(
                f"{case['id']} {decision['bar_open_ist']}: decision {index} python text "
                f"{decision['python']!r} differs from its call {rendered!r}"
            )
    return problems


def decisions_sit_on_strategy_bar_closes(case: Case) -> Violations:
    """Rule: a decision is timed at the close of its strategy bar, in the base slice that
    closes at that instant."""
    problems = []
    exchange = _exchange(case)
    interval = case["settings"]["strategy_interval"]
    for index, decision in enumerate(case["strategy"]["decisions"]):
        bar_open = decision["bar_open_ist"]
        closes = format_ist(bar_close(case, exchange, bar_open, interval))
        slot = decision_slice(case, exchange, bar_open)
        if (decision["decision_time_ist"], decision["decision_slice_ist"]) != (closes, slot):
            problems.append(
                f"{case['id']} {bar_open}: decision {index} must be at {closes} in slice {slot}"
            )
    return problems


def _strategy_side(case: Case, slice_ist: str) -> str:
    """flat, long or short over all instruments of the strategy at the end of a slice."""
    held = sum(
        ledger.position_after(case, instrument["symbol"], slice_ist).quantity_units
        for instrument in case["instruments"]
    )
    return "flat" if held == 0 else ("long" if held > 0 else "short")


def position_conditions_follow_the_fills(case: Case) -> Violations:
    """Rule: a decision's position condition is met exactly when the fills up to its slice
    leave the position in that state, and only then is an order created."""
    problems = []
    outcomes = case["expected"]["decisions"]
    for decision, outcome in zip(case["strategy"]["decisions"], outcomes, strict=True):
        side = _strategy_side(case, decision["decision_slice_ist"])
        met = decision["if_position"] in ("any", side)
        where = f"{case['id']} {decision['decision_slice_ist']}: decision {outcome['decision']}"
        if outcome["position_condition_met"] != met:
            problems.append(
                f"{where}: position_condition_met is {outcome['position_condition_met']}, but "
                f"the decision requires {decision['if_position']} and the position is {side}"
            )
        if (outcome["outcome"] != "no_order") != met:
            problems.append(f"{where}: an order is created exactly when the condition is met")
    return problems


def filled_orders_fill_at_the_next_open(case: Case) -> Violations:
    """Rule: an order that fills does so once, in step 2, at the open of the first bar its
    instrument has after the decision slice."""
    problems = []
    outcomes = case["expected"]["decisions"]
    for decision, outcome in zip(case["strategy"]["decisions"], outcomes, strict=True):
        where = f"{case['id']} {decision['decision_slice_ist']}: order {outcome['order_id']}"
        fills = [
            fill for fill in case["expected"]["fills"] if fill["order_id"] == outcome["order_id"]
        ]
        if outcome["outcome"] != "filled":
            if outcome["order_id"] is not None and fills:
                problems.append(f"{where} is {outcome['outcome']} but has a fill")
            continue
        if len(fills) != 1:
            problems.append(f"{where} must have exactly one fill, it has {len(fills)}")
            continue
        fill = fills[0]
        later = [
            row[0]
            for row in case["bars"]["rows"][fill["instrument"]]
            if parse_ist(row[0]) > parse_ist(decision["decision_slice_ist"])
        ]
        expected_slice = later[0] if later else None
        placed = (expected_slice, vocab.STEP_FILLS_AT_OPEN, "open")
        if (fill["slice_ist"], fill["step"], fill["fill_at"]) != placed:
            problems.append(f"{where} must fill in step 2 at the open of {expected_slice}")
    return problems


def strategy_bars_are_the_aggregate_of_the_base_bars(case: Case) -> Violations:
    """Rule: the strategy bars handed to on_bar are the session-aligned aggregates of the
    base bars that exist, for every strategy bar that closes inside the run."""
    stated = case["expected"]["strategy_bars"]
    if stated is None:
        return []
    exchange = _exchange(case)
    settings = case["settings"]
    instrument = case["instruments"][0]["symbol"]
    run_end = bar_close(case, exchange, settings["last_slice_ist"], settings["base"])
    bars = resample(
        case, instrument, case["bars"]["rows"][instrument], settings["strategy_interval"]
    )
    aggregated = [
        (bar_open, bar.open, bar.high, bar.low, bar.close)
        for bar_open, bar in bars.items()
        if bar_close(case, exchange, bar_open, settings["strategy_interval"]) <= run_end
    ]
    listed = [
        (bar["bar_open_ist"], *(rupees(bar[key]) for key in ("open", "high", "low", "close")))
        for bar in stated
    ]
    if listed == aggregated:
        return []
    return [
        f"{case['id']} {settings['first_slice_ist']}: strategy_bars are not the "
        f"{settings['strategy_interval']} aggregates of the base bars: expected "
        f"{[tuple(map(str, bar)) for bar in aggregated]}"
    ]


def _equivalent_decisions(conditions: dict[str, Any]) -> list[tuple[str, str, str, Any]]:
    """The on_bar of 14-strategy-api 4.1 for entry_when and exit_when with pyramiding 0:
    'if flat and entry: buy(**entry_args)' and 'if long and exit: exit()', bar by bar."""
    entry, exit_ = conditions["entry_when"], conditions["exit_when"]
    decisions = []
    for index, bar_open in enumerate(conditions["bar_opens_ist"]):
        if entry is not None and entry["values"][index]:
            decisions.append((bar_open, "flat", "buy", entry["order_arguments"]))
        if exit_ is not None and exit_["values"][index]:
            decisions.append((bar_open, "long", "exit", {}))
    return decisions


def condition_arrays_equal_the_scripted_decisions(case: Case) -> Violations:
    """Rule: for a condition strategy, the scripted decisions are the equivalent on_bar of
    its condition arrays."""
    conditions = case["strategy"]["conditions"]
    if conditions is None:
        return []
    where = f"{case['id']} {case['settings']['first_slice_ist']}: conditions"
    if conditions["pyramiding"] or conditions["short_when"] or conditions["cover_when"]:
        return [f"{where} use pyramiding or the short side, which this rule does not cover"]
    scripted = [
        (
            decision["bar_open_ist"],
            decision["if_position"],
            decision["call"]["method"],
            decision["call"]["arguments"],
        )
        for decision in case["strategy"]["decisions"]
    ]
    problems = []
    if scripted != _equivalent_decisions(conditions):
        problems.append(f"{where} do not reduce to the scripted decisions under 14 section 4.1")
    entry = conditions["entry_when"]
    if entry is not None:
        entry_text = f"self.entry_when(entry_cond, {render_arguments(entry['order_arguments'])})"
        if entry["python"] != entry_text:
            problems.append(f"{where}: entry_when python text differs from its order arguments")
    return problems
