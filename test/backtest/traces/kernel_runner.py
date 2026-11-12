"""Runs a trace case through the kernel and returns its result in the fixture's shape.

The case is run for real: its calendar becomes a `Calendar`, its rows the bars of
`bt.backtest(data=...)`, its instruments `Instrument`s, its settings the run arguments,
and its scripted decisions the `on_bar` of a `Strategy` subclass ("script" form) or its
condition arrays the `entry_when`, `exit_when`, `short_when` and `cover_when` of a
condition strategy ("conditions" form, run by the Rust signal driver). The result is then
written with the keys and units of the case's `expected` section.

A rolled futures case (`settings.roll`) runs as the frozen contract has it: the strategy
names the underlying and the contract rank (its class attributes), the fixture's contracts
are given as `instrument=` with their expiry and underlying, and `roll=` is the case's
`sessions_before_expiry`. The contracts the run registered must be the case's `chain`.

A case that needs a part of the kernel that is not built yet is skipped with a reason
that names the part (`unsupported_reason`), so the trace suite reports exactly which
cases the kernel runs. Only test code lives here; nothing of it ships in the package.
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import Any

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._conditions import strategy_frame
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.reference import Instrument

from .kernel_hook import CONDITIONS_FORM
from .timegrid import Case, format_ist, parse_ist

# The calls the kernel runs today, with the arguments it accepts.
LEG_RULE_ARGUMENTS = {"stop", "target", "targets", "trail", "breakeven", "after_first_target"}
ORDER_PRICE_ARGUMENTS = {"price", "trigger_price", "validity"}
OPENING_ARGUMENTS = {"qty", "lots", "size", "reverse"} | LEG_RULE_ARGUMENTS | ORDER_PRICE_ARGUMENTS
SUPPORTED_ARGUMENTS = {"buy": OPENING_ARGUMENTS, "sell": OPENING_ARGUMENTS, "exit": set()}
CONDITION_METHODS = ("entry_when", "exit_when", "short_when", "cover_when")
FORWARDED_SETTINGS = (
    "base",
    "entry_from",
    "entry_until",
    "square_off",
    "fill",
    "capital",
    "quantity_mode",
    "round_levels",
    "max_daily_loss",
    "risk_basis",
    "after_risk_exit",
)


def unsupported_reason(case: Case, form: str) -> str | None:
    """Why the kernel cannot run this case yet, or None when it can. The conditions form
    runs the same order arguments as the script, so one check covers both forms."""
    if len({instrument["exchange"] for instrument in case["instruments"]}) != 1:
        return "runs across exchanges"
    for decision in case["strategy"]["decisions"]:
        call = decision["call"]
        allowed = SUPPORTED_ARGUMENTS.get(call["method"])
        if allowed is None or not set(call["arguments"]) <= allowed:
            return f"{decision['python']} (sizers or this order method)"
    return None


def _argument(value: Any) -> Any:
    """A fixture's call argument as the Python value: {"bt": name, "args": [...]} is the
    helper bt.<name>(*args) (a dotted name such as size.qty included), and a list is
    targets= as (level, fraction) tuples."""
    if isinstance(value, dict):
        helper = functools.reduce(getattr, value["bt"].split("."), bt)
        return helper(*(_argument(item) for item in value["args"]))
    if isinstance(value, list):
        return [tuple(_argument(part) for part in pair) for pair in value]
    return value


def _calendar(case: Case) -> Calendar:
    calendar = Calendar(builtin=False)
    first = parse_ist(case["settings"]["first_slice_ist"]).date()
    last = parse_ist(case["settings"]["last_slice_ist"]).date()
    for exchange, sessions in case["calendar"].items():
        listed = {row["date"]: row for row in sessions}
        day = first
        while day <= last:
            row = listed.get(day.isoformat())
            if row is None:
                calendar.add_closure(exchange, day)
            else:
                calendar.add_session(exchange, day, row["open_ist"], row["close_ist"])
            day += dt.timedelta(days=1)
    return calendar


def _frames(case: Case) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol, rows in case["bars"]["rows"].items():
        index = pd.DatetimeIndex([parse_ist(row[0]) for row in rows], name="timestamp")
        columns = ["open", "high", "low", "close", "volume"]
        frames[symbol] = pd.DataFrame([row[1:] for row in rows], index=index, columns=columns)
    return frames


def _instruments(case: Case) -> dict[str, Instrument]:
    return {
        row["symbol"]: Instrument(
            symbol=row["symbol"],
            exchange=row["exchange"],
            kind=row["type"],
            lot_size=row["lot_size_units"],
            tick_size=row["tick_size_rupees"],
            expiry=None
            if row["expiry_date"] is None
            else dt.date.fromisoformat(row["expiry_date"]),
            underlying=row["underlying"],
        )
        for row in case["instruments"]
    }


def _condition_met(condition: str, position: Any) -> bool:
    return {
        "any": True,
        "flat": position.is_flat,
        "long": position.is_long,
        "short": position.is_short,
    }[condition]


def _scripted_strategy(case: Case, log: dict[str, Any]) -> type:
    attributes = dict(case["strategy"]["class_attributes"])
    decisions = case["strategy"]["decisions"]

    def on_bar(self, bar):
        bar_open = format_ist(bar.time)
        log["bars"].append(
            {
                "bar_open_ist": bar_open,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
            }
        )
        for number, decision in enumerate(decisions):
            if decision["bar_open_ist"] != bar_open:
                continue
            met = _condition_met(decision["if_position"], self.position)
            call = decision["call"]
            arguments = {name: _argument(value) for name, value in call["arguments"].items()}
            order = getattr(self, call["method"])(**arguments) if met else None
            log["decisions"][number] = (met, order)

    attributes["on_bar"] = on_bar
    return type("ScriptedTrace", (bt.Strategy,), attributes)


def _bar_row(time: Any, bar: Any) -> dict[str, Any]:
    return {
        "bar_open_ist": format_ist(time.to_pydatetime()),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
    }


def _condition_strategy(case: Case, log: dict[str, Any]) -> type:
    """The case's conditions as a condition strategy: each array a Series indexed by its
    bar opens, declared with entry_when, exit_when, short_when, cover_when in init. The
    strategy bars logged are those the conditions are evaluated on."""
    attributes = dict(case["strategy"]["class_attributes"])
    conditions = case["strategy"]["conditions"]
    attributes["pyramiding"] = conditions["pyramiding"]
    opens = pd.DatetimeIndex([parse_ist(text) for text in conditions["bar_opens_ist"]])

    def init(self):
        frame = strategy_frame(self)
        log["bars"] = [_bar_row(time, bar) for time, bar in frame.iterrows()]
        for method in CONDITION_METHODS:
            declared = conditions[method]
            if declared is None:
                continue
            arguments = declared.get("order_arguments", {})
            values = pd.Series(declared["values"], index=opens)
            getattr(self, method)(values, **{k: _argument(v) for k, v in arguments.items()})

    attributes["init"] = init
    return type("ConditionTrace", (bt.Strategy,), attributes)


def run_trace_case(case: Case, strategy_form: str) -> dict[str, Any]:
    """Run one case through the kernel; the result in the shape of `expected`. Skips
    (pytest.skip) when the case needs a part of the kernel not built yet."""
    reason = unsupported_reason(case, strategy_form)
    if reason is not None:
        pytest.skip(f"kernel does not run this case yet: {reason}")
    settings = case["settings"]
    log: dict[str, Any] = {"bars": [], "decisions": {}}
    build = _condition_strategy if strategy_form == CONDITIONS_FORM else _scripted_strategy
    strategy = build(case, log)
    product = case["instruments"][0]["product"]
    result = bt.backtest(
        strategy,
        data=_frames(case),
        instrument=_instruments(case),
        calendar=_calendar(case),
        start=parse_ist(settings["first_slice_ist"]),
        end=parse_ist(settings["last_slice_ist"]),
        costs=settings["costs"],
        slippage=settings["slippage"],
        product=product,
        roll=None if settings["roll"] is None else settings["roll"]["sessions_before_expiry"],
        **{name: settings[name] for name in FORWARDED_SETTINGS},
    )
    _check_chain(case, result)
    return translate(case, result, log)


def _check_chain(case: Case, result: Any) -> None:
    """A rolled case must have registered exactly the contracts of its chain, in order."""
    roll = case["settings"]["roll"]
    if roll is None:
        return
    registered = [row["symbol"] for row in result.manifest["instruments"]]
    if registered != roll["chain"]:
        raise AssertionError(f"{case['id']}: the run traded {registered}, not {roll['chain']}")


def translate(case: Case, result: Any, log: dict[str, Any]) -> dict[str, Any]:
    """The kernel's result with the keys and units of a case's `expected` section."""
    assumptions = result.assumptions
    return {
        "assumptions": {key: assumptions[key] for key in case["expected"]["assumptions"]},
        "strategy_bars": log["bars"],
        "decisions": _decisions(case, log),
        "fills": [_fill(row) for row in result.fills.to_dict("records")],
        "levels": [_level(row) for row in result.positions.to_dict("records")],
        "events": [_event(result, event) for event in result.events.to_dict("records")],
        "stale_marks": [
            {
                "slice_ist": _ist(m["slice_ts"]),
                "instrument": m["symbol"],
                "mark_price_rupees": m["mark_rupees"],
            }
            for m in result.stale_marks.to_dict("records")
        ],
        "final_positions": _final_positions(case, result),
        "gross_pnl": {
            "realized_rupees": result.gross_pnl["realized"],
            "unrealized_rupees": result.gross_pnl["unrealized"],
            "total_rupees": result.gross_pnl["total"],
        },
    }


def _ist(ts_ns: int) -> str:
    return format_ist(pd.Timestamp(int(ts_ns), tz="UTC").to_pydatetime())


def _none(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def _order_id(number: Any) -> str | None:
    number = _none(number)
    return None if number is None else f"o{int(number)}"


def _gtt_id(number: Any) -> str | None:
    number = _none(number)
    return None if number is None else f"g{int(number)}"


def _decisions(case: Case, log: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for number in range(len(case["strategy"]["decisions"])):
        met, order = log["decisions"].get(number, (False, None))
        outcome = "no_order" if order is None else order.status.lower()
        rows.append(
            {
                "decision": number,
                "position_condition_met": met,
                "order_id": None if order is None else order.id,
                "outcome": outcome,
            }
        )
    return rows


def _fill(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slice_ist": _ist(row["slice_ts"]),
        "step": row["step"],
        "instrument": row["symbol"],
        "side": row["side"],
        "quantity_units": row["quantity_units"],
        "price_rupees": row["price_rupees"],
        "fill_at": row["fill_at"],
        "reason": row["reason"],
        "order_id": _order_id(row["order_id"]),
        "target_index": _none(row["target_index"]),
        "flags": list(row["flags"]),
        "gtt_id": _gtt_id(row["gtt_id"]),
    }


def _level(row: dict[str, Any]) -> dict[str, Any]:
    stop = _none(row["stop_rupees"])
    trail_armed = _none(row["trail_armed"])
    breakeven_armed = _none(row["breakeven_armed"])
    return {
        "slice_ist": _ist(row["slice_ts"]),
        "instrument": row["symbol"],
        "position_quantity_units": row["quantity_units"],
        "average_price_rupees": _none(row["average_price_rupees"]),
        "stop": None
        if stop is None
        else {"level_rupees": stop, "quantity_units": row["stop_units"]},
        "targets": [
            {
                "index": i,
                "level_rupees": level,
                "quantity_units": units,
                "state": "filled" if filled else "working",
            }
            for i, level, units, filled in row["targets"]
        ],
        "remainder_quantity_units": row["remainder_units"],
        "trail": None
        if trail_armed is None
        else {"armed": trail_armed, "favourable_extreme_rupees": row["trail_extreme_rupees"]},
        "breakeven": None
        if breakeven_armed is None
        else {
            "armed": breakeven_armed,
            "trigger_price_rupees": row["breakeven_trigger_rupees"],
            "level_rupees": row["breakeven_level_rupees"],
        },
    }


def _event(result: Any, event: dict[str, Any]) -> dict[str, Any]:
    head = {
        "slice_ist": _ist(event["slice_ts"]),
        "step": event["step"],
        "type": event["type"],
        "name": event["name"],
    }
    symbols = result.manifest["instruments"]
    if event["type"] == "order":
        return {
            **head,
            "order_id": _order_id(event["order_id"]),
            "quantity_units": _none(event["quantity_units"]),
            "price_rupees": _none(event["price_rupees"]),
            "reject_code": _none(event["reject_code"]),
        }
    index = _none(event["instrument"])
    symbol = None if index is None else symbols[int(index)]["symbol"]
    if event["type"] == "position":
        return {
            **head,
            "instrument": symbol,
            "position_quantity_units": event["position_quantity_units"],
        }
    if event["type"] == "gtt":
        return {
            **head,
            "gtt_id": _gtt_id(event["gtt_id"]),
            "instrument": symbol,
            "quantity_units": event["quantity_units"],
            "stop_trigger_rupees": _none(event["stop_trigger_rupees"]),
            "target_trigger_rupees": _none(event["target_trigger_rupees"]),
            "fired_leg": _none(event["fired_leg"]),
            "child_limit_rupees": _none(event["child_limit_rupees"]),
            "price_rupees": _none(event["price_rupees"]),
        }
    return {
        **head,
        "scope": event["scope"],
        "instrument": symbol,
        "level_rupees": _none(event["level_rupees"]),
        "target_index": _none(event["target_index"]),
        "observed_rupees": _none(event["observed_rupees"]),
    }


def _final_positions(case: Case, result: Any) -> list[dict[str, Any]]:
    held = {row["symbol"]: row for row in result.final_positions.to_dict("records")}
    rows = []
    for instrument in case["instruments"]:
        row = held.get(instrument["symbol"])
        quantity = 0.0 if row is None else row["quantity_units"]
        average = None if row is None else _none(row["average_price_rupees"])
        rows.append(
            {
                "instrument": instrument["symbol"],
                "quantity_units": quantity,
                "average_price_rupees": average,
            }
        )
    return rows
