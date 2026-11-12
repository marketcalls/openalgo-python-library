"""Resolving the settings of a run: the documented defaults, the strategy's class
attributes and the run arguments, into one `RunPlan` the kernel receives explicitly.

Precedence (lead ruling): a strategy class attribute wins over the run argument of the
same name, which wins over the documented default. The documented defaults are those of
03-architecture 4.1 and 04-execution 2: product MIS for intraday intervals, CNC for daily
equity and NRML for daily derivatives; entry window and square-off from the calendar's
dated table (MIS only, SPEC-32); fill at the next open; fractional quantities; Zerodha
charges; the per-segment slippage. Every value used is listed in the Assumptions block.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._calendar_convert import interval_nanos, to_day, to_seconds
from ._layout import NANOS_PER_SECOND, parse_interval
from ._series import IST
from .errors import ConfigError
from .units import Distance

DERIVATIVE_EXCHANGES = ("NFO", "BFO", "MCX", "CDS", "BCD")
FILL_MODES = ("next_open", "close")
QUANTITY_MODES = ("fractional", "venue")
RISK_BASES = ("gross", "net")
AFTER_RISK_EXIT = ("lock_session", "allow_reentry")
# Run settings a strategy class may override (lead ruling on several strategies).
CLASS_OVERRIDES = (
    "max_daily_loss",
    "entry_from",
    "entry_until",
    "square_off",
    "drawdown_halt",
    "cooldown",
)
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class RunPlan:
    """Every resolved setting of one run, in the units the kernel takes."""

    symbols: tuple[str, ...]
    exchange: str
    interval: str
    base: str
    product: str
    start: dt.date
    end: dt.date
    first_slice: int | None
    last_slice: int | None
    capital: float
    fill: str
    quantity_mode: str
    costs: str | None
    slippage: tuple[str, float, int]
    risk: dict[str, Any]
    entry_from: str | None
    entry_until: str | None
    square_off: str | None
    warnings: tuple[str, ...]

    def native_config(self, deliver_events: bool) -> dict[str, Any]:
        """The config dict of the native engine."""
        strategy = interval_nanos(self.interval) if self.interval != "D" else "D"
        minutes = {
            f"{name}_minute": None
            if value is None
            else to_seconds(value, what=name) // SECONDS_PER_MINUTE
            for name, value in (
                ("entry_from", self.entry_from),
                ("entry_until", self.entry_until),
                ("square_off", self.square_off),
            )
        }
        return {
            "first_day": to_day(self.start),
            "last_day": to_day(self.end),
            "first_slice": self.first_slice,
            "last_slice": self.last_slice,
            "base_nanos": interval_nanos(self.base),
            "strategy_interval": strategy,
            "fill": self.fill,
            "quantity_mode": self.quantity_mode,
            "capital": self.capital,
            "costs": self.costs,
            "slippage": self.slippage,
            "risk": self.risk,
            "deliver_events": deliver_events,
            **minutes,
        }


def _choice(value: Any, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{name} takes one of {allowed}, got {value!r}")
    return value


def _slippage(value: Any) -> tuple[str, float, int]:
    if value == "default":
        return ("segment_default", 0.0, 0)
    if isinstance(value, int | float) and not isinstance(value, bool) and value == 0:
        return ("zero", 0.0, 0)
    if isinstance(value, Distance) and value.unit in ("ticks", "pct"):
        kind = "ticks" if value.unit == "ticks" else "percent"
        return (kind, value.value, 0)
    raise ConfigError("slippage takes 'default', 0, bt.ticks(n) or bt.pct(p)")


def _instant(value: Any) -> int | None:
    """UTC nanoseconds of a datetime bound (naive is IST), or None for a plain date."""
    if isinstance(value, dt.datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=IST)
        return int(pd.Timestamp(moment).value)
    return None


def _frame_range(data: Any) -> tuple[dt.date, dt.date]:
    """The IST dates a frame of own data covers. Raises ConfigError when unknown."""
    frames = list(data.values()) if isinstance(data, dict) else [data]
    stamps: list[pd.Timestamp] = []
    for frame in frames:
        if isinstance(frame, pd.DataFrame) and "ts_open" in frame.columns:
            stamps += [pd.Timestamp(int(frame["ts_open"].min()), tz="UTC")]
            stamps += [pd.Timestamp(int(frame["ts_open"].max()), tz="UTC")]
        elif isinstance(frame, pd.DataFrame) and getattr(frame.index, "tz", None) is not None:
            stamps += [frame.index.min(), frame.index.max()]
        else:
            raise ConfigError("pass start= and end= (the data has no tz-aware timestamps)")
    days = [stamp.tz_convert(IST).date() for stamp in stamps]
    return min(days), max(days)


def _dates(start: Any, end: Any, data: Any) -> tuple[dt.date, dt.date]:
    if start is None or end is None:
        if data is None:
            raise ConfigError("pass start= and end= (IST dates)")
        first, last = _frame_range(data)
        start = first if start is None else start
        end = last if end is None else end
    first_day = pd.Timestamp(start).date() if not isinstance(start, dt.date) else start
    last_day = pd.Timestamp(end).date() if not isinstance(end, dt.date) else end
    first_day = first_day.date() if isinstance(first_day, dt.datetime) else first_day
    last_day = last_day.date() if isinstance(last_day, dt.datetime) else last_day
    if first_day > last_day:
        raise ConfigError(f"start {first_day} is after end {last_day}")
    return first_day, last_day


def _product(strategy: type, product: str | None, interval: str, exchange: str) -> str:
    chosen = strategy.product or product
    if chosen is not None:
        return _choice(chosen, ("CNC", "MIS", "NRML"), "product")
    if parse_interval(interval)[0] == "intraday":
        return "MIS"
    return "NRML" if exchange in DERIVATIVE_EXCHANGES else "CNC"


def _base(base: str | None, interval: str, warnings: list[str]) -> str:
    if base is not None:
        parse_interval(base)
        return base
    if parse_interval(interval)[0] != "intraday":
        return "D"
    warnings.append(
        f"base interval {interval}: the strategy interval is used because choosing the "
        "1-minute base by data coverage is not implemented yet; pass base='1m' for a "
        "1-minute base"
    )
    return interval


def _risk(args: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_daily_loss": overrides["max_daily_loss"],
        "max_positions": args["max_positions"],
        "drawdown_halt": overrides["drawdown_halt"],
        "cooldown_bars": overrides["cooldown"],
        "risk_basis": _choice(args["risk_basis"], RISK_BASES, "risk_basis"),
        "after_risk_exit": _choice(args["after_risk_exit"], AFTER_RISK_EXIT, "after_risk_exit"),
        "round_levels": bool(args["round_levels"]),
        "strict_config": bool(args["strict_config"]),
    }


def resolve(strategy: type, args: dict[str, Any]) -> RunPlan:
    """The plan of a run of `strategy` with the run arguments `args` (the keyword
    arguments of `bt.backtest`). Raises ConfigError for unusable values and
    NotImplementedError for settings of later work."""
    if args["roll"] is not None:
        raise NotImplementedError("futures roll is not implemented yet (Phase 1 work in progress)")
    symbols = strategy.traded_symbols()
    if not strategy.exchange or not strategy.interval:
        raise ConfigError(f"{strategy.__name__}: set exchange and interval on the class")
    parse_interval(strategy.interval)
    capital = args["capital"]
    if isinstance(capital, bool) or not isinstance(capital, int | float) or capital <= 0:
        raise ConfigError("capital must be a positive number of rupees")
    warnings: list[str] = []
    if args["costs"] is None:
        warnings.append("costs=None: no charges are modelled; results overstate P&L")
    overrides = {
        name: getattr(strategy, name) if getattr(strategy, name) is not None else args[name]
        for name in CLASS_OVERRIDES
    }
    start, end = _dates(args["start"], args["end"], args["data"])
    product = _product(strategy, args["product"], strategy.interval, strategy.exchange)
    return RunPlan(
        symbols=symbols,
        exchange=strategy.exchange,
        interval=strategy.interval,
        base=_base(args["base"], strategy.interval, warnings),
        product=product,
        start=start,
        end=end,
        first_slice=_instant(args["start"]),
        last_slice=_instant(args["end"]),
        capital=float(capital),
        fill=_choice(args["fill"], FILL_MODES, "fill"),
        quantity_mode=_choice(args["quantity_mode"], QUANTITY_MODES, "quantity_mode"),
        costs=args["costs"],
        slippage=_slippage(args["slippage"]),
        risk=_risk(args, overrides),
        entry_from=overrides["entry_from"],
        entry_until=overrides["entry_until"],
        square_off=overrides["square_off"],
        warnings=tuple(warnings),
    )


def assumptions(plan: RunPlan, calendar: Any) -> dict[str, Any]:
    """The Assumptions block: what the run used. Entry window and square-off apply to
    MIS only (SPEC-32); their defaults are the calendar's for the first session."""
    block: dict[str, Any] = {"base": plan.base, "product": plan.product}
    entry_until = square_off = None
    if plan.product == "MIS":
        times = calendar.intraday_times(plan.exchange, plan.start)
        entry_until = plan.entry_until or (times and _clock(times.entry_until))
        square_off = plan.square_off or (times and _clock(times.square_off))
    block.update(entry_until=entry_until, square_off=square_off)
    block.update(
        fill=plan.fill,
        quantity_mode=plan.quantity_mode,
        costs=plan.costs,
        slippage=plan.slippage[0],
        capital=plan.capital,
    )
    return block


def _clock(ts_ns: int) -> str:
    moment = dt.datetime.fromtimestamp(ts_ns / NANOS_PER_SECOND, tz=IST)
    return moment.strftime("%H:%M")


__all__ = ["CLASS_OVERRIDES", "RunPlan", "assumptions", "resolve"]
