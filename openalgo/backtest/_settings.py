"""Resolving the settings of a run: the documented defaults, the strategy's class
attributes and the run arguments, into one `RunPlan` the kernel receives explicitly.

Precedence (lead ruling): a strategy class attribute wins over the run argument of the
same name, which wins over the documented default. The documented defaults are those of
03-architecture 4.1 and 04-execution 2: product MIS for intraday intervals, CNC for daily
equity and NRML for daily derivatives; entry window and square-off from the calendar's
dated table (MIS only, SPEC-32); fill at the next open; whole shares and lots (the venue
rule, decision D26); Zerodha charges; the per-segment slippage. Every value used is
listed in the Assumptions block.

Risk settings take the units of 14-strategy-api: `max_daily_loss` in rupees (a number or
`bt.rupees`), `drawdown_halt` in percent (`bt.pct(x)` or a number of percent),
`cooldown`, `max_hold` and `max_positions` as whole numbers (base bars, positions), the
scale-in limits as `_scale_in` reads them, and `reinvest` and `size_caps` as
`_size_controls` reads them. The combined rules of a run are refused
until Phase 3 builds them; no other setting is accepted and then ignored
(`_setting_checks`). The futures margin, `margin=`, is read by `_futures_margin`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import slippage as slippage_module
from ._calendar_convert import interval_nanos, to_day, to_seconds
from ._futures import check_roll
from ._futures_margin import MarginSetting, parse_margin
from ._intraday_assumptions import TIMES as INTRADAY_TIMES
from ._intraday_assumptions import intraday_times
from ._layout import parse_interval
from ._messages import not_supported
from ._scale_in import describe as describe_scale_limits
from ._scale_in import scale_limits
from ._series import IST
from ._setting_checks import class_leverage, refuse_unusable, unused_settings
from ._size_controls import describe as describe_size_controls
from ._size_controls import size_controls
from .data._sessions import resolve_now
from .data.loader import own_reader
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
# Combined rules of a run, class attributes for strategies that open single positions
# (14-strategy-api 2.1); refused until Phase 3 builds them.
COMBINED_RULES = ("combined_stop", "combined_target", "lock_profit", "trail_sl_to_entry")
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
    # The present, UTC nanoseconds (`now=`, the clock by default): the run and the data
    # load stop at it (`rust/oa_bt/src/engine/run_end.rs`).
    present: int
    capital: float
    fill: str
    quantity_mode: str
    costs: str | None
    slippage: tuple[str, float, int]
    risk: dict[str, Any]
    entry_from: str | None
    entry_until: str | None
    square_off: str | None
    leverage: float | None
    warnings: tuple[str, ...]
    # Sessions before expiry at which a future rolls, or None to settle at expiry.
    roll: int | None = 1
    # The run's own futures margin (`margin=`, `_futures_margin`), None for the default.
    margin: MarginSetting = None

    def native_config(self, deliver_events: bool) -> dict[str, Any]:
        """The config dict of the native engine."""
        # Nanoseconds for an intraday interval, else its code ("D", or "W", "M", "Q" and
        # "Y", which the kernel refuses saying what to do instead).
        intraday = parse_interval(self.interval)[0] == "intraday"
        strategy = interval_nanos(self.interval) if intraday else self.interval
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
            "present": self.present,
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
    """The run-wide slippage form; a table (the default) is resolved per instrument
    by `_market`. Raises ConfigError for an unknown form."""
    resolved = slippage_module.resolve(value)
    if isinstance(resolved, slippage_module.SlippageTable):
        return ("segment_default", 0.0, 0)
    return resolved.to_native()


def _instant(value: Any) -> int | None:
    """UTC nanoseconds of a datetime bound (naive is IST), or None for a plain date."""
    if isinstance(value, dt.datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=IST)
        return int(pd.Timestamp(moment).value)
    return None


def _frame_range(strategy: type, args: dict[str, Any]) -> tuple[dt.date, dt.date]:
    """The IST dates the own data (`data=`) of the strategy's symbols covers, its
    timestamps read as the data layer reads them (naive ones in `tz=`). Raises the data
    layer's errors for naive timestamps without `tz=` or a file it cannot read, and
    ConfigError for data given by a provider that is not own data."""
    data = args["data"]
    symbols = strategy.traded_symbols()
    frames = data if isinstance(data, dict) else dict.fromkeys(symbols, data)
    stamps: list[int] = []
    for symbol in symbols:
        if symbol not in frames:
            raise ConfigError(f"data= has no bars for {symbol}")
        reader = own_reader(frames[symbol], args["tz"])
        if reader is None:
            raise ConfigError(
                "pass start= and end= (IST dates): the run reads its dates only from own "
                "data (a DataFrame, a file path or bt.data.Frame or Files), or a dict of "
                "them by symbol"
            )
        stamps += reader.stamp_range(symbol, strategy.exchange, strategy.interval)
    days = [pd.Timestamp(stamp, tz="UTC").tz_convert(IST).date() for stamp in stamps]
    return min(days), max(days)


def _dates(strategy: type, args: dict[str, Any]) -> tuple[dt.date, dt.date]:
    start, end = args["start"], args["end"]
    if start is None or end is None:
        if args["data"] is None:
            raise ConfigError("pass start= and end= (IST dates)")
        first, last = _frame_range(strategy, args)
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


def _base(base: str | None, interval: str) -> str:
    """The base interval before data is seen: `base=`, daily for daily or longer
    strategies, else the strategy interval, which `_base_interval.choose_base` replaces
    with 1 minute when 1-minute data covers the run (03-architecture 4.1)."""
    if base is not None:
        parse_interval(base)
        return base
    if parse_interval(interval)[0] != "intraday":
        return "D"
    return interval


def _rupees(value: Any, name: str) -> float | None:
    """None, or a positive amount of rupees (a number or bt.rupees(x))."""
    if value is None:
        return None
    if isinstance(value, Distance) and value.unit == "rupees":
        return value.value
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{name} takes a positive amount of rupees, got {value!r}")
    return float(value)


def _percent(value: Any, name: str) -> float | None:
    """None, or a percent above 0 and below 100 (bt.pct(x) or a bare number of percent)."""
    if value is None:
        return None
    if isinstance(value, Distance) and value.unit == "pct":
        value = value.value
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 100:
        raise ConfigError(f"{name} takes bt.pct(x) with x above 0 and below 100, got {value!r}")
    return float(value)


def _count(value: Any, name: str, unit: str) -> int | None:
    """None, or a positive whole number of `unit`."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} takes a positive whole number of {unit}, got {value!r}")
    return value


def _bars(value: Any, name: str) -> int | None:
    """None, or a positive whole number of base bars. A time ("30m") is not built yet."""
    if isinstance(value, str):
        raise not_supported(
            f"{name} as a time ({value!r})", f"give {name} as a whole number of base bars"
        )
    return _count(value, name, "base bars")


def _refuse_combined_rules(strategy: type) -> None:
    """Combined rules of a run (05-risk-management 4) are Phase 3; never ignore them."""
    for name in COMBINED_RULES:
        if getattr(strategy, name, None) not in (None, False):
            raise not_supported(
                f"{name} (a combined rule over the run's positions)",
                "give each order its own stop=, target= or trail=, and the run max_daily_loss=",
            )


def _risk(strategy: type, args: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_daily_loss": _rupees(overrides["max_daily_loss"], "max_daily_loss"),
        "max_positions": _count(args["max_positions"], "max_positions", "positions"),
        "drawdown_halt": _percent(overrides["drawdown_halt"], "drawdown_halt"),
        "cooldown_bars": _bars(overrides["cooldown"], "cooldown"),
        "max_hold_bars": _bars(getattr(strategy, "max_hold", None), "max_hold"),
        "risk_basis": _choice(args["risk_basis"], RISK_BASES, "risk_basis"),
        "after_risk_exit": _choice(args["after_risk_exit"], AFTER_RISK_EXIT, "after_risk_exit"),
        "round_levels": bool(args["round_levels"]),
        "strict_config": bool(args["strict_config"]),
        **scale_limits(strategy),
        **size_controls(strategy),
    }


def resolve(strategy: type, args: dict[str, Any]) -> RunPlan:
    """The plan of a run of `strategy` with the run arguments `args` (the keyword
    arguments of `bt.backtest`). Raises ConfigError for unusable values and
    NotImplementedError for settings of later work."""
    symbols = strategy.traded_symbols()
    if not strategy.exchange or not strategy.interval:
        raise ConfigError(f"{strategy.__name__}: set exchange and interval on the class")
    parse_interval(strategy.interval)
    capital = args["capital"]
    if isinstance(capital, bool) or not isinstance(capital, int | float) or capital <= 0:
        raise ConfigError("capital must be a positive number of rupees")
    _refuse_combined_rules(strategy)
    refuse_unusable(strategy, args)
    warnings: list[str] = []
    if args["costs"] is None:
        warnings.append("costs=None: no charges are modelled; results overstate P&L")
    overrides = {
        name: getattr(strategy, name) if getattr(strategy, name) is not None else args[name]
        for name in CLASS_OVERRIDES
    }
    start, end = _dates(strategy, args)
    product = _product(strategy, args["product"], strategy.interval, strategy.exchange)
    warnings += unused_settings(strategy, args, product, overrides)
    return RunPlan(
        symbols=symbols,
        exchange=strategy.exchange,
        interval=strategy.interval,
        base=_base(args["base"], strategy.interval),
        product=product,
        start=start,
        end=end,
        first_slice=_instant(args["start"]),
        last_slice=_instant(args["end"]),
        present=resolve_now(args["now"]),
        capital=float(capital),
        fill=_choice(args["fill"], FILL_MODES, "fill"),
        quantity_mode=_choice(args["quantity_mode"], QUANTITY_MODES, "quantity_mode"),
        costs=args["costs"],
        slippage=_slippage(args["slippage"]),
        risk=_risk(strategy, args, overrides),
        entry_from=overrides["entry_from"],
        entry_until=overrides["entry_until"],
        square_off=overrides["square_off"],
        leverage=class_leverage(strategy),
        warnings=tuple(warnings),
        roll=check_roll(args["roll"]),
        margin=parse_margin(args["margin"]),
    )


def assumptions(plan: RunPlan, calendar: Any) -> dict[str, Any]:
    """The Assumptions block: what the run used. Entry window and square-off apply to
    MIS only (SPEC-32); they are stated as they applied over the whole run
    (`_intraday_assumptions`)."""
    block: dict[str, Any] = {"base": plan.base, "product": plan.product}
    times: dict[str, str | None] = dict.fromkeys(INTRADAY_TIMES)
    if plan.product == "MIS":
        given = {name: getattr(plan, name) for name in INTRADAY_TIMES}
        times = intraday_times(calendar, plan.exchange, plan.start, plan.end, given)
    block.update(times)
    block.update(
        fill=plan.fill,
        quantity_mode=plan.quantity_mode,
        costs=plan.costs,
        slippage=plan.slippage[0],
        capital=plan.capital,
    )
    block.update(_risk_assumptions(plan))
    return block


def _risk_assumptions(plan: RunPlan) -> dict[str, Any]:
    """The strategy and session rules of the run, and where their exits fill: at the
    next base-bar open, or on a daily base at the close of the day that breached
    (SPEC-17)."""
    risk = plan.risk
    block = {
        "max_daily_loss": risk["max_daily_loss"],
        "risk_basis": risk["risk_basis"],
        "after_risk_exit": risk["after_risk_exit"],
        "drawdown_halt": risk["drawdown_halt"],
        "cooldown": risk["cooldown_bars"],
        "max_positions": risk["max_positions"],
        "max_hold": risk["max_hold_bars"],
        "leverage": plan.leverage,
        **describe_scale_limits(risk),
        **describe_size_controls(risk),
    }
    if risk["max_daily_loss"] is not None or risk["drawdown_halt"] is not None:
        block["risk_exit_fill"] = (
            "at the close of the day that breached, flagged"
            if plan.base == "D"
            else "at the next base-bar open"
        )
    return block


__all__ = ["CLASS_OVERRIDES", "RunPlan", "assumptions", "resolve"]
