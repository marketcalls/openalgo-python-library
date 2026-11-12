"""The base interval of a run and its bars (03-architecture 4.1, decision D18).

Orders and risk rules are evaluated on one base interval per run:

- `base=` given: that interval.
- A daily or longer strategy: daily.
- An intraday strategy whose instruments all have 1-minute data covering the whole run:
  1 minute.
- Any other intraday strategy (short history, or data only at the strategy interval):
  the strategy interval.

Coverage is decided by a probe: each instrument's bars are loaded at 1 minute through
the data layer (the OpenAlgo source, or the user's own data). They cover the run when
the load succeeds, the bars really are 1 minute apart (data at a coarser interval can
pass the calendar's checks), the source returned the whole requested range and no
session of the range is without bars. When every instrument passes, those bars are the
run's bars; otherwise every instrument is loaded at the strategy interval (own 1-minute
data is resampled to it). The Assumptions block states the base, why it was chosen and
what it means for fills.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._layout import IST_OFFSET_NANOS, NANOS_PER_MINUTE, parse_interval
from .data import BarsResult, load_bars
from .errors import ConfigError, DataError

ONE_MINUTE = "1m"
NANOS_PER_DAY = 24 * 60 * NANOS_PER_MINUTE
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
DAILY_MEANING = (
    "stops are tested on the day's range and fill at the stop level, or at the open on a "
    "gap; combined and daily-loss exits fill at the day's close"
)


@dataclass(frozen=True)
class BaseChoice:
    """The chosen base interval, the bars loaded at it by symbol, and why it was chosen."""

    base: str
    bars: dict[str, BarsResult]
    reason: str


def _data_for(data: Any, symbol: str) -> Any:
    """The own data of one symbol: `data` itself, or its entry in a dict by symbol."""
    if not isinstance(data, dict):
        return data
    if symbol not in data:
        raise ConfigError(f"data= has no bars for {symbol}")
    return data[symbol]


def _one_minute_apart(bars: Any) -> bool:
    """True when consecutive bars of a session are a whole number of minutes apart with
    1 minute as the greatest common step: real 1-minute bars."""
    opens = bars["ts_open"].to_numpy()
    days = (opens + IST_OFFSET_NANOS) // NANOS_PER_DAY
    steps = np.diff(opens)[days[1:] == days[:-1]]
    return len(steps) > 0 and int(np.gcd.reduce(steps)) == NANOS_PER_MINUTE


def _gap(symbol: str, loaded: BarsResult) -> str | None:
    """Why the 1-minute bars of a symbol do not cover the run, or None when they do."""
    if not _one_minute_apart(loaded.bars):
        return f"{symbol}: the bars are further apart than 1 minute"
    meta = loaded.meta
    missing = meta.quality.count("missing_sessions") if meta.quality else 0
    if meta.short_history or missing:
        return (
            f"{symbol}: 1-minute bars run from {meta.obtained_start} to {meta.obtained_end} "
            f"with {missing} sessions missing, for {meta.requested_start} to "
            f"{meta.requested_end}"
        )
    return None


def _first_line(error: Exception) -> str:
    return str(error).split(". ")[0].rstrip(".")


def choose_base(
    plan: Any, args: dict[str, Any], client: Any, calendar: Any, windows: dict | None = None
) -> BaseChoice:
    """The base interval of `plan` (resolved without data) and its bars, by the rule of
    the module docstring. `windows` gives some symbols their own (first, last) dates (a
    futures contract is loaded for the dates it is traded). Raises the data layer's
    errors when the bars of the chosen base cannot be loaded."""

    def load(symbol: str, interval: str, base: str | None = None) -> BarsResult:
        start, end = (windows or {}).get(symbol, (plan.start, plan.end))
        return load_bars(
            symbol,
            plan.exchange,
            interval,
            start,
            end,
            source=args["source"],
            client=client,
            data=_data_for(args["data"], symbol),
            calendar=calendar,
            tz=args["tz"],
            base=base,
        )

    if args["base"] is not None:
        return _load_all(plan, plan.base, load, "set by base=")
    if parse_interval(plan.interval)[0] != "intraday":
        return _load_all(plan, "D", load, "a daily or longer strategy")
    if plan.interval == ONE_MINUTE:
        return _load_all(plan, ONE_MINUTE, load, "the strategy interval is 1 minute")
    return _probe(plan, args, load)


def _load_all(plan: Any, base: str, load: Callable[..., BarsResult], reason: str) -> BaseChoice:
    return BaseChoice(base, {symbol: load(symbol, base) for symbol in plan.symbols}, reason)


def _probe(plan: Any, args: dict[str, Any], load: Callable[..., BarsResult]) -> BaseChoice:
    loaded: dict[str, BarsResult] = {}
    gaps: list[str] = []
    for symbol in plan.symbols:
        try:
            loaded[symbol] = load(symbol, ONE_MINUTE)
        except DataError as exc:
            gaps.append(f"{symbol}: no usable 1-minute data ({_first_line(exc)})")
            continue
        gap = _gap(symbol, loaded[symbol])
        if gap is not None:
            gaps.append(gap)
    if not gaps:
        reason = "1-minute data covers the run for every instrument"
        return BaseChoice(ONE_MINUTE, loaded, reason)
    own_minutes = {
        symbol
        for symbol, result in loaded.items()
        if args["data"] is not None and _one_minute_apart(result.bars)
    }
    bars = {
        symbol: load(symbol, plan.interval, ONE_MINUTE if symbol in own_minutes else None)
        for symbol in plan.symbols
    }
    reason = f"1-minute data does not cover the run ({'; '.join(gaps)})"
    return BaseChoice(plan.interval, bars, reason)


def _span(interval: str) -> str:
    seconds = parse_interval(interval)[1]
    minutes = seconds // SECONDS_PER_MINUTE
    if minutes % MINUTES_PER_HOUR == 0:
        return f"{minutes // MINUTES_PER_HOUR}-hour"
    return f"{minutes}-minute"


def _clock(minute: int) -> str:
    return f"{minute // MINUTES_PER_HOUR:02d}:{minute % MINUTES_PER_HOUR:02d}"


def meaning(base: str, square_off: str | None, session_open: str | None) -> str:
    """What the base means for fills and stops, as the Assumptions block says it: for
    example "stops are tested on 15-minute ranges and a 15:10 square-off fills at the
    15:15 open". `square_off` and `session_open` are IST "HH:MM" (None without MIS)."""
    if parse_interval(base)[0] != "intraday":
        return DAILY_MEANING
    span = _span(base)
    text = f"stops are tested on {span} ranges and orders fill at {span} bar opens"
    if square_off is None or session_open is None:
        return text
    step = parse_interval(base)[1] // SECONDS_PER_MINUTE
    opens = int(session_open[:2]) * MINUTES_PER_HOUR + int(session_open[3:])
    target = int(square_off[:2]) * MINUTES_PER_HOUR + int(square_off[3:])
    fill = opens + -(-(target - opens) // step) * step
    return f"{text}; a {square_off} square-off fills at the {_clock(fill)} open"


__all__ = ["BaseChoice", "choose_base", "meaning"]
