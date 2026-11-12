"""IST timestamps, session grids and strategy-bar boundaries for the trace fixtures.

Every timestamp in a fixture is an ISO 8601 string in IST (offset +05:30) and names the
instant a bar opens. A daily bar carries the session open of its date. A "slice" is one
slot of the base-interval grid, identified by the open of its base bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from openalgo.backtest._layout import parse_interval

IST = timezone(timedelta(hours=5, minutes=30))

Case = dict[str, Any]


@dataclass(frozen=True)
class Session:
    """One trading session of an exchange: its open and close instants in IST."""

    open: datetime
    close: datetime


def parse_ist(time_ist: str) -> datetime:
    """Parse a fixture timestamp. Raises ValueError unless the offset is +05:30."""
    moment = datetime.fromisoformat(time_ist)
    if moment.utcoffset() != timedelta(hours=5, minutes=30):
        raise ValueError(f"{time_ist!r} is not an IST timestamp (offset +05:30)")
    return moment


def format_ist(moment: datetime) -> str:
    """The fixture spelling of an IST instant, to the second."""
    return moment.astimezone(IST).isoformat(timespec="seconds")


def interval_step(interval: str) -> timedelta | None:
    """Length of an intraday interval, or None for the daily interval "D".

    Raises ValueError for weekly and longer intervals, which no trace uses.
    """
    kind, count = parse_interval(interval)
    if kind == "intraday":
        return timedelta(seconds=count)
    if kind == "D":
        return None
    raise ValueError(f"interval {interval!r} is not supported by the trace fixtures")


def exchange_of(case: Case, instrument: str) -> str:
    """Exchange code of an instrument declared by the case. Raises KeyError if unknown."""
    for declared in case["instruments"]:
        if declared["symbol"] == instrument:
            return declared["exchange"]
    raise KeyError(f"{case['id']}: instrument {instrument!r} is not declared")


def sessions(case: Case, exchange: str) -> list[Session]:
    """The sessions the case declares for an exchange, in calendar order."""
    result = []
    for row in case["calendar"][exchange]:
        opens = parse_ist(f"{row['date']}T{row['open_ist']}:00+05:30")
        closes = parse_ist(f"{row['date']}T{row['close_ist']}:00+05:30")
        result.append(Session(opens, closes))
    return result


def session_of(case: Case, exchange: str, moment: datetime) -> Session:
    """The session that contains an instant (open inclusive, close exclusive)."""
    for session in sessions(case, exchange):
        if session.open <= moment < session.close:
            return session
    raise ValueError(f"{case['id']}: {format_ist(moment)} is outside every {exchange} session")


def grid_slices(case: Case, exchange: str) -> list[str]:
    """Every base-interval slot of the run window, as IST strings, in time order."""
    step = interval_step(case["settings"]["base"])
    first = parse_ist(case["settings"]["first_slice_ist"])
    last = parse_ist(case["settings"]["last_slice_ist"])
    slots: list[datetime] = []
    for session in sessions(case, exchange):
        if step is None:
            slots.append(session.open)
            continue
        slot = session.open
        while slot < session.close:
            slots.append(slot)
            slot += step
    return [format_ist(slot) for slot in slots if first <= slot <= last]


def bar_close(case: Case, exchange: str, bar_open_ist: str, interval: str) -> datetime:
    """The instant a bar of the given interval closes (the session close for a daily bar)."""
    opens = parse_ist(bar_open_ist)
    session = session_of(case, exchange, opens)
    step = interval_step(interval)
    if step is None:
        return session.close
    return min(opens + step, session.close)


def bar_open_containing(case: Case, exchange: str, time_ist: str, interval: str) -> str:
    """Open of the session-aligned bar of `interval` that contains an instant."""
    moment = parse_ist(time_ist)
    session = session_of(case, exchange, moment)
    step = interval_step(interval)
    if step is None:
        return format_ist(session.open)
    whole_steps = (moment - session.open) // step
    return format_ist(session.open + whole_steps * step)


def decision_slice(case: Case, exchange: str, bar_open_ist: str) -> str:
    """The base slot whose close is the close of a strategy bar (where on_bar runs)."""
    settings = case["settings"]
    closes = bar_close(case, exchange, bar_open_ist, settings["strategy_interval"])
    step = interval_step(settings["base"])
    if step is None:
        return bar_open_ist
    return format_ist(closes - step)
