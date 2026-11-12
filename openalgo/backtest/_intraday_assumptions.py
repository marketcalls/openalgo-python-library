"""The intraday times of a run as the Assumptions block states them: `entry_from`,
`entry_until` and `square_off`, for MIS only (SPEC-32).

The kernel resolves them per session (`rust/oa_bt/src/engine/session.rs`): a time given
to the run is that clock time in every session; otherwise `entry_from` is the session's
open, and `entry_until` and `square_off` are the calendar's dated defaults, which for MCX
and special sessions sit a fixed time before that session's close. So a time can change
during a run, on a dated change of the defaults (NSE and BSE cash from 2026-08-03,
decision TIMES-1) or with the session hours (MCX when United States daylight saving
starts or ends). Each time is stated as it applied over the whole run:

* one clock time, "HH:MM", when it was the same in every session;
* the rule, "each session's open" or "N minutes before each session's close", when it
  followed the session and its clock time changed;
* otherwise its periods in date order, "HH:MM from DATE to DATE; ...".
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from ._layout import NANOS_PER_SECOND
from ._messages import counted
from ._series import IST

TIMES = ("entry_from", "entry_until", "square_off")
NANOS_PER_MINUTE = 60 * NANOS_PER_SECOND
OPEN_RULE = "each session's open"

# One session's value of a time: its date, the instant (UTC nanoseconds), the minutes it
# sits before the session's close (None when it is not set from the close, 0 for the
# open), and whether it follows the session rather than a clock time.
SessionTime = tuple[dt.date, int, int | None, bool]


def _clock(ts_ns: int) -> str:
    """The IST clock time "HH:MM" of UTC nanoseconds."""
    return dt.datetime.fromtimestamp(ts_ns / NANOS_PER_SECOND, tz=IST).strftime("%H:%M")


def one_clock_time(stated: str | None) -> str | None:
    """`stated` when it is one clock time "HH:MM" for the whole run; None when it is a
    rule, periods, or not set."""
    if stated is None or len(stated) != len("HH:MM") or stated[2] != ":":
        return None
    return stated


def _session_times(calendar: Any, exchange: str, day: dt.date) -> dict[str, SessionTime]:
    """The default times of one session."""
    opens, closes = calendar.session(exchange, day)
    defaults = calendar.intraday_times(exchange, day)
    found = {"entry_from": (day, opens, None, True)}
    if defaults is not None:
        relative = defaults.relative_to_close
        for name in ("entry_until", "square_off"):
            instant = getattr(defaults, name)
            found[name] = (day, instant, (closes - instant) // NANOS_PER_MINUTE, relative)
    return found


def _periods(values: list[SessionTime]) -> str:
    groups: list[list[Any]] = []
    for day, instant, _, _ in values:
        text = _clock(instant)
        if groups and groups[-1][0] == text:
            groups[-1][2] = day
        else:
            groups.append([text, day, day])
    return "; ".join(f"{text} from {first} to {last}" for text, first, last in groups)


def _statement(name: str, values: list[SessionTime]) -> str | None:
    if not values:
        return None
    clocks = {_clock(instant) for _, instant, _, _ in values}
    if len(clocks) == 1:
        return clocks.pop()
    if name == "entry_from":
        return OPEN_RULE
    offsets = {offset for _, _, offset, _ in values}
    if all(follows for *_, follows in values) and len(offsets) == 1:
        return f"{counted(offsets.pop(), 'minute')} before each session's close"
    return _periods(values)


def intraday_times(
    calendar: Any, exchange: str, start: dt.date, end: dt.date, given: dict[str, str | None]
) -> dict[str, str | None]:
    """`entry_from`, `entry_until` and `square_off` of an MIS run from `start` to `end`
    (IST dates) as the module docstring states them; `given` holds the times given to
    the run ("HH:MM" or None). A time with no session in the range is None."""
    stated = {name: given[name] for name in TIMES if given[name] is not None}
    wanted = [name for name in TIMES if name not in stated]
    if not wanted:
        return stated
    collected: dict[str, list[SessionTime]] = {name: [] for name in wanted}
    for day in calendar.sessions(exchange, start, end)["date"]:
        found = _session_times(calendar, exchange, day)
        for name in wanted:
            if name in found:
                collected[name].append(found[name])
    stated.update({name: _statement(name, collected[name]) for name in wanted})
    return {name: stated[name] for name in TIMES}


__all__ = ["TIMES", "intraday_times", "one_clock_time"]
