"""The checks of `record`'s arguments (private to `recorder`): the instruments, the
client's feed calls, the queue size, and the instant the recording stops. Each raises a
ConfigError (or CalendarError) whose message says what to pass instead.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .._calendar_convert import format_ist
from ..calendar import Calendar, default_calendar
from ..errors import CalendarError, ConfigError
from ._tick_files import ist_day

#: The feed modes a recording can subscribe in: LTP, quote (the default) and depth. The
#: SDK's depth data carries no volume and no last trade time; quote data no bid or ask.
RECORD_MODES = ("ltp", "quote", "depth")
# The client calls a recording makes besides the mode's subscribe and unsubscribe.
_FEED_CALLS = ("connect", "disconnect", "on_disconnect", "on_reconnect", "remove_hook")


def instruments_of(instruments: Any) -> tuple[tuple[str, str], ...]:
    """(exchange, symbol) of each instrument, in order, without repeats. Raises
    ConfigError unless `instruments` is a non-empty list of {"exchange", "symbol"}."""
    if isinstance(instruments, str | Mapping) or not isinstance(instruments, Iterable):
        raise ConfigError(
            "instruments is a list of {'exchange': ..., 'symbol': ...}, got "
            f"{type(instruments).__name__}"
        )
    pairs = []
    for item in instruments:
        exchange = item.get("exchange") if isinstance(item, Mapping) else None
        symbol = item.get("symbol") if isinstance(item, Mapping) else None
        if not (isinstance(exchange, str) and exchange and isinstance(symbol, str) and symbol):
            raise ConfigError(
                f"each instrument is {{'exchange': ..., 'symbol': ...}} with both named, as "
                f"OpenAlgo writes them; got {item!r}"
            )
        pairs.append((exchange, symbol))
    if not pairs:
        raise ConfigError("instruments is empty: name at least one {'exchange', 'symbol'}")
    return tuple(dict.fromkeys(pairs))


def feed_calls(client: Any, mode: str) -> tuple[Callable, Callable]:
    """The client's subscribe and unsubscribe calls of `mode`, once every feed call a
    recording makes is there. Raises ConfigError for an unknown mode or missing calls."""
    if mode not in RECORD_MODES:
        raise ConfigError(f"mode must be one of {', '.join(RECORD_MODES)}, got {mode!r}")
    names = (f"subscribe_{mode}", f"unsubscribe_{mode}", *_FEED_CALLS)
    missing = [name for name in names if not callable(getattr(client, name, None))]
    if missing:
        raise ConfigError(
            f"the client has no {', '.join(missing)}: pass an openalgo.api client, whose "
            "WebSocket feed the recording uses"
        )
    return getattr(client, names[0]), getattr(client, names[1])


def queue_size_of(value: Any) -> int:
    """`value` as a number of messages. Raises ConfigError unless it is an int of 1 or
    more."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ConfigError(f"queue_size is a number of messages, 1 or more; got {value!r}")
    return int(value)


def stop_instant(
    stop_at_ns: Any,
    wanted: tuple[tuple[str, str], ...],
    started_ns: int,
    calendar: Calendar | None,
) -> int:
    """The instant the recording stops, UTC nanoseconds: `stop_at_ns`, or when None the
    latest close of the instruments' exchanges on the IST day of the start. Raises
    ConfigError for a stop that is not an int or not after the start, CalendarError
    when no exchange of the instruments has a session that day."""
    if stop_at_ns is None:
        stop_ns = _close_of_day(wanted, started_ns, calendar)
        if stop_ns <= started_ns:
            raise ConfigError(
                f"the sessions of {ist_day(started_ns)} closed at {format_ist(stop_ns)}: "
                "pass stop_at_ns (UTC nanoseconds) to record after the close"
            )
        return stop_ns
    if isinstance(stop_at_ns, bool) or not isinstance(stop_at_ns, numbers.Integral):
        raise ConfigError(
            f"stop_at_ns is an instant in UTC nanoseconds (an int), got {stop_at_ns!r}"
        )
    if stop_at_ns <= started_ns:
        raise ConfigError(
            f"stop_at_ns ({format_ist(int(stop_at_ns))}) is not after the start "
            f"({format_ist(started_ns)}): pass a later instant"
        )
    return int(stop_at_ns)


def _close_of_day(
    wanted: tuple[tuple[str, str], ...], started_ns: int, calendar: Calendar | None
) -> int:
    """The latest close of the instruments' exchanges on the IST day of the start."""
    calendar = default_calendar() if calendar is None else calendar
    day = ist_day(started_ns)
    exchanges = list(dict.fromkeys(exchange for exchange, _ in wanted))
    sessions = [calendar.session(exchange, day) for exchange in exchanges]
    closes = [session[1] for session in sessions if session is not None]
    if not closes:
        raise CalendarError(
            f"{', '.join(exchanges)} has no session on {day}: pass stop_at_ns (UTC "
            "nanoseconds) to record on a day the calendar holds no session"
        )
    return max(closes)


__all__ = ["RECORD_MODES", "feed_calls", "instruments_of", "queue_size_of", "stop_instant"]
