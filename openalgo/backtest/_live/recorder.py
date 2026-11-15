"""The tick recorder behind `bt.record` (08-live-trading section 2a, FR-LIVE-19): the
day's ticks from the OpenAlgo WebSocket, written to the package's local cache so that a
backtest can replay them later in the order a session received them.

- `record` connects the client's feed, subscribes, and writes every tick until the stop
  (the day's close by default, or Ctrl+C), then finalizes the day.
- `finalize` turns the raw files a stopped or crashed recording left into day files.
- `read_ticks` and `read_day` read a finalized day back, verified.

A recorded session is the day's ticks plus the journal: order updates are not recorded
here, because the live session's journal (`journal`) already holds every one of them.

How the files are laid out, and what a crash can cost, is in `_tick_files` and
`_tick_days`; the queue and the writer loop are in `_recorder_loop`; how a feed message
becomes a tick is in `ticks`. One recording or finalize writes a cache at a time (the
recorder's lock, `_recorder_lock`). Times are UTC nanoseconds, prices rupees, volume
units. Nothing here prints: the recorder logs to the `openalgo.backtest.live` logger.
"""

from __future__ import annotations

import enum
import logging
import numbers
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .._calendar_convert import DateLike, format_ist, from_day, to_day
from ..calendar import Calendar, default_calendar
from ..errors import CalendarError, ConfigError, LiveError
from ._recorder_lock import RecorderLock
from ._recorder_loop import DEFAULT_QUEUE_SIZE, FLUSH_INTERVAL_NS, Recording
from ._tick_days import DayWritten, Outage, RecordedDay, finalize_day
from ._tick_days import read_day as _read_day
from ._tick_files import TICK_DTYPE, ist_day, ticks_of, ticks_root
from ._types import LOGGER_NAME, Now, Sleep, Tick

#: The feed modes a recording can subscribe in: LTP, quote (the default) and depth. The
#: SDK's depth data carries no volume and no last trade time; quote data no bid or ask.
RECORD_MODES = ("ltp", "quote", "depth")
# The client calls a recording makes besides the mode's subscribe and unsubscribe.
_FEED_CALLS = ("connect", "disconnect", "on_disconnect", "on_reconnect", "remove_hook")

logger = logging.getLogger(LOGGER_NAME)


class Stop(enum.Enum):
    """Why a recording stopped."""

    CLOSE = "close"  # the day's close from the calendar, the default stop
    STOP_AT = "stop_at"  # the `stop_at_ns` the call gave
    INTERRUPT = "interrupt"  # Ctrl+C (KeyboardInterrupt)


@dataclass(frozen=True)
class RecordSummary:
    """What a recording did.

    `ticks`: ticks recorded per (exchange, symbol), every instrument asked for included.
    `outages`: the feed's outages, in order. `files`: the day files written, with each
    day's tick count. `dropped`: feed messages that made no tick (`ticks.TickTally`).
    `event_times_rejected`: ticks kept without an implausible last trade time.
    `overflowed`: messages lost because the writer's queue was full. `notes`: files that
    failed verification on the way, and client calls that failed while stopping.
    """

    started_ns: int
    stopped_ns: int
    stopped_by: Stop
    ticks: Mapping[tuple[str, str], int]
    outages: tuple[Outage, ...]
    files: tuple[DayWritten, ...]
    dropped: int
    event_times_rejected: int
    overflowed: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FinalizeReport:
    """What `finalize` did: the day files it wrote and what failed verification."""

    files: tuple[DayWritten, ...]
    notes: tuple[str, ...]


def record(
    client: Any,
    instruments: Sequence[Mapping[str, str]],
    *,
    now: Now,
    mode: str = "quote",
    cache_dir: str | os.PathLike | None = None,
    stop_at_ns: int | None = None,
    wait: Sleep | None = None,
    calendar: Calendar | None = None,
    queue_size: int = DEFAULT_QUEUE_SIZE,
) -> RecordSummary:
    """Record the ticks of `instruments` from the client's WebSocket feed into the cache.

    `client` is an `openalgo.api` client (or anything with its feed calls);
    `instruments` is `[{"exchange": "NSE", "symbol": "SBIN"}, ...]`; `mode` one of
    `RECORD_MODES`. The recording takes the client's data callback of that mode: record
    with a client no live session uses. It connects, subscribes and writes every tick
    until `stop_at_ns` (UTC nanoseconds; by default the latest close of the instruments'
    exchanges on the IST day of the start, from `calendar`, the package's calendar when
    None) or until Ctrl+C, then unsubscribes, disconnects and finalizes the day: each
    instrument gets its day file, one with no ticks too (see `read_ticks`). Feed outages
    (a disconnect until its reconnect) are kept with the day. Raw files an earlier
    recording left for the same instrument and day are finalized first, with their
    outages not known.

    `now` is the clock (UTC nanoseconds); `wait` blocks the writer for at most a number
    of seconds (by default until a tick arrives); `cache_dir` is resolved as the cache
    resolves it. `queue_size` bounds the messages waiting for the writer: the feed never
    waits for it, and a message a full queue cannot take is counted (`overflowed`).

    Returns the summary. Raises ConfigError for an argument that is not usable (the
    message says which), a stop that is not after the start, or a cache that cannot be
    used; CalendarError when the default stop is asked for and no exchange of the
    instruments has a session that day (or the calendar does not know one); LiveError
    when another recording holds the cache or the feed does not connect; OSError when a
    file cannot be written. An error while recording (a full disk, say) stops it: the
    feed is disconnected and the error raised, and the raw files stay for `finalize`.
    """
    wanted = _instruments(instruments)
    subscribe, unsubscribe = _feed_calls(client, mode)
    queue_size = _queue_size(queue_size)
    started_ns = now()
    stop_ns = _stop_instant(stop_at_ns, wanted, started_ns, calendar)
    stopped_by = Stop.CLOSE if stop_at_ns is None else Stop.STOP_AT
    root = ticks_root(cache_dir)
    requests = [{"exchange": exchange, "symbol": symbol} for exchange, symbol in wanted]
    with RecorderLock(root):
        recording = Recording(root, now=now, wait=wait, queue_size=queue_size)
        client.on_disconnect(recording.on_disconnect)
        client.on_reconnect(recording.on_reconnect)
        try:
            if not client.connect():
                raise LiveError(
                    "the feed did not connect: check that OpenAlgo and its WebSocket proxy "
                    "run and the API key is valid; nothing was recorded"
                )
            recording.open(wanted, ist_day(started_ns))
            if not subscribe(requests, recording.on_message):
                logger.warning(
                    "the subscription was not sent now: the feed sends it when it reconnects"
                )
            try:
                recording.run_until(stop_ns)
            except KeyboardInterrupt:
                stopped_by = Stop.INTERRUPT
        finally:
            _hang_up(client, unsubscribe, requests, recording)
            stopped_ns = now()
            recording.finish(stopped_ns)
        files = recording.finalize()
    summary = _summary(recording, wanted, started_ns, stopped_ns, stopped_by, files)
    _log_summary(summary)
    return summary


def finalize(day: DateLike, *, cache_dir: str | os.PathLike | None = None) -> FinalizeReport:
    """Turn the raw files a stopped or crashed recording left for an IST day (a
    `datetime.date` or ISO string) into day files. Their outages are not known, so the
    days record none (`RecordedDay.outages` is None). A raw file whose last record is
    torn loses that record (`DayWritten.torn_bytes`).

    Raises ConfigError for a day that is not a date or a cache that cannot be used,
    LiveError while a recording holds the cache, OSError when a file cannot be written.
    """
    date = from_day(to_day(day))
    root = ticks_root(cache_dir)
    notes: list[str] = []
    with RecorderLock(root):
        files = finalize_day(root, date, None, notes)
    for note in notes:
        logger.warning(note)
    return FinalizeReport(tuple(files), tuple(notes))


def read_day(
    exchange: str, symbol: str, day: DateLike, *, cache_dir: str | os.PathLike | None = None
) -> RecordedDay:
    """One instrument's recorded ticks of an IST day: the records (`TICK_DTYPE`, in
    receive order) and the day's feed outages, verified against the index's hash.

    Raises ConfigError for a day that is not a date; DataError when the day was not
    recorded or not finalized yet, and when its file fails verification (the file is
    then moved aside, never deleted).
    """
    return _read_day(ticks_root(cache_dir), exchange, symbol, from_day(to_day(day)))


def read_ticks(
    exchange: str, symbol: str, day: DateLike, *, cache_dir: str | os.PathLike | None = None
) -> list[Tick]:
    """One instrument's recorded ticks of an IST day as `Tick`s, in receive order; an
    empty list for a day recorded without a tick. Raises as `read_day` does."""
    return ticks_of(read_day(exchange, symbol, day, cache_dir=cache_dir).records, exchange, symbol)


def _instruments(instruments: Any) -> tuple[tuple[str, str], ...]:
    """(exchange, symbol) of each instrument, in order, without repeats."""
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


def _feed_calls(client: Any, mode: str) -> tuple[Callable, Callable]:
    """The client's subscribe and unsubscribe calls of `mode`, once every feed call a
    recording makes is there."""
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


def _queue_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ConfigError(f"queue_size is a number of messages, 1 or more; got {value!r}")
    return int(value)


def _stop_instant(
    stop_at_ns: Any,
    wanted: tuple[tuple[str, str], ...],
    started_ns: int,
    calendar: Calendar | None,
) -> int:
    """The instant the recording stops: `stop_at_ns`, or the day's latest close."""
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


def _hang_up(
    client: Any, unsubscribe: Callable, requests: list[dict[str, str]], recording: Recording
) -> None:
    """Unsubscribe, disconnect and remove the hooks. A call that fails is logged and
    noted, and the others are still made: the recording must still be finalized."""
    calls = (
        ("unsubscribe", lambda: unsubscribe(requests)),
        ("disconnect", client.disconnect),
        ("remove the disconnect hook", lambda: client.remove_hook(recording.on_disconnect)),
        ("remove the reconnect hook", lambda: client.remove_hook(recording.on_reconnect)),
    )
    for what, call in calls:
        try:
            call()
        except Exception as exc:
            logger.error("the recording could not %s: %s", what, exc, exc_info=True)
            recording.notes.append(f"the recording could not {what} ({exc})")


def _summary(
    recording: Recording,
    wanted: tuple[tuple[str, str], ...],
    started_ns: int,
    stopped_ns: int,
    stopped_by: Stop,
    files: list[DayWritten],
) -> RecordSummary:
    counted = recording.ticks
    ticks = {instrument: counted.pop(instrument, 0) for instrument in wanted} | counted
    return RecordSummary(
        started_ns=started_ns,
        stopped_ns=stopped_ns,
        stopped_by=stopped_by,
        ticks=MappingProxyType(ticks),
        outages=recording.outages,
        files=tuple(files),
        dropped=recording.tally.dropped,
        event_times_rejected=recording.tally.event_times_rejected,
        overflowed=recording.overflowed,
        notes=tuple(recording.notes),
    )


def _log_summary(summary: RecordSummary) -> None:
    logger.info(
        "recorded %d ticks of %d instruments from %s to %s (%s), %d feed outages",
        sum(summary.ticks.values()),
        len(summary.ticks),
        format_ist(summary.started_ns),
        format_ist(summary.stopped_ns),
        summary.stopped_by.value,
        len(summary.outages),
    )
    for (exchange, symbol), count in summary.ticks.items():
        if count == 0:
            logger.warning("no tick of %s:%s was recorded: check the symbol", exchange, symbol)
    if summary.overflowed:
        logger.warning(
            "%d feed messages were lost: the writer fell behind the feed (a slow disk?)",
            summary.overflowed,
        )
    if summary.event_times_rejected:
        logger.warning(
            "%d ticks carried a last trade time that is no plausible instant; they were kept "
            "without it",
            summary.event_times_rejected,
        )
    for note in summary.notes:
        logger.warning(note)


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "FLUSH_INTERVAL_NS",
    "RECORD_MODES",
    "TICK_DTYPE",
    "DayWritten",
    "FinalizeReport",
    "Outage",
    "RecordSummary",
    "RecordedDay",
    "Stop",
    "finalize",
    "read_day",
    "read_ticks",
    "record",
]
