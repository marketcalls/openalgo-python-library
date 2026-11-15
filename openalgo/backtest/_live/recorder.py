"""The tick recorder behind `bt.record` (08-live-trading section 2a, FR-LIVE-19): the
day's ticks from the OpenAlgo WebSocket, written to the package's local cache so that a
backtest can replay them later in the exact order a session received them.

- `record` connects the client's feed, subscribes, and writes every tick until the stop
  (the day's close by default, or Ctrl+C), then finalizes the day.
- `finalize` turns the raw files a crashed recording left into day files, and moves
  aside a day file found corrupt.
- `read_ticks`, `read_day` and `read_merged_ticks` read a finalized day back, verified;
  they never write.

A recorded session is the day's ticks plus the journal: order updates are not recorded
here, because the live session's journal (`journal`) already holds every one of them.

Recorded ticks cannot be fetched again: the cache must be an absolute folder, nothing
damaged is ever deleted (it is moved aside), and a copy of the cache is the only
backup. How the files are laid out and what a crash can cost is in `_tick_files`,
`_tick_index` and `_tick_days`; what a day says about its coverage in `_tick_spans`; the
queue and the writer loop in `_recorder_loop`; how a feed message becomes a tick in
`ticks`. One recording or finalize writes a cache at a time (`_recorder_lock`). Times
are UTC nanoseconds, prices rupees, volume units. Nothing here prints: the recorder logs
to the `openalgo.backtest.live` logger.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .._calendar_convert import DateLike, from_day, to_day
from ..calendar import Calendar
from ..errors import LiveError
from ._recorder_args import RECORD_MODES, feed_calls, instruments_of, queue_size_of, stop_instant
from ._recorder_lock import RecorderLock
from ._recorder_loop import DEFAULT_QUEUE_SIZE, FLUSH_INTERVAL_NS, Monotonic, Recording
from ._recorder_summary import FinalizeReport, RecordSummary, Stop, log_summary
from ._tick_days import DayWritten, finalize_day
from ._tick_days import read_day as _read_day
from ._tick_days import read_merged as _read_merged
from ._tick_files import TICK_DTYPE, ticks_of, ticks_root
from ._tick_spans import Coverage, Ending, Outage, RecordedDay, Span
from ._types import LOGGER_NAME, Now, Sleep, Tick

logger = logging.getLogger(LOGGER_NAME)


def record(
    client: Any,
    instruments: Sequence[Mapping[str, str]],
    *,
    now: Now,
    monotonic: Monotonic,
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
    None), Ctrl+C or a write that fails. Then it writes what it holds, closes its files,
    unsubscribes and disconnects (a second Ctrl+C or an error there is noted in the
    summary), removes its hooks and finalizes: each instrument gets its day file, one
    with no ticks too, with the recording's span and feed outages, and any raw file a
    crashed recording left for the same instruments and days is merged in.

    `now` is the wall clock (UTC nanoseconds), which stamps the ticks; `monotonic` a
    clock that never steps back, which schedules the writes; `wait` blocks the writer for
    at most a number of seconds (by default until a tick arrives); `cache_dir` is
    resolved as the cache resolves it and must be an absolute folder (a leading `~` is
    expanded): recorded ticks cannot be fetched again. `queue_size` bounds the messages
    waiting for the writer: the feed never waits for it, and a message a full queue
    cannot take is counted (`overflowed`).

    Returns the summary, also when a write failed (`stopped_by` is then WRITE_FAILED and
    `error` holds the error; the raw files that could not be finalized stay for
    `finalize`). Raises ConfigError for an argument that is not usable (the message says
    which), a stop that is not after the start, or a cache folder that is relative or
    cannot be used; CalendarError when the default stop is asked for and no exchange of
    the instruments has a session that day (or the calendar does not know one);
    LiveError when another recording holds the cache or the feed does not connect;
    OSError when the raw files cannot be made at the start.
    """
    wanted = instruments_of(instruments)
    subscribe, unsubscribe = feed_calls(client, mode)
    queue_size = queue_size_of(queue_size)
    root = ticks_root(cache_dir)
    started_ns = now()
    stop_ns = stop_instant(stop_at_ns, wanted, started_ns, calendar)
    requests = [{"exchange": exchange, "symbol": symbol} for exchange, symbol in wanted]
    stopped_by = Stop.CLOSE if stop_at_ns is None else Stop.STOP_AT
    with RecorderLock(root):
        recording = Recording(
            root, instruments=wanted, now=now, monotonic=monotonic, wait=wait, queue_size=queue_size
        )
        try:
            stopped_by = _run(client, subscribe, requests, recording, stop_ns, stopped_by)
        finally:
            stopped_ns = now()
            _stop(client, unsubscribe, requests, recording, stopped_ns)
        files = recording.finalize()
    summary = RecordSummary(
        started_ns=started_ns,
        stopped_ns=stopped_ns,
        stopped_by=stopped_by,
        recording=recording.recording,
        ticks=MappingProxyType(_ticks_of(recording, wanted)),
        outages=recording.outages,
        files=tuple(files),
        dropped=recording.tally.dropped,
        event_times_rejected=recording.tally.event_times_rejected,
        overflowed=recording.overflowed,
        lost=recording.lost,
        error=recording.write_error,
        notes=tuple(recording.notes),
    )
    log_summary(summary)
    return summary


def finalize(day: DateLike, *, cache_dir: str | os.PathLike | None = None) -> FinalizeReport:
    """Settle an IST day (a `datetime.date` or ISO string) of the cache: turn the raw
    files a crashed recording left into day files (its span ends at its last tick, by a
    crash, its outages unknown), check every instrument's day file of that day, move a
    corrupt one aside (never deleting it: recorded ticks cannot be fetched again) and
    remove stale ones, and rebuild an index that cannot be parsed from the day files.

    Raises ConfigError for a day that is not a date or a cache folder that is relative
    or cannot be used, LiveError while a recording holds the cache, OSError naming the
    path when a file cannot be read or written (nothing is moved for it).
    """
    date = from_day(to_day(day))
    root = ticks_root(cache_dir)
    notes: list[str] = []
    with RecorderLock(root):
        files = finalize_day(root, date, notes)
    for note in notes:
        logger.warning(note)
    return FinalizeReport(tuple(files), tuple(notes))


def read_day(
    exchange: str, symbol: str, day: DateLike, *, cache_dir: str | os.PathLike | None = None
) -> RecordedDay:
    """One instrument's recorded ticks of an IST day: the records (`TICK_DTYPE`, in
    receive order), the recordings' spans with their feed outages, and whether a raw
    file of the day is still pending. Verified against the index's hash; never writes.

    Raises ConfigError for a day that is not a date or a relative cache folder;
    DataError when the day was not recorded or not finalized yet, and when its file or
    index is corrupt (left as it is: `finalize` moves it aside); OSError naming the path
    when a file cannot be read.
    """
    return _read_day(ticks_root(cache_dir), exchange, symbol, from_day(to_day(day)))


def read_ticks(
    exchange: str, symbol: str, day: DateLike, *, cache_dir: str | os.PathLike | None = None
) -> list[Tick]:
    """One instrument's recorded ticks of an IST day as `Tick`s, in receive order; an
    empty list for a day recorded without a tick. Raises as `read_day` does."""
    return ticks_of(read_day(exchange, symbol, day, cache_dir=cache_dir).records, exchange, symbol)


def read_merged_ticks(
    instruments: Sequence[Mapping[str, str]],
    day: DateLike,
    *,
    cache_dir: str | os.PathLike | None = None,
) -> list[Tick]:
    """The recorded ticks of several instruments (`[{"exchange", "symbol"}, ...]`) of an
    IST day, merged in the exact order they were received: the recordings in the order
    of their start, each by its sequence numbers, also where receive times tie or the
    wall clock stepped back. Raises as `read_day` does, and ConfigError for instruments
    that are not usable."""
    root = ticks_root(cache_dir)
    return _read_merged(root, instruments_of(instruments), from_day(to_day(day)))


def _run(
    client: Any,
    subscribe: Callable,
    requests: list[dict[str, str]],
    recording: Recording,
    stop_ns: int,
    stopped_by: Stop,
) -> Stop:
    """Connect, subscribe and record until the stop; why it stopped."""
    client.on_disconnect(recording.on_disconnect)
    client.on_reconnect(recording.on_reconnect)
    if not client.connect():
        raise LiveError(
            "the feed did not connect: check that OpenAlgo and its WebSocket proxy run and "
            "the API key is valid; nothing was recorded"
        )
    recording.open()
    if not subscribe(requests, recording.on_message):
        logger.warning("the subscription was not sent now: the feed sends it when it reconnects")
    try:
        written = recording.run_until(stop_ns)
    except KeyboardInterrupt:
        return Stop.INTERRUPT
    return stopped_by if written else Stop.WRITE_FAILED


def _stop(
    client: Any,
    unsubscribe: Callable,
    requests: list[dict[str, str]],
    recording: Recording,
    end_ns: int,
) -> None:
    """Stop a recording so that a second Ctrl+C loses nothing: write what it holds and
    close its files first (once more if that was interrupted), then unsubscribe and
    disconnect, each failure noted; the hooks are removed whatever happens."""
    try:
        if not _attempt(recording, "write the last ticks", lambda: recording.finish(end_ns)):
            _attempt(recording, "write the last ticks again", lambda: recording.finish(end_ns))
        _attempt(recording, "unsubscribe", lambda: unsubscribe(requests))
        _attempt(recording, "disconnect", client.disconnect)
    finally:
        try:
            recording.close()
        finally:
            for name, hook in (
                ("disconnect", recording.on_disconnect),
                ("reconnect", recording.on_reconnect),
            ):
                _attempt(recording, f"remove its {name} hook", lambda h=hook: client.remove_hook(h))


def _attempt(recording: Recording, what: str, call: Callable[[], object]) -> bool:
    """Make one call of the stop; a Ctrl+C or an error in it is logged and noted, and
    the stop goes on. Returns whether it succeeded."""
    try:
        call()
    except (KeyboardInterrupt, Exception) as exc:
        detail = f": {exc}" if str(exc) else ""
        note = f"the recording could not {what} ({type(exc).__name__}{detail})"
        logger.error(note, exc_info=not isinstance(exc, KeyboardInterrupt))
        recording.notes.append(note)
        return False
    return True


def _ticks_of(recording: Recording, wanted: tuple[tuple[str, str], ...]) -> dict:
    """Ticks written per instrument: every instrument asked for, then any other."""
    counted = recording.ticks
    return {instrument: counted.pop(instrument, 0) for instrument in wanted} | counted


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "FLUSH_INTERVAL_NS",
    "RECORD_MODES",
    "TICK_DTYPE",
    "Coverage",
    "DayWritten",
    "Ending",
    "FinalizeReport",
    "Monotonic",
    "Outage",
    "RecordSummary",
    "RecordedDay",
    "Span",
    "Stop",
    "finalize",
    "read_day",
    "read_merged_ticks",
    "read_ticks",
    "record",
]
