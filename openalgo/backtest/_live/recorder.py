"""The tick recorder behind `bt.record` (08-live-trading section 2a, FR-LIVE-19): the
day's ticks from the OpenAlgo WebSocket, written to the package's local cache so that a
backtest can replay them later in the exact order a session received them.

- `record` connects the client's feed, subscribes, writes every tick to the
  recording's raw file until the stop (the day's close by default, or Ctrl+C), writes
  how it stopped, and finalizes.
- `finalize` finalizes the raw files a crashed recording left; `repair` moves a damaged
  finalized file aside.
- `read_ticks`, `read_day` and `read_merged_ticks` read a day back, verified; they never
  write.

Every recording has a number, a counter for the cache: recordings are ordered by it.
A recording covers one IST day; finalizing writes each instrument's ticks of it to a
file of its own, once, and never rewrites it. A recorded session is the day's ticks
plus the journal: order updates are not recorded here, because the live session's
journal (`journal`) already holds every one of them.

Recorded ticks cannot be fetched again: the cache must be an absolute folder, nothing is
ever deleted but a raw file whose ticks are finalized and verified, a damaged file is
moved aside, and a copy of the cache is the only backup. How the files are laid out is
in `_tick_files`, `_tick_raw` and `_tick_sidecar`; finalizing in `_tick_finalize`;
reading and repair in `_tick_read`; what a day says about its coverage in
`_tick_spans`; the queue and the writer loop in `_recorder_loop`; how a feed message
becomes a tick in `ticks`. One recording, finalize or repair writes a cache at a time
(`_recorder_lock`). Times are UTC nanoseconds, prices rupees, volume units. Nothing
here prints: the recorder logs to the `openalgo.backtest.live` logger.
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
from ._recorder_summary import FinalizeReport, RecordSummary, log_summary, report_of
from ._tick_files import TICK_DTYPE, ist_day, next_recording_number, ticks_of, ticks_root
from ._tick_finalize import FinalizedFile, RecordingFinalized, finalize_pending
from ._tick_read import MergedTicks, RepairReport, read_merged, repair_day
from ._tick_read import read_day as _read_day
from ._tick_spans import Coverage, Outage, RecordedDay, Span, Stop
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
    `instruments` is `[{"exchange": "NSE", "symbol": "SBIN"}, ...]`, upper-cased as
    OpenAlgo writes them; `mode` one of `RECORD_MODES`. The recording takes the client's
    data callback of that mode: record with a client no live session uses. It connects,
    subscribes and writes every tick until `stop_at_ns` (UTC nanoseconds, at most the
    end of the IST day the recording starts on: one recording is one day; by default
    the latest close of the instruments' exchanges that day, from `calendar`, the
    package's calendar when None), Ctrl+C or a write that fails. Then it writes what it
    holds, closes its raw file, writes how it stopped (its stop, why, its feed
    outages), unsubscribes and disconnects (a second Ctrl+C or an error there is noted),
    removes its hooks and finalizes, with any raw file a crashed recording left that day.
    A Ctrl+C or an error while finalizing is noted too: the raw file stays for a later
    `finalize`, and the summary is still returned.

    `now` is the wall clock (UTC nanoseconds), which stamps the ticks; `monotonic` a
    clock that never steps back, which schedules the writes; `wait` blocks the writer for
    at most a number of seconds (by default until a tick arrives); `cache_dir` is
    resolved as the cache resolves it and must be an absolute folder (a leading `~` is
    expanded): recorded ticks cannot be fetched again. `queue_size` bounds the messages
    waiting for the writer: the feed never waits for it, and a message a full queue
    cannot take is counted (`overflowed`).

    The recording starts (its number, its start sidecar, its raw file, its span's start)
    before it connects, so that a Ctrl+C while connecting or subscribing stops it as one
    while recording does: written as interrupted, and the summary returned.

    Returns the summary, also when a write failed (`stopped_by` is then WRITE_FAILED and
    `error` holds the error). Raises ConfigError for an argument that is not usable (the
    message says which), a stop that is not after the start or after the end of its
    day, or a cache folder that is relative or cannot be used; CalendarError when the
    default stop is asked for and no exchange of the instruments has a session that day
    (or the calendar does not know one); LiveError when another recording holds the
    cache or the feed does not connect; OSError when the raw file cannot be made.
    """
    wanted = instruments_of(instruments)
    subscribe, unsubscribe = feed_calls(client, mode)
    queue_size = queue_size_of(queue_size)
    root = ticks_root(cache_dir)
    first_ns = now()
    stop_ns = stop_instant(stop_at_ns, wanted, first_ns, calendar)
    requests = [{"exchange": exchange, "symbol": symbol} for exchange, symbol in wanted]
    planned = Stop.CLOSE if stop_at_ns is None else Stop.STOP_AT
    with RecorderLock(root):
        recording = Recording(
            root,
            number=next_recording_number(root, ist_day(first_ns)),
            instruments=wanted,
            now=now,
            monotonic=monotonic,
            wait=wait,
            queue_size=queue_size,
        )
        recording.open()
        outcome: list[Stop] = []
        try:
            outcome.append(_run(client, subscribe, requests, recording, stop_ns, planned))
        finally:
            stopped_ns = now()
            _stop(client, unsubscribe, requests, recording, stopped_ns, outcome)
        finalized, error = _finalize(recording)
    summary = _summary(recording, wanted, outcome[0], stopped_ns, finalized, error)
    log_summary(summary)
    return summary


def finalize(day: DateLike, *, cache_dir: str | os.PathLike | None = None) -> FinalizeReport:
    """Finalize the raw files of an IST day (a `datetime.date` or ISO string) that a
    crashed recording left: each instrument's ticks of each recording to a file of its
    own (kept as it is when it is there and verifies), with the recording's span (its
    stop and outages from its stop sidecar, or ended by a crash at its greatest receive
    time), and the raw file removed once every file of it verifies.

    Raises ConfigError for a day that is not a date or a cache folder that is relative
    or cannot be used, LiveError while a recording holds the cache, OSError naming the
    path when a file cannot be read, written or moved.
    """
    date = from_day(to_day(day))
    root = ticks_root(cache_dir)
    notes: list[str] = []
    with RecorderLock(root):
        finalized = finalize_pending(root, date, notes)
    for note in notes:
        logger.warning(note)
    return report_of(finalized, notes)


def repair(day: DateLike, *, cache_dir: str | os.PathLike | None = None) -> RepairReport:
    """Move every damaged finalized file of an IST day aside (never deleting it:
    recorded ticks cannot be fetched again), so that the day's other recordings read;
    a recording whose raw file is still there can then be finalized again. Returns
    every file moved and where it went, and every move that failed (another process may
    hold the file open: close it and repair again), each also logged.

    Raises ConfigError for a day that is not a date or a relative cache folder, LiveError
    while a recording holds the cache (read with `skip_damaged=True` meanwhile), OSError
    naming the path when a file cannot be read.
    """
    date = from_day(to_day(day))
    root = ticks_root(cache_dir)
    with RecorderLock(root):
        report = repair_day(root, date)
    for note in report.notes:
        logger.warning(note)
    for path, why in report.failed:
        logger.error("repair could not move %s aside: %s", path, why)
    return report


def read_day(
    exchange: str,
    symbol: str,
    day: DateLike,
    *,
    cache_dir: str | os.PathLike | None = None,
    skip_damaged: bool = False,
) -> RecordedDay:
    """One instrument's recorded ticks of an IST day from every recording that wrote it:
    the records (`TICK_DTYPE`, in receive order), the recordings' spans with their feed
    outages, and the recordings still pending. Each file is verified against its
    sidecar; nothing is written and no raw file is opened. With `skip_damaged=True` a
    damaged file is left out and listed in `RecordedDay.damaged` instead of raising.

    Raises ConfigError for a day that is not a date or a relative cache folder;
    DataError when no recording of the day was finalized for the instrument, and when
    one of its files is damaged (naming it, left where it is: `repair` moves it aside)
    unless `skip_damaged`; OSError naming the path when a file cannot be read.
    """
    root = ticks_root(cache_dir)
    date = from_day(to_day(day))
    return _read_day(root, exchange, symbol, date, skip_damaged=skip_damaged)


def read_ticks(
    exchange: str, symbol: str, day: DateLike, *, cache_dir: str | os.PathLike | None = None
) -> list[Tick]:
    """One instrument's recorded ticks of an IST day as `Tick`s, in receive order; an
    empty list for a day recorded without a tick. Raises as `read_day` does."""
    recorded = read_day(exchange, symbol, day, cache_dir=cache_dir)
    return ticks_of(recorded.records, recorded.exchange, recorded.symbol)


def read_merged_ticks(
    instruments: Sequence[Mapping[str, str]],
    day: DateLike,
    *,
    cache_dir: str | os.PathLike | None = None,
    skip_damaged: bool = False,
) -> MergedTicks:
    """The recorded ticks of several instruments (`[{"exchange", "symbol"}, ...]`) of an
    IST day, merged in the exact order they were received: by recording number, then
    sequence number, also where receive times tie or the wall clock stepped back. A
    recording pending for any of the instruments is left out for all of them, so the
    merge never holds half a recording, and named in `MergedTicks.left_out`;
    `skip_damaged` is as for `read_day`. Raises as `read_day` does (an instrument with
    nothing finalized is no error while a recording of it is pending), and ConfigError
    for instruments that are not usable."""
    root = ticks_root(cache_dir)
    wanted, date = instruments_of(instruments), from_day(to_day(day))
    return read_merged(root, wanted, date, skip_damaged=skip_damaged)


def _run(
    client: Any,
    subscribe: Callable,
    requests: list[dict[str, str]],
    recording: Recording,
    stop_ns: int,
    planned: Stop,
) -> Stop:
    """Connect, subscribe and record until the stop; why it stopped. A Ctrl+C while
    connecting or subscribing stops it as one while recording does. A feed that does not
    connect raises LiveError, the empty recording's files removed."""
    try:
        client.on_disconnect(recording.on_disconnect)
        client.on_reconnect(recording.on_reconnect)
        if not client.connect():
            recording.discard()
            raise LiveError(
                "the feed did not connect: check that OpenAlgo and its WebSocket proxy run "
                "and the API key is valid; nothing was recorded"
            )
        if not subscribe(requests, recording.on_message):
            logger.warning("the subscription was not sent now: the feed sends it on reconnect")
        written = recording.run_until(stop_ns)
    except KeyboardInterrupt:
        return Stop.INTERRUPT
    return planned if written else Stop.WRITE_FAILED


def _stop(
    client: Any,
    unsubscribe: Callable,
    requests: list[dict[str, str]],
    recording: Recording,
    end_ns: int,
    outcome: list[Stop],
) -> None:
    """Stop a recording so that a second Ctrl+C loses nothing: write what it holds and
    close its raw file, then write how it stopped (each once more if interrupted; not
    when it ended by an error nothing here handles, so that it counts as crashed), then
    unsubscribe and disconnect, each failure noted; the hooks are removed whatever
    happens."""
    try:
        if not _attempt(recording, "write the last ticks", lambda: recording.finish(end_ns)):
            _attempt(recording, "write the last ticks again", lambda: recording.finish(end_ns))
        if outcome:

            def write() -> None:
                recording.write_stop(outcome[0])

            if not _attempt(recording, "write how it stopped", write):
                _attempt(recording, "write how it stopped again", write)
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


def _finalize(recording: Recording) -> tuple[list[RecordingFinalized], Exception | None]:
    """Finalize after the stop. An exception is kept (the summary's `error`) and a Ctrl+C
    noted; either way the raw file stays for a later `finalize`. What finalize found is
    logged."""
    already = len(recording.notes)
    error = None
    try:
        finalized = recording.finalize()
    except (KeyboardInterrupt, Exception) as exc:
        finalized = []
        error = exc if isinstance(exc, Exception) else None
        recording.notes.append(
            f"finalizing was cut short ({type(exc).__name__}: {exc}); the raw file stays, and "
            "a later finalize of the day finalizes it"
        )
    for note in recording.notes[already:]:
        logger.warning(note)
    return finalized, error


def _summary(
    recording: Recording,
    wanted: tuple[tuple[str, str], ...],
    stopped_by: Stop,
    stopped_ns: int,
    finalized: list[RecordingFinalized],
    error: Exception | None,
) -> RecordSummary:
    """The summary. Its tick counts are those of the recording's finalized files, what
    reached the disk, when every instrument's file of it was finalized (else the
    writer's own count of what it wrote)."""
    ours = [done for done in finalized if done.recording == recording.number]
    files = [file for done in ours for file in done.files]
    if len(files) == len(wanted):
        ticks = {(file.exchange, file.symbol): file.ticks for file in files}
    else:
        ticks = recording.ticks
    ticks = {instrument: ticks.get(instrument, 0) for instrument in wanted}
    set_apart = recording.tally.dropped + recording.not_asked
    return RecordSummary(
        started_ns=recording.started_ns,
        stopped_ns=stopped_ns,
        stopped_by=stopped_by,
        recording=recording.number,
        ticks=MappingProxyType(ticks),
        outages=recording.outages,
        files=tuple(file for done in finalized for file in done.files),
        dropped=recording.tally.dropped,
        not_asked=recording.not_asked,
        event_times_rejected=recording.tally.event_times_rejected,
        overflowed=recording.overflowed,
        lost=max(recording.taken - set_apart - sum(ticks.values()), 0),
        rejected=sum(done.rejected for done in ours),
        duplicates=sum(done.duplicates for done in ours),
        error=recording.write_error or error,
        notes=tuple(recording.notes),
    )


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


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "FLUSH_INTERVAL_NS",
    "RECORD_MODES",
    "TICK_DTYPE",
    "Coverage",
    "FinalizeReport",
    "FinalizedFile",
    "MergedTicks",
    "Monotonic",
    "Outage",
    "RecordSummary",
    "RecordedDay",
    "RepairReport",
    "Span",
    "Stop",
    "finalize",
    "read_day",
    "read_merged_ticks",
    "read_ticks",
    "record",
    "repair",
]
