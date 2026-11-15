"""The writing side of a recording (private to `recorder`): the queue the feed's thread
fills, the outages its hooks note, and the one writer loop that takes the ticks off the
queue into raw files.

The feed's callbacks run on its thread and only queue or note. The data callback puts
(message, the instant it ran) on a bounded queue without ever waiting, and counts a
message the full queue cannot take; the disconnect and reconnect hooks note when an
outage started and ended. The writer loop runs on the thread that called `record`: it
takes each message off the queue, numbers it (the recording's sequence, across all
symbols), stamps `ts_recv_ns` from the wall clock at that instant (the session loop does
the same, so a replay sees what a session saw), adapts it (`ticks.tick_from_feed`) and
keeps its record for the raw file of its instrument and IST day.

Every `FLUSH_INTERVAL_NS` on the monotonic clock (a wall clock stepped back never
holds a flush up) it writes what it kept. A batch leaves memory only once it is written,
so a Ctrl+C or a failed write in the middle loses nothing; a batch written twice that
way is kept once when the day is finalized. A write that fails stops the recording; the
final flush tries once more, on a new raw file if the torn end of the old one cannot be
cut off, and what still cannot be written is counted as lost.
"""

from __future__ import annotations

import datetime as dt
import logging
import queue
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._tick_days import DayWritten, RecordingFacts, finalize_raw
from ._tick_files import (
    RawFile,
    instrument_folder,
    ist_day,
    new_id,
    pending_raw_files,
    recording_number,
    records_of,
    tick_row,
)
from ._tick_spans import Outage
from ._types import LOGGER_NAME, Now, Sleep
from .ticks import TickTally, tick_from_feed

#: A monotonic clock in nanoseconds: it never steps back, whatever the wall clock does.
Monotonic = Callable[[], int]

#: How often the writer writes what it kept, on the monotonic clock: at least once a
#: second, so a crash loses at most the last second of ticks.
FLUSH_INTERVAL_NS = NANOS_PER_SECOND
#: Messages the queue between the feed and the writer holds: 50 seconds of a feed that
#: delivers 1000 ticks a second, about 100 MB at most. The writer takes them off far
#: faster than a feed delivers, so only a writer held up (a stalled disk) fills it.
DEFAULT_QUEUE_SIZE = 50_000

Key = tuple[str, str, dt.date]

logger = logging.getLogger(LOGGER_NAME)


class Recording:
    """One recording's queue, outages and raw files (see the module docstring).

    `root` is the folder of recorded ticks; `instruments` the (exchange, symbol) pairs
    recorded; `now` the wall clock (UTC nanoseconds) and `monotonic` a clock that never
    steps back; `wait` what blocks the writer for at most a number of seconds (by default
    until the feed queues a message or the time is up). One recording, then `finish`.
    """

    def __init__(
        self,
        root: Path,
        *,
        instruments: Iterable[tuple[str, str]],
        now: Now,
        monotonic: Monotonic,
        wait: Sleep | None = None,
        queue_size: int,
    ) -> None:
        self.recording = new_id()
        self.started_ns: int | None = None
        self.stopped_ns: int | None = None
        self.notes: list[str] = []
        self.write_error: OSError | None = None
        self._root = root
        self._instruments = tuple(instruments)
        self._number = recording_number(self.recording)
        self._now, self._monotonic = now, monotonic
        self._queue: queue.Queue[tuple[Any, int]] = queue.Queue(maxsize=queue_size)
        self._arrived = threading.Event()
        self._wait = self._wait_for_arrival if wait is None else wait
        self._lock = threading.Lock()  # guards what the feed's thread writes
        self._finished = False
        self._overflowed = 0
        self._down_since_ns: int | None = None
        self._outages: list[Outage] = []
        self._tally = TickTally()
        self._files: dict[Key, RawFile] = {}
        self._retired: list[tuple[Key, RawFile]] = []  # files a torn end ended
        self._days: set[dt.date] = set()
        self._kept: dict[Key, list[tuple]] = {}
        self._ticks: dict[tuple[str, str], int] = {}
        self._seq = self._taken = self._written = 0
        self._failed_paths: set[Path] = set()
        self._flushed_at = monotonic()

    # ----- the feed's thread: queue or note, never wait --------------------------------

    def on_message(self, message: Any) -> None:
        """The data callback: queue the message with the instant it arrived. A full queue
        drops it and counts it; after `finish` nothing is queued."""
        if self._finished:
            return
        try:
            self._queue.put_nowait((message, self._now()))
        except queue.Full:
            with self._lock:
                self._overflowed += 1
        self._arrived.set()

    def on_disconnect(self, event: Mapping[str, Any]) -> None:
        """The feed's disconnect hook: an outage starts, unless `disconnect()` was asked
        for (the end of the recording)."""
        if isinstance(event, Mapping) and event.get("requested"):
            return
        with self._lock:
            if self._down_since_ns is None:
                self._down_since_ns = self._now()

    def on_reconnect(self, event: Mapping[str, Any]) -> None:
        """The feed's reconnect hook: the outage under way ends."""
        with self._lock:
            if self._down_since_ns is not None:
                self._outages.append(Outage(self._down_since_ns, self._now()))
                self._down_since_ns = None

    # ----- the writer's thread --------------------------------------------------------

    def open(self) -> None:
        """Start the recording now: create the raw file of each instrument for today, so
        that every instrument recorded gets its day file, one with no ticks too. Raises
        OSError when a file cannot be made."""
        self.started_ns = self._now()
        self._open_day(ist_day(self.started_ns))

    def run_until(self, stop_ns: int) -> bool:
        """Take the queued ticks into the files until the wall clock reaches `stop_ns`.
        Returns False when a write failed (`write_error`), which stops the recording. A
        KeyboardInterrupt goes to the caller."""
        while True:
            self._arrived.clear()
            self._take_queued()
            if self._monotonic() - self._flushed_at >= FLUSH_INTERVAL_NS and not self._flush():
                return False
            now_ns = self._now()
            if now_ns >= stop_ns:
                return True
            flush_in_ns = self._flushed_at + FLUSH_INTERVAL_NS - self._monotonic()
            self._wait(max(min(flush_in_ns, stop_ns - now_ns), 0) / NANOS_PER_SECOND)

    def finish(self, end_ns: int) -> None:
        """End the recording at `end_ns` (the first call decides it): an outage still
        under way ends there, what is still queued is taken, and what was kept is
        written. A second call writes what the first could not (after a Ctrl+C in the
        middle). Then `close`."""
        if not self._finished:
            self._finished = True
            self.stopped_ns = end_ns
            with self._lock:
                if self._down_since_ns is not None:
                    self._outages.append(Outage(self._down_since_ns, end_ns))
                    self._down_since_ns = None
        self._take_queued()
        self._flush()
        self.close()

    def close(self) -> None:
        """Close every raw file not closed yet; a failure is noted."""
        for file in (*self._files.values(), *(file for _, file in self._retired)):
            try:
                file.close()
            except OSError as exc:
                self._note(f"{file.path.name} could not be closed ({exc})")

    def finalize(self) -> list[DayWritten]:
        """Turn this recording's raw files (with its span and outages) into day files,
        after `finish`, together with any raw file a crashed recording left for the same
        instruments and days. One result per instrument and day. A file that cannot be
        read or written is noted (`write_error` keeps the first such error) and its raw
        file stays for a later finalize."""
        facts = {self.recording: RecordingFacts(self.started_ns, self.stopped_ns, self.outages)}
        results: dict[Key, DayWritten] = {}
        for key in dict.fromkeys((*self._files, *(key for key, _ in self._retired))):
            folder = instrument_folder(self._root, key[0], key[1])
            for path in pending_raw_files(folder, key[2]):
                try:
                    written = finalize_raw(path, key[2], facts, self.notes)
                except OSError as exc:
                    self._failed(path, exc)
                    continue
                if written is not None:
                    torn = written.torn_bytes + getattr(results.get(key), "torn_bytes", 0)
                    results[key] = DayWritten(written.path, *key[:2], key[2], written.ticks, torn)
        return list(results.values())

    # ----- what the recording counted -------------------------------------------------

    @property
    def ticks(self) -> dict[tuple[str, str], int]:
        """Ticks written per (exchange, symbol)."""
        return dict(self._ticks)

    @property
    def outages(self) -> tuple[Outage, ...]:
        """The outages that ended so far, in order."""
        with self._lock:
            return tuple(self._outages)

    @property
    def overflowed(self) -> int:
        """Messages dropped because the queue was full."""
        with self._lock:
            return self._overflowed

    @property
    def tally(self) -> TickTally:
        """What the adapter could not use."""
        return self._tally

    @property
    def lost(self) -> int:
        """Messages taken off the queue that were neither written nor dropped as not
        adaptable: a write that kept failing, a Ctrl+C between taking and keeping."""
        return self._taken - self._tally.dropped - self._written

    # ----- inside ------------------------------------------------------------------------

    def _wait_for_arrival(self, seconds: float) -> None:
        self._arrived.wait(seconds)

    def _take_queued(self) -> None:
        while True:
            try:
                message, queued_ns = self._queue.get_nowait()
            except queue.Empty:
                return
            self._taken += 1
            self._seq += 1
            seq = self._seq
            tick = tick_from_feed(message, recv_ns=self._now(), tally=self._tally)
            if tick is None:
                continue
            key = (tick.exchange, tick.symbol, ist_day(tick.ts_recv_ns))
            row = tick_row(tick, recording=self._number, seq=seq, queued_ns=queued_ns)
            self._kept.setdefault(key, []).append(row)

    def _flush(self) -> bool:
        """Write every kept batch; each leaves memory once written. Returns False when a
        write failed (the first error is kept as `write_error`)."""
        whole = True
        for key in list(self._kept):
            rows = self._kept[key]
            try:
                self._write(key, rows)
            except OSError as exc:
                self._failed(self._where(key), exc)
                whole = False
                continue
            del self._kept[key]
            self._written += len(rows)
            self._ticks[key[:2]] = self._ticks.get(key[:2], 0) + len(rows)
        self._flushed_at = self._monotonic()
        return whole

    def _write(self, key: Key, rows: list[tuple]) -> None:
        records = records_of(rows)
        file = self._file(key)
        try:
            file.append(records)
        except OSError:
            if not file.cut_failed:
                raise
            # Its torn end could not be cut off: nothing more is written after it.
            self._retire(key)
            self._file(key).append(records)

    def _file(self, key: Key) -> RawFile:
        if key[2] not in self._days:
            self._open_day(key[2])
        file = self._files.get(key)
        if file is None:
            file = self._files[key] = self._create(key)
        return file

    def _open_day(self, day: dt.date) -> None:
        """Create the raw files of every instrument for a day the recording reaches."""
        self._days.add(day)
        for exchange, symbol in self._instruments:
            key = (exchange, symbol, day)
            if key not in self._files:
                self._files[key] = self._create(key)

    def _create(self, key: Key) -> RawFile:
        folder = instrument_folder(self._root, key[0], key[1])
        return RawFile.create(folder, key[2], recording=self.recording, start_ns=self.started_ns)

    def _retire(self, key: Key) -> None:
        file = self._files.pop(key)
        self._retired.append((key, file))
        self._note(
            f"the end a failed write tore off {file.path.name} could not be cut off: writing "
            "goes on in a new raw file"
        )
        try:
            file.close()
        except OSError as exc:
            self._note(f"{file.path.name} could not be closed ({exc})")

    def _failed(self, path: Path | None, exc: OSError) -> None:
        """Note a failed write or finalize; the first error is kept."""
        if self.write_error is None:
            self.write_error = exc
        if path not in self._failed_paths:
            self._failed_paths.add(path)
            logger.error("recorded ticks could not be written to %s: %s", path, exc)
        self.notes.append(f"recorded ticks could not be written to {path} ({exc})")

    def _note(self, text: str) -> None:
        logger.warning(text)
        self.notes.append(text)

    def _where(self, key: Key) -> Path:
        """The raw file a batch of `key` goes to, or its folder before there is one."""
        file = self._files.get(key)
        return file.path if file is not None else instrument_folder(self._root, *key[:2])


__all__ = ["DEFAULT_QUEUE_SIZE", "FLUSH_INTERVAL_NS", "Monotonic", "Recording"]
