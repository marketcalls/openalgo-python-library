"""The writing side of a recording (private to `recorder`): the queue the feed's thread
fills, the outages its hooks note, and the one writer loop that takes the ticks off the
queue into raw files.

The feed's callbacks run on its thread and only queue or note. The data callback puts
(message, the instant it ran) on a bounded queue without ever waiting, and counts a
message the full queue cannot take; the disconnect and reconnect hooks note when an
outage started and ended. The writer loop runs on the thread that called `record`: it
takes each message off the queue, stamps `ts_recv_ns` at that instant (the session loop
does the same, so a replay sees what a session saw), adapts it (`ticks.tick_from_feed`)
and keeps its record for the raw file of its instrument and IST day. Every
`FLUSH_INTERVAL_NS` it appends what it kept to the files and hands them to the system,
so a crash of the process loses at most that much.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._tick_days import DayWritten, Outage, finalize_raw
from ._tick_files import RawFile, ist_day, raw_path, records_of, tick_row
from ._types import Now, Sleep
from .ticks import TickTally, tick_from_feed

#: How often the writer hands what it kept to the files: at least once a second, so a
#: crash loses at most the last second of ticks.
FLUSH_INTERVAL_NS = NANOS_PER_SECOND
#: Messages the queue between the feed and the writer holds: 50 seconds of a feed that
#: delivers 1000 ticks a second, about 100 MB at most. The writer takes them off far
#: faster than a feed delivers, so only a writer held up (a stalled disk) fills it.
DEFAULT_QUEUE_SIZE = 50_000

Key = tuple[str, str, dt.date]


class Recording:
    """One recording's queue, outages and raw files (see the module docstring).

    `root` is the folder of recorded ticks, `now` the clock (UTC nanoseconds), `wait`
    what blocks the writer for at most a number of seconds (by default until the feed
    queues a message or the time is up). Not reusable: one recording, then `finish`.
    """

    def __init__(self, root: Path, *, now: Now, wait: Sleep | None = None, queue_size: int) -> None:
        self._root = root
        self._now = now
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
        self._kept: dict[Key, list[tuple]] = {}
        self._ticks: dict[tuple[str, str], int] = {}
        self._earlier: list[DayWritten] = []
        self.notes: list[str] = []
        self._flushed_ns = now()

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

    def open(self, instruments: Iterable[tuple[str, str]], day: dt.date) -> None:
        """Create the raw file of each (exchange, symbol) for the IST day, so that every
        instrument recorded gets its day file, one with no ticks too. Raises OSError when
        a file cannot be made."""
        for exchange, symbol in instruments:
            self._file((exchange, symbol, day))

    def run_until(self, stop_ns: int) -> None:
        """Take the queued ticks into the files until the clock reaches `stop_ns`. A
        KeyboardInterrupt raised while waiting goes to the caller."""
        while True:
            self._arrived.clear()
            self._take_queued()
            now_ns = self._now()
            if now_ns - self._flushed_ns >= FLUSH_INTERVAL_NS:
                self._flush(now_ns)
            if now_ns >= stop_ns:
                return
            until_ns = min(self._flushed_ns + FLUSH_INTERVAL_NS, stop_ns)
            self._wait((until_ns - now_ns) / NANOS_PER_SECOND)

    def finish(self, end_ns: int) -> None:
        """End the recording at `end_ns`: an outage still under way ends there, what is
        still queued is taken, everything is handed to the files and they are closed."""
        self._finished = True
        with self._lock:
            if self._down_since_ns is not None:
                self._outages.append(Outage(self._down_since_ns, end_ns))
                self._down_since_ns = None
        try:
            self._take_queued()
            self._flush(end_ns)
        finally:
            for file in self._files.values():
                file.close()

    def finalize(self) -> list[DayWritten]:
        """Turn this recording's raw files into day files, each with the outages that
        started on its day, after `finish`. Returns them after the day files of earlier
        recordings this one finalized when it found their raw files. Raises OSError when a
        file cannot be written."""
        written = list(self._earlier)
        outages = self.outages
        for (_, _, day), file in self._files.items():
            of_day = tuple(outage for outage in outages if ist_day(outage.start_ns) == day)
            result = finalize_raw(file.path, day, of_day, self.notes)
            if result is not None:
                written.append(result)
        return written

    # ----- what the recording counted -------------------------------------------------

    @property
    def ticks(self) -> dict[tuple[str, str], int]:
        """Ticks recorded per (exchange, symbol)."""
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

    # ----- inside ------------------------------------------------------------------------

    def _wait_for_arrival(self, seconds: float) -> None:
        self._arrived.wait(seconds)

    def _take_queued(self) -> None:
        while True:
            try:
                message, queued_ns = self._queue.get_nowait()
            except queue.Empty:
                return
            tick = tick_from_feed(message, recv_ns=self._now(), tally=self._tally)
            if tick is None:
                continue
            instrument = (tick.exchange, tick.symbol)
            key = (*instrument, ist_day(tick.ts_recv_ns))
            self._kept.setdefault(key, []).append(tick_row(tick, queued_ns))
            self._ticks[instrument] = self._ticks.get(instrument, 0) + 1

    def _flush(self, now_ns: int) -> None:
        # Each batch leaves `_kept` before it is written: a Ctrl+C in the middle of a
        # flush, after which `finish` flushes again, can never write a batch twice.
        while self._kept:
            key, rows = self._kept.popitem()
            self._file(key).append(records_of(rows))
        self._flushed_ns = now_ns

    def _file(self, key: Key) -> RawFile:
        file = self._files.get(key)
        if file is None:
            path = raw_path(self._root, *key)
            if path.exists():
                # An earlier recording stopped without finalizing this day: its ticks go
                # to the day file first (their outages are not known), so this recording
                # writes a raw file of its own.
                earlier = finalize_raw(path, key[2], None, self.notes)
                self._earlier.extend([] if earlier is None else [earlier])
            file = self._files[key] = RawFile.create(path)
        return file


__all__ = ["DEFAULT_QUEUE_SIZE", "FLUSH_INTERVAL_NS", "Recording"]
