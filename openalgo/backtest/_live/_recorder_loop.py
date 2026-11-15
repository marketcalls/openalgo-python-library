"""The writing side of a recording (private to `recorder`): the queue the feed's thread
fills, the outages its hooks note, and the one writer loop that takes the ticks off the
queue into the recording's raw file.

The feed's callbacks run on its thread and only queue or note. The data callback puts
(message, the instant it ran) on a bounded queue without ever waiting, and counts a
message the full queue cannot take; the disconnect and reconnect hooks note when an
outage started and ended. The writer loop runs on the thread that called `record`: it
takes each message off the queue, numbers it (the recording's sequence from 1, across
all instruments), stamps `ts_recv_ns` from the wall clock at that instant (the session
loop does the same, so a replay sees what a session saw), adapts it
(`ticks.tick_from_feed`) and keeps its record for the raw file.

Every `FLUSH_INTERVAL_NS` on the monotonic clock (a wall clock stepped back never holds
a flush up) it appends what it kept and syncs it to the disk. A batch leaves memory only
once it is written, so a Ctrl+C or a failed write in the middle loses nothing; a batch
written twice that way is kept once (its sequence numbers do not increase). A write that
fails stops the recording; the final flush tries once more, in the recording's next raw
file part if the torn end cannot be cut off, and what still cannot be written is lost
and counted.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._tick_files import instrument_key, ist_day, raw_path, records_of, tick_row
from ._tick_finalize import RecordingFinalized, StopFacts, finalize_pending, write_stop
from ._tick_raw import RawFile, RawHeader
from ._tick_spans import Outage, Stop
from ._types import LOGGER_NAME, Now, Sleep
from .ticks import TickTally, tick_from_feed

#: A monotonic clock in nanoseconds: it never steps back, whatever the wall clock does.
Monotonic = Callable[[], int]

#: How often the writer writes what it kept, on the monotonic clock: at least once a
#: second, so a crash (or a power loss: each write is synced) loses at most a second.
FLUSH_INTERVAL_NS = NANOS_PER_SECOND
#: Messages the queue between the feed and the writer holds: 50 seconds of a feed that
#: delivers 1000 ticks a second, about 100 MB at most. The writer takes them off far
#: faster than a feed delivers, so only a writer held up (a stalled disk) fills it.
DEFAULT_QUEUE_SIZE = 50_000

logger = logging.getLogger(LOGGER_NAME)


class Recording:
    """One recording's queue, outages and raw file (see the module docstring).

    `root` is the folder of recorded ticks and `number` the recording's number (taken
    under the recorder's lock); `instruments` the (exchange, symbol) pairs recorded, in
    upper case; `now` the wall clock (UTC nanoseconds) and `monotonic` a clock that
    never steps back; `wait` what blocks the writer for at most a number of seconds (by
    default until the feed queues a message or the time is up). One recording, then
    `finish`.
    """

    def __init__(
        self,
        root: Path,
        *,
        number: int,
        instruments: Iterable[tuple[str, str]],
        now: Now,
        monotonic: Monotonic,
        wait: Sleep | None = None,
        queue_size: int,
    ) -> None:
        self.number = number
        self.started_ns: int | None = None
        self.stopped_ns: int | None = None
        self.notes: list[str] = []
        self.write_error: OSError | None = None
        self._root = root
        self._instruments = tuple(instruments)
        self._index = {instrument: n for n, instrument in enumerate(self._instruments)}
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
        self._header: RawHeader | None = None
        self._file: RawFile | None = None
        self._parts: list[RawFile] = []
        self._kept: list[tuple] = []
        self._ticks: dict[tuple[str, str], int] = {}
        self._not_asked: dict[tuple[str, str], int] = {}
        self._seq = self._taken = self._written = 0
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
        """Start the recording now: create its raw file, whose header lists the
        instruments. Raises OSError when it cannot be made."""
        self.started_ns = self._now()
        day = ist_day(self.started_ns)
        self._header = RawHeader(self.number, 0, day, self.started_ns, self._instruments)
        self._file = self._create(0)

    def run_until(self, stop_ns: int) -> bool:
        """Take the queued ticks into the raw file until the wall clock reaches
        `stop_ns`. Returns False when a write failed (`write_error`), which stops the
        recording. A KeyboardInterrupt goes to the caller."""
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
            self.stopped_ns = end_ns
            self._finished = True
            with self._lock:
                if self._down_since_ns is not None:
                    self._outages.append(Outage(self._down_since_ns, end_ns))
                    self._down_since_ns = None
        self._take_queued()
        self._flush()
        self.close()

    def write_stop(self, stopped_by: Stop) -> None:
        """Write how the recording stopped beside its raw file, after `finish`, so that a
        finalize knows its stop and outages. Raises OSError when it cannot be written."""
        if self._header is not None:
            facts = StopFacts(stopped_by, self.stopped_ns, self.outages)
            write_stop(self._root, self._header.day, self.number, facts)

    def close(self) -> None:
        """Close every raw file part not closed yet; a failure is noted."""
        for part in self._parts:
            try:
                part.close()
            except OSError as exc:
                self._note(f"{part.path.name} could not be closed ({exc})")

    def finalize(self) -> list[RecordingFinalized]:
        """Finalize this recording and any other its day left pending (a crashed one),
        after `finish`. Raises OSError naming the path when a file cannot be read,
        written or moved; what was found on the way is in `notes`."""
        if self._header is None:
            return []
        return finalize_pending(self._root, self._header.day, self.notes)

    # ----- what the recording counted -------------------------------------------------

    @property
    def ticks(self) -> dict[tuple[str, str], int]:
        """Ticks written per (exchange, symbol)."""
        return dict(self._ticks)

    @property
    def not_asked(self) -> int:
        """Ticks of an instrument the recording did not ask for: not recorded."""
        return sum(self._not_asked.values())

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
        """Messages taken off the queue that were neither written nor set apart (not
        adaptable, or of an instrument not asked for): writes that kept failing, a
        Ctrl+C between taking and keeping."""
        return self._taken - self._tally.dropped - self.not_asked - self._written

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
            instrument = instrument_key(tick.exchange, tick.symbol)
            index = self._index.get(instrument)
            if index is None:
                self._not_asked_for(instrument)
                continue
            row = tick_row(
                tick, instrument=index, recording=self.number, seq=seq, queued_ns=queued_ns
            )
            self._kept.append(row)

    def _flush(self) -> bool:
        """Write the kept batch; it leaves memory once written. Returns False when the
        write failed (the first error is kept as `write_error`)."""
        if self._kept:
            batch = self._kept
            try:
                self._write(records_of(batch))
            except OSError as exc:
                self._failed(exc)
                return False
            self._kept = []
            self._written += len(batch)
            for row in batch:
                instrument = self._instruments[row[0]]
                self._ticks[instrument] = self._ticks.get(instrument, 0) + 1
        self._flushed_at = self._monotonic()
        return True

    def _write(self, records) -> None:
        try:
            self._file.append(records)
        except OSError:
            if not self._file.cut_failed:
                raise
            # Its torn end could not be cut off: nothing more is written after it.
            self._note(
                f"the end a failed write tore off {self._file.path.name} could not be cut off: "
                "writing goes on in a new raw file part"
            )
            self._file = self._create(len(self._parts))
            self._file.append(records)

    def _create(self, part: int) -> RawFile:
        header = RawHeader(
            self.number, part, self._header.day, self._header.start_ns, self._instruments
        )
        file = RawFile.create(raw_path(self._root, header.day, self.number, part), header)
        self._parts.append(file)
        return file

    def _not_asked_for(self, instrument: tuple[str, str]) -> None:
        if instrument not in self._not_asked:
            self._note(f"ticks of {instrument[0]}:{instrument[1]} arrived unasked: not recorded")
        self._not_asked[instrument] = self._not_asked.get(instrument, 0) + 1

    def _failed(self, exc: OSError) -> None:
        """Note a failed write; the first error is kept."""
        if self.write_error is None:
            self.write_error = exc
            logger.error("recorded ticks could not be written to %s: %s", self._file.path, exc)
        self.notes.append(f"recorded ticks could not be written to {self._file.path} ({exc})")

    def _note(self, text: str) -> None:
        logger.warning(text)
        self.notes.append(text)


__all__ = ["DEFAULT_QUEUE_SIZE", "FLUSH_INTERVAL_NS", "Monotonic", "Recording"]
