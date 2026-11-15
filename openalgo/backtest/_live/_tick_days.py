"""The finished day files of recorded ticks (private to `recorder`): finalize turns raw
files into them under the recorder's lock, and readers verify them without writing.

Each instrument's folder (see `_tick_files`) holds one `.npy` per IST day, a structured
array of `TICK_DTYPE` without pickle, in receive order: the recordings in the order of
their start, each in its own sequence. Its index (`_tick_index`) holds the file's hash
and the spans of the recordings that wrote it.

Finalizing a raw file merges its records into the day: a record already there (the same
recording and sequence number, from a batch written twice or a raw file merged before a
crash removed it) is kept once. The new day file is written under a new name, then the
index, then the old day file and the raw file are removed; a removal that fails (a file
another process holds open) is noted and left to a later finalize, which removes every
file of the day older than the index's. So a crash at any point leaves the old day and
its raw file, or the new day, and never loses or doubles a tick.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.cache import _quarantine, atomic_write
from ..errors import DataError
from ._tick_files import (
    DAY_SUFFIX,
    RawContent,
    day_bounds,
    day_files,
    instrument_folder,
    instrument_folders,
    instrument_of,
    new_id,
    pending_raw_files,
    raw_files_of,
    read_raw,
    recording_name,
    recording_number,
    ticks_of,
)
from ._tick_index import (
    Days,
    check_day,
    day_entry,
    load_index_locked,
    read_index,
    save_index,
)
from ._tick_spans import Ending, Outage, RecordedDay, Span, joined
from ._types import Tick


@dataclass(frozen=True)
class DayWritten:
    """A day file finalize wrote: its path, instrument and IST day, its tick count (the
    whole day's), and the bytes of torn last records cut off the raw files merged."""

    path: Path
    exchange: str
    symbol: str
    day: dt.date
    ticks: int
    torn_bytes: int


@dataclass(frozen=True)
class RecordingFacts:
    """What a recording knows of itself when it finalizes its own raw files: its start
    and stop (UTC nanoseconds) and its feed outages."""

    start_ns: int
    stop_ns: int
    outages: tuple[Outage, ...]


def finalize_raw(
    path: Path, day: dt.date, facts: Mapping[str, RecordingFacts], notes: list[str]
) -> DayWritten | None:
    """Merge one raw file into its instrument's day file, under the recorder's lock (see
    the module docstring). `facts` holds what the recordings finalizing their own files
    know; any other recording's raw file is one that crashed. Returns None for a file
    that is not a raw file (it is moved aside and noted). Raises OSError naming the path
    when a file cannot be read or written."""
    try:
        raw = read_raw(path)
    except DataError as exc:
        notes.append(f"{exc}; it was moved to {_quarantine(path).name}")
        return None
    folder = path.parent
    exchange, symbol = instrument_of(folder)
    key = day.isoformat()
    days = load_index_locked(folder, exchange, symbol, notes)
    earlier, spans = _earlier(folder, days, key, notes)
    combined = raw.records if earlier is None else np.concatenate([earlier, raw.records])
    records, spans = _in_receive_order(combined, (*spans, _span_of(raw, day, facts)))
    file_name = f"{key}.{new_id()}{DAY_SUFFIX}"
    atomic_write(folder / file_name, lambda handle: np.save(handle, records, allow_pickle=False))
    days[key] = day_entry(file_name, records, spans)
    save_index(folder, exchange, symbol, days)
    _remove_stale(folder, key, file_name, notes)
    _remove(path, "the raw file, merged already,", notes)
    return DayWritten(folder / file_name, exchange, symbol, day, len(records), raw.torn_bytes)


def finalize_day(root: Path, day: dt.date, notes: list[str]) -> list[DayWritten]:
    """Under the recorder's lock: merge every raw file of an IST day (none of them is a
    running recording's, so each is one that crashed), then check every instrument's
    day file of that day, moving a corrupt one aside and removing stale ones. What was
    found on the way is added to `notes`. Raises OSError naming the path when a file
    cannot be read or written."""
    written = [finalize_raw(path, day, {}, notes) for path in raw_files_of(root, day)]
    for folder in instrument_folders(root):
        _check_locked(folder, day.isoformat(), notes)
    return [result for result in written if result is not None]


def read_day(root: Path, exchange: str, symbol: str, day: dt.date) -> RecordedDay:
    """One instrument's verified ticks of an IST day. Never writes. Raises DataError when
    the day was not recorded (or not finalized yet) and when its file or index is
    corrupt (both are left as they are), OSError naming the path when a file cannot be
    read."""
    folder = instrument_folder(root, exchange, symbol)
    key = day.isoformat()
    for attempt in range(2):
        entry = read_index(folder, exchange, symbol).get(key)
        if entry is None:
            raise DataError(_not_recorded(folder, exchange, symbol, day))
        try:
            loaded = check_day(folder, entry)
            break
        except FileNotFoundError:
            # A finalize replaced the day file after the index was read: read it again.
            if attempt:
                raise
    if isinstance(loaded, str):
        raise DataError(
            f"the recorded ticks of {exchange}:{symbol} on {day} in {folder / entry.file} "
            f"failed verification ({loaded}); they are left as they are: a finalize of the day "
            "moves the file aside. Recorded ticks cannot be fetched again: keep a copy of the "
            "cache to restore it from"
        )
    pending = bool(pending_raw_files(folder, day))
    return RecordedDay(exchange, symbol, day, loaded, entry.spans, pending)


def read_merged(root: Path, instruments: Sequence[tuple[str, str]], day: dt.date) -> list[Tick]:
    """The ticks of several instruments' IST day, merged in the order they were received:
    the recordings by their start, each by its sequence numbers. Raises as `read_day`."""
    recorded = [read_day(root, exchange, symbol, day) for exchange, symbol in instruments]
    starts: dict[int, int] = {}
    for one in recorded:
        for span, first_ns in zip(one.spans, _first_receipts(one), strict=True):
            number = recording_number(span.recording)
            known = span.start_ns if span.start_ns is not None else first_ns
            starts[number] = min(starts.get(number, known), known)
    rank = {number: n for n, number in enumerate(sorted(starts, key=lambda n: (starts[n], n)))}
    ranks, seqs, owners, rows = [], [], [], []
    for owner, one in enumerate(recorded):
        ranks.append(np.array([rank[n] for n in one.records["recording"].tolist()], dtype=np.int64))
        seqs.append(one.records["seq"])
        owners.append(np.full(len(one.records), owner))
        rows.append(np.arange(len(one.records)))
    order = np.lexsort((np.concatenate(seqs), np.concatenate(ranks)))
    ticks = [ticks_of(one.records, one.exchange, one.symbol) for one in recorded]
    owner_of, row_of = np.concatenate(owners)[order], np.concatenate(rows)[order]
    return [
        ticks[owner][row] for owner, row in zip(owner_of.tolist(), row_of.tolist(), strict=True)
    ]


def _earlier(
    folder: Path, days: Days, key: str, notes: list[str]
) -> tuple[np.ndarray | None, tuple[Span, ...]]:
    """The day's records and spans finalized so far, checked again under the lock; a
    corrupt day file is moved aside and dropped from the index (its spans with it)."""
    entry = days.get(key)
    if entry is None:
        return None, ()
    loaded = check_day(folder, entry)
    if isinstance(loaded, str):
        _set_aside(folder, days, key, loaded, notes)
        return None, ()
    return loaded, entry.spans


def _span_of(raw: RawContent, day: dt.date, facts: Mapping[str, RecordingFacts]) -> Span:
    """The span a raw file's recording covered on the day, cut to the day."""
    day_start, day_end = day_bounds(day)
    known = facts.get(raw.recording)
    if known is None:
        # A recording that crashed: it ran from its start to its last tick.
        start = max(raw.start_ns, day_start)
        last = int(raw.records["ts_recv_ns"][-1]) if len(raw.records) else start
        return Span(raw.recording, start, max(last, start), Ending.CRASH, None)
    start = max(known.start_ns, day_start)
    stop = max(min(known.stop_ns, day_end), start)
    outages = tuple(outage for outage in known.outages if day_start <= outage.start_ns < day_end)
    return Span(raw.recording, start, stop, Ending.STOP, outages)


def _in_receive_order(
    records: np.ndarray, spans: Sequence[Span]
) -> tuple[np.ndarray, tuple[Span, ...]]:
    """The records in receive order with each (recording, seq) once, and their spans in
    the order of their start (a span whose start is not known by its first tick)."""
    spans = joined(spans)
    numbers = records["recording"]
    named = {recording_number(span.recording) for span in spans}
    for number in dict.fromkeys(numbers.tolist()):
        if number not in named:  # records of a recording no span names: times unknown
            spans.append(Span(recording_name(number), None, None, Ending.UNKNOWN, None))
    first_ns = _first_receipt_by_number(records)

    def started(span: Span) -> tuple[float, str]:
        start = (
            span.start_ns
            if span.start_ns is not None
            else first_ns.get(recording_number(span.recording), float("inf"))
        )
        return start, span.recording

    ordered = tuple(sorted(spans, key=started))
    rank = {recording_number(span.recording): n for n, span in enumerate(ordered)}
    ranks = np.array([rank[number] for number in numbers.tolist()], dtype=np.int64)
    order = np.lexsort((records["seq"], ranks))
    records, ranks = records[order], ranks[order]
    first_of_kind = np.ones(len(records), dtype=bool)
    first_of_kind[1:] = (ranks[1:] != ranks[:-1]) | (records["seq"][1:] != records["seq"][:-1])
    return records[first_of_kind], ordered


def _first_receipt_by_number(records: np.ndarray) -> dict[int, int]:
    numbers, first = np.unique(records["recording"], return_index=True)
    receipts = records["ts_recv_ns"][first]
    return dict(zip(numbers.tolist(), receipts.tolist(), strict=True))


def _first_receipts(recorded: RecordedDay) -> list[float]:
    """Per span of a day, the receive time of its recording's first tick (infinity when
    it has none)."""
    first = _first_receipt_by_number(recorded.records)
    return [first.get(recording_number(span.recording), float("inf")) for span in recorded.spans]


def _check_locked(folder: Path, key: str, notes: list[str]) -> None:
    """Under the lock: check an instrument's day file of `key`; move a corrupt one aside,
    remove the stale ones."""
    exchange, symbol = instrument_of(folder)
    days = load_index_locked(folder, exchange, symbol, notes)
    entry = days.get(key)
    if entry is None:
        return
    loaded = check_day(folder, entry)
    if isinstance(loaded, str):
        _set_aside(folder, days, key, loaded, notes)
    else:
        _remove_stale(folder, key, entry.file, notes)


def _set_aside(folder: Path, days: Days, key: str, problem: str, notes: list[str]) -> None:
    """Move a corrupt day file aside (never delete it), drop its day from the index and
    note it."""
    path = folder / days[key].file
    moved = _quarantine(path)
    del days[key]
    exchange, symbol = instrument_of(folder)
    save_index(folder, exchange, symbol, days)
    notes.append(
        f"the recorded ticks of {key} in {path} failed verification ({problem}); the file was "
        f"moved to {moved.name}, and that day is no longer read"
    )


def _remove_stale(folder: Path, key: str, current: str, notes: list[str]) -> None:
    """Remove the files of the day `key` older than its current day file `current`."""
    current_ns = (folder / current).stat().st_mtime_ns
    for path in day_files(folder, key):
        if path.name != current and path.stat().st_mtime_ns <= current_ns:
            _remove(path, "the replaced day file", notes)


def _remove(path: Path, what: str, notes: list[str]) -> None:
    """Remove a file whose content is kept elsewhere; a failure is noted, and a later
    finalize removes it."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        notes.append(
            f"{what} {path.name} could not be removed ({exc}); a later finalize removes it"
        )


def _not_recorded(folder: Path, exchange: str, symbol: str, day: dt.date) -> str:
    waiting = bool(pending_raw_files(folder, day))
    advice = (
        f"its raw file is waiting: finalize {day} first (a recording finalizes when it stops)"
        if waiting
        else "record that day first"
    )
    return f"no ticks of {exchange}:{symbol} were recorded on {day}: {advice}"


__all__ = [
    "DayWritten",
    "RecordingFacts",
    "finalize_day",
    "finalize_raw",
    "read_day",
    "read_merged",
]
