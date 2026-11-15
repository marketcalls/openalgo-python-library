"""Reading recorded ticks back (private to `recorder`), and the one repair.

Reading an instrument's IST day lists its finalized files of that day (one per
recording), checks each against its sidecar (`_tick_sidecar.check`) and merges them by
(recording number, sequence number): the exact order a session received them. Readers
never write and never open a raw file: what is pending (a recording running, or one
that crashed or was cut short while finalizing) comes from the directory listing and
the recordings' start sidecars. A damaged file is reported with its path and left where
it is; with `skip_damaged=True` a reader returns what verifies and lists the damaged
files instead, never skipping one silently. `repair`, under the recorder's lock, moves
damaged files aside (`_tick_files.set_aside`, never a deletion) and reports every file
it moved and every move that failed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DataError
from ._tick_files import (
    TICK_DTYPE,
    finalized_numbers,
    finalized_paths,
    instrument_folder,
    instrument_folders,
    instrument_key,
    instrument_of,
    raw_parts_of,
    set_aside,
    start_path,
    ticks_of,
)
from ._tick_finalize import StartFacts, read_start
from ._tick_sidecar import State, check
from ._tick_spans import RecordedDay, Span
from ._types import Tick

_NO_SIDECAR = "it has no sidecar, so it cannot be verified"


@dataclass(frozen=True)
class MergedTicks:
    """Several instruments' ticks of a day merged in the order they were received.
    `left_out` names the recordings pending for any of the instruments (not finalized
    for all of them yet), whose ticks are left out for every instrument so that the
    merge never holds half a recording; `damaged` lists the files skipped as damaged
    (only with `skip_damaged=True`)."""

    ticks: list[Tick]
    left_out: tuple[int, ...]
    damaged: tuple[Path, ...]


@dataclass(frozen=True)
class RepairReport:
    """What `repair` did: where each file it moved aside went, each file it could not
    move with why (another process may hold it open: repair again), and its notes."""

    moved: tuple[Path, ...]
    failed: tuple[tuple[Path, str], ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _DayParts:
    records: np.ndarray
    spans: tuple[Span, ...]
    pending: tuple[int, ...]
    damaged: tuple[Path, ...]


def read_day(
    root: Path, exchange: str, symbol: str, day: dt.date, *, skip_damaged: bool = False
) -> RecordedDay:
    """One instrument's verified ticks of an IST day, from every recording that wrote
    it. Never writes. Raises DataError when no recording of the day was finalized for it,
    and when one of its files is damaged (naming it, left where it is) unless
    `skip_damaged`; OSError naming the path when a file cannot be read."""
    exchange, symbol = instrument_key(exchange, symbol)
    parts = _parts(root, exchange, symbol, day, skip_damaged)
    if not parts.spans and not parts.damaged:
        raise DataError(_not_recorded(exchange, symbol, day, bool(parts.pending)))
    return RecordedDay(
        exchange, symbol, day, parts.records, parts.spans, parts.pending, parts.damaged
    )


def read_merged(
    root: Path,
    instruments: Sequence[tuple[str, str]],
    day: dt.date,
    *,
    skip_damaged: bool = False,
) -> MergedTicks:
    """The ticks of several instruments' IST day, merged in the order they were
    received (by recording number, then sequence number), without any recording that is
    pending for one of them. Raises as `read_day`; an instrument with nothing finalized
    is no error while a recording of it is pending."""
    keys = [instrument_key(exchange, symbol) for exchange, symbol in instruments]
    parts = [_parts(root, exchange, symbol, day, skip_damaged) for exchange, symbol in keys]
    for (exchange, symbol), of in zip(keys, parts, strict=True):
        if not (of.spans or of.damaged or of.pending):
            raise DataError(_not_recorded(exchange, symbol, day, False))
    left_out = tuple(sorted({number for of in parts for number in of.pending}))
    kept = [of.records[~np.isin(of.records["recording"], left_out)] for of in parts]
    if not kept:
        return MergedTicks([], left_out, ())
    recordings = np.concatenate([records["recording"] for records in kept])
    seqs = np.concatenate([records["seq"] for records in kept])
    owners = np.concatenate([np.full(len(records), n) for n, records in enumerate(kept)])
    rows = np.concatenate([np.arange(len(records)) for records in kept])
    order = np.lexsort((seqs, recordings))
    ticks = [ticks_of(records, *key) for records, key in zip(kept, keys, strict=True)]
    pairs = zip(owners[order].tolist(), rows[order].tolist(), strict=True)
    damaged = tuple(path for of in parts for path in of.damaged)
    return MergedTicks([ticks[owner][row] for owner, row in pairs], left_out, damaged)


def repair_day(root: Path, day: dt.date) -> RepairReport:
    """Under the recorder's lock: move every damaged finalized file of an IST day aside
    (its ticks and its sidecar), and a ticks file without a sidecar whose recording has
    no raw file left to finalize it. A move that fails is reported and the others are
    still made. Raises OSError naming the path when a file cannot be read."""
    moved, failed, notes = [], [], []
    pending = raw_parts_of(root, day)
    for folder in instrument_folders(root):
        for number in finalized_numbers(folder, day):
            ticks_path, sidecar_path = finalized_paths(folder, day, number)
            found = check(ticks_path, sidecar_path, day, number)
            if found.state is State.VERIFIED or (found.state is State.ABSENT and number in pending):
                continue
            exchange, symbol = instrument_of(folder)
            notes.append(
                f"the finalized ticks of {exchange}:{symbol} of recording {number} on {day} are "
                f"damaged ({found.problem or _NO_SIDECAR})"
            )
            for path in (ticks_path, sidecar_path):
                if not path.exists():
                    continue
                try:
                    moved.append(set_aside(path))
                except OSError as exc:
                    failed.append((path, str(exc)))
                    notes.append(f"{path} could not be moved aside ({exc}): repair again")
                    continue
                notes.append(f"{path} was moved aside to {moved[-1].name}")
    return RepairReport(tuple(moved), tuple(failed), tuple(notes))


def _parts(root: Path, exchange: str, symbol: str, day: dt.date, skip_damaged: bool) -> _DayParts:
    """The verified records and spans of an instrument's day, what is pending for it,
    and the damaged files (raising for the first one unless `skip_damaged`)."""
    folder = instrument_folder(root, exchange, symbol)
    pending = _pending(root, day, exchange, symbol)
    pieces, spans, damaged = [], [], []
    for number in finalized_numbers(folder, day):
        ticks_path, sidecar_path = finalized_paths(folder, day, number)
        found = check(ticks_path, sidecar_path, day, number)
        if found.state is State.VERIFIED:
            pieces.append(found.records)
            spans.append(found.finalized.span)
            continue
        if found.state is State.ABSENT and number in pending:
            continue  # being finalized: its raw file is still there
        if not skip_damaged:
            raise DataError(_damaged(ticks_path, found.problem or _NO_SIDECAR, day))
        damaged.append(ticks_path)
    records = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TICK_DTYPE)
    return _DayParts(records, tuple(spans), tuple(sorted(pending)), tuple(damaged))


def _pending(root: Path, day: dt.date, exchange: str, symbol: str) -> set[int]:
    """The recordings of the day whose raw file is still there and whose start sidecar
    lists the instrument (or cannot be read: then it may, and counts as pending)."""
    waiting = set()
    for number in raw_parts_of(root, day):
        start = read_start(start_path(root, day, number))
        if not isinstance(start, StartFacts) or (exchange, symbol) in start.instruments:
            waiting.add(number)
    return waiting


def _damaged(path: Path, problem: str, day: dt.date) -> str:
    return (
        f"the recorded ticks {path} are damaged ({problem}); they are left where they are: "
        f"read with skip_damaged=True to read the day's other recordings, or "
        f"repair({day.isoformat()!r}) to move them aside. Recorded ticks cannot be fetched "
        "again: keep a copy of the cache"
    )


def _not_recorded(exchange: str, symbol: str, day: dt.date, pending: bool) -> str:
    advice = (
        f"a raw file of it is waiting: finalize {day} first (a recording finalizes when it stops)"
        if pending
        else "record that day first"
    )
    return f"no ticks of {exchange}:{symbol} were recorded on {day}: {advice}"


__all__ = ["MergedTicks", "RepairReport", "read_day", "read_merged", "repair_day"]
