"""Reading recorded ticks back (private to `recorder`), and the one repair.

Reading an instrument's IST day lists its finalized files of that day (one per
recording), checks each against its sidecar (`_tick_sidecar.check`) and merges them by
(recording number, sequence number): the exact order a session received them. Readers
never write: a damaged file is reported with its path and left where it is. `repair`,
under the recorder's lock, moves a damaged file aside (`_tick_files.set_aside`, never a
deletion); the day's other recordings read as before, and the recording whose file was
moved aside can be finalized again while its raw file is still there.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
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
    ticks_of,
)
from ._tick_raw import read_raw_header
from ._tick_sidecar import State, check
from ._tick_spans import RecordedDay
from ._types import Tick


def read_day(root: Path, exchange: str, symbol: str, day: dt.date) -> RecordedDay:
    """One instrument's verified ticks of an IST day, from every recording that wrote
    it. Never writes. Raises DataError when no recording of the day was finalized for it
    and when one of its files is damaged (naming the file, left where it is), OSError
    naming the path when a file cannot be read."""
    exchange, symbol = instrument_key(exchange, symbol)
    folder = instrument_folder(root, exchange, symbol)
    pending = _pending(root, day, exchange, symbol)
    pieces, spans = [], []
    for number in finalized_numbers(folder, day):
        ticks_path, sidecar_path = finalized_paths(folder, day, number)
        found = check(ticks_path, sidecar_path, day, number)
        if found.state is State.ABSENT:
            if number in pending:
                continue  # being finalized: its raw file is still there
            found_problem = "it has no sidecar, so it cannot be verified"
            raise DataError(_damaged(ticks_path, found_problem, day))
        if found.state is State.DAMAGED:
            raise DataError(_damaged(ticks_path, found.problem, day))
        pieces.append(found.records)
        spans.append(found.finalized.span)
    if not spans:
        raise DataError(_not_recorded(exchange, symbol, day, bool(pending)))
    records = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TICK_DTYPE)
    return RecordedDay(exchange, symbol, day, records, tuple(spans), bool(pending))


def read_merged(root: Path, instruments: Sequence[tuple[str, str]], day: dt.date) -> list[Tick]:
    """The ticks of several instruments' IST day, merged in the order they were
    received: by recording number, then sequence number. Raises as `read_day`."""
    recorded = [read_day(root, exchange, symbol, day) for exchange, symbol in instruments]
    if not recorded:
        return []
    recordings = np.concatenate([one.records["recording"] for one in recorded])
    seqs = np.concatenate([one.records["seq"] for one in recorded])
    owners = np.concatenate([np.full(len(one.records), n) for n, one in enumerate(recorded)])
    rows = np.concatenate([np.arange(len(one.records)) for one in recorded])
    order = np.lexsort((seqs, recordings))
    ticks = [ticks_of(one.records, one.exchange, one.symbol) for one in recorded]
    pairs = zip(owners[order].tolist(), rows[order].tolist(), strict=True)
    return [ticks[owner][row] for owner, row in pairs]


def repair_day(root: Path, day: dt.date, notes: list[str]) -> list[Path]:
    """Under the recorder's lock: move every damaged finalized file of an IST day aside
    (its ticks and its sidecar), and a ticks file without a sidecar whose recording has
    no raw file left to finalize it. Returns where they went; each move is added to
    `notes`. Raises OSError naming the path when a file cannot be read or moved."""
    moved = []
    pending = raw_parts_of(root, day)
    for folder in instrument_folders(root):
        for number in finalized_numbers(folder, day):
            ticks_path, sidecar_path = finalized_paths(folder, day, number)
            found = check(ticks_path, sidecar_path, day, number)
            if found.state is State.VERIFIED or (found.state is State.ABSENT and number in pending):
                continue
            problem = found.problem or "it has no sidecar, so it cannot be verified"
            for path in (ticks_path, sidecar_path):
                if path.exists():
                    moved.append(set_aside(path))
            exchange, symbol = instrument_of(folder)
            notes.append(
                f"the finalized ticks of {exchange}:{symbol} of recording {number} on {day} were "
                f"damaged ({problem}); they were moved aside beside {ticks_path.name}"
            )
    return moved


def _pending(root: Path, day: dt.date, exchange: str, symbol: str) -> set[int]:
    """The recordings of the day whose raw file lists the instrument and waits."""
    waiting = set()
    for number, parts in raw_parts_of(root, day).items():
        try:
            listed = read_raw_header(parts[0]).instruments
        except DataError:
            continue  # not a raw file of ours: finalize moves it aside
        if (exchange, symbol) in listed:
            waiting.add(number)
    return waiting


def _damaged(path: Path, problem: str, day: dt.date) -> str:
    return (
        f"the recorded ticks {path} are damaged ({problem}); they are left where they are: "
        f"repair({day.isoformat()!r}) moves them aside, and the day's other recordings read "
        "as before. Recorded ticks cannot be fetched again: keep a copy of the cache"
    )


def _not_recorded(exchange: str, symbol: str, day: dt.date, pending: bool) -> str:
    advice = (
        f"a raw file of it is waiting: finalize {day} first (a recording finalizes when it stops)"
        if pending
        else "record that day first"
    )
    return f"no ticks of {exchange}:{symbol} were recorded on {day}: {advice}"


__all__ = ["read_day", "read_merged", "repair_day"]
