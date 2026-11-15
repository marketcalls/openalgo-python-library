"""The finished day files of recorded ticks (private to `recorder`): finalize turns raw
files into them, and reads verify them.

Each instrument's folder (see `_tick_files`) holds one `.npy` per IST day, a structured
array of `TICK_DTYPE` without pickle, in receive order, and an `index.json` that records
per day: the file, its SHA-256 content hash, the tick count, the first and last receive
time, the feed outages of the day (None when they are not known: a day finalized from
raw files a crashed recording left), and the ids of the raw files merged into it.

Writes follow the cache's rules (`data.cache.atomic_write`): a file is written under a
temporary name and moved into place. A day file is written under a new name, then the
index, then the old day file and the raw file are removed; so a crash at any point leaves
either the old day and its raw file, or the new day, and a raw file whose id the index
lists as merged is only removed, never merged twice. A file that fails verification is
moved aside (`data.cache._quarantine`), never served and never deleted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.cache import _READ_ERRORS, INDEX_NAME, _quarantine, atomic_write
from ..errors import DataError
from ._tick_files import (
    RAW_SUFFIX,
    TICK_DTYPE,
    instrument_folder,
    instrument_of,
    raw_files_of,
    raw_path,
    read_raw,
)

#: The version of the index of an instrument's recorded ticks.
TICK_INDEX_VERSION = 1
_INDEX_KIND = "ticks"
# Mixed into every hash, so that a change of the stored form changes the hashes.
_HASH_HEADER = b"openalgo.backtest recorded ticks v1\n"
_DAY_SUFFIX = ".npy"


@dataclass(frozen=True)
class Outage:
    """A span in which the feed was down, UTC nanoseconds: from the feed's disconnect to
    its reconnect, or to the end of the recording when it was still down then."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class RecordedDay:
    """One instrument's recorded ticks of an IST day: `records` of `TICK_DTYPE` in receive
    order, and the feed outages of the day (None when they are not known)."""

    exchange: str
    symbol: str
    day: dt.date
    records: np.ndarray
    outages: tuple[Outage, ...] | None


@dataclass(frozen=True)
class DayWritten:
    """One day file that finalize wrote: its path, its instrument and its tick count
    (the whole day's), and the bytes of a torn last record cut off its raw file."""

    path: Path
    exchange: str
    symbol: str
    ticks: int
    torn_bytes: int


def tick_hash(records: np.ndarray) -> str:
    """SHA-256 of recorded ticks, as 'sha256:<hex>'."""
    digest = hashlib.sha256(_HASH_HEADER)
    digest.update(np.ascontiguousarray(records, dtype=TICK_DTYPE).tobytes())
    return "sha256:" + digest.hexdigest()


def finalize_day(
    root: Path, day: dt.date, outages: tuple[Outage, ...] | None, notes: list[str]
) -> list[DayWritten]:
    """Turn every raw file of an IST day in `root` into its instrument's day file, with
    `outages` as the day's feed outages (None: not known). What failed verification on
    the way is added to `notes`. Raises OSError when a file cannot be written."""
    written = []
    for path in raw_files_of(root, day):
        result = finalize_raw(path, day, outages, notes)
        if result is not None:
            written.append(result)
    return written


def finalize_raw(
    path: Path, day: dt.date, outages: tuple[Outage, ...] | None, notes: list[str]
) -> DayWritten | None:
    """Merge one raw file into its instrument's day file (see the module docstring).
    Returns None when the raw file was already merged or was not a raw file (it is then
    moved aside and noted). Raises OSError when a file cannot be written."""
    try:
        raw = read_raw(path)
    except DataError as exc:
        moved = _quarantine(path)
        notes.append(f"{exc}; it was moved to {moved.name}")
        return None
    folder = path.parent
    exchange, symbol = instrument_of(folder)
    index = load_index(folder, exchange, symbol, notes)
    key = day.isoformat()
    entry = index["days"].get(key)
    if isinstance(entry, dict) and raw.raw_id in entry.get("merged", ()):
        path.unlink()
        return None
    earlier = load_day(folder, index, key, notes)
    if entry is not None and earlier is None:
        # The day's earlier ticks were moved aside: the day is incomplete, and the
        # outages of the part that remains are not known either.
        outages, entry = None, None
    records = raw.records if earlier is None else np.concatenate([earlier, raw.records])
    file_name = f"{key}.{raw.raw_id}{_DAY_SUFFIX}"
    atomic_write(folder / file_name, lambda handle: np.save(handle, records, allow_pickle=False))
    index["days"][key] = _day_entry(file_name, records, _joined(entry, outages), entry, raw.raw_id)
    save_index(folder, index)
    if entry is not None and entry["file"] != file_name:
        (folder / entry["file"]).unlink(missing_ok=True)
    path.unlink()
    return DayWritten(folder / file_name, exchange, symbol, len(records), raw.torn_bytes)


def read_day(root: Path, exchange: str, symbol: str, day: dt.date) -> RecordedDay:
    """One instrument's verified ticks of an IST day. Raises DataError when the day was
    not recorded (or not finalized yet) and when its file or index fails verification
    (the file is then moved aside)."""
    folder = instrument_folder(root, exchange, symbol)
    notes: list[str] = []
    index = load_index(folder, exchange, symbol, notes)
    key = day.isoformat()
    entry = index["days"].get(key)
    if entry is None:
        raise DataError(_not_recorded(root, exchange, symbol, day, notes))
    records = load_day(folder, index, key, notes)
    if records is None:
        raise DataError("; ".join(notes))
    outages = entry.get("outages")
    known = None if outages is None else tuple(Outage(*span) for span in outages)
    return RecordedDay(exchange, symbol, day, records, known)


def load_index(folder: Path, exchange: str, symbol: str, notes: list[str]) -> dict:
    """The index of an instrument's folder; a fresh one when there is none, and when it
    is unusable (it is then moved aside and noted)."""
    fresh = {
        "version": TICK_INDEX_VERSION,
        "kind": _INDEX_KIND,
        "exchange": exchange,
        "symbol": symbol,
        "days": {},
    }
    path = folder / INDEX_NAME
    try:
        index = json.loads(path.read_bytes().decode("utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return fresh
    except (OSError, ValueError) as exc:
        problem = f"it cannot be read ({exc})"
    else:
        same = isinstance(index, dict) and all(
            index.get(name) == value for name, value in fresh.items() if name != "days"
        )
        if same and isinstance(index.get("days"), dict):
            return index
        problem = "it belongs to another instrument or format"
    moved = _quarantine(path)
    notes.append(
        f"the tick index {path} failed verification ({problem}); it was moved to "
        f"{moved.name}, and the day files it described are kept but can no longer be verified"
    )
    return fresh


def save_index(folder: Path, index: dict) -> None:
    """Write an instrument's index. Raises OSError when it cannot be written."""
    payload = json.dumps(index, indent=1, sort_keys=True).encode("utf-8")
    atomic_write(folder / INDEX_NAME, lambda handle: handle.write(payload))


def load_day(folder: Path, index: dict, key: str, notes: list[str]) -> np.ndarray | None:
    """The verified records of the day `key` (an ISO date), or None when the index has
    no such day, or when its file fails verification: the file is then moved aside, the
    day left out of the index and the failure added to `notes`."""
    entry = index["days"].get(key)
    if entry is None:
        return None
    name = entry.get("file") if isinstance(entry, dict) else None
    # Only a plain file name of this folder is read, so an edited index cannot point
    # the read (or the move aside) anywhere else.
    if not (isinstance(name, str) and Path(name).name == name and name.endswith(_DAY_SUFFIX)):
        return _failed(folder, index, key, None, "its index entry is malformed", notes)
    path = folder / name
    try:
        records = np.load(path, allow_pickle=False)
        matches = records.dtype == TICK_DTYPE and tick_hash(records) == entry.get("sha256")
        problem = None if matches else "its content does not match the hash in the index"
    except _READ_ERRORS as exc:
        problem = f"it cannot be read ({exc})"
    if problem is None:
        return records
    return _failed(folder, index, key, path, problem, notes)


def _failed(
    folder: Path, index: dict, key: str, path: Path | None, problem: str, notes: list[str]
) -> None:
    """Move a day file that failed verification aside (`path`, None when the index names
    no file of this folder), drop its day from the index and note it."""
    moved = None if path is None else _quarantine(path)
    del index["days"][key]
    save_index(folder, index)
    where = f"; it was moved to {moved.name}" if moved is not None else ""
    notes.append(
        f"the recorded ticks of {key} in {folder} failed verification ({problem}){where}, "
        "and that day is no longer read"
    )


def _day_entry(
    file_name: str,
    records: np.ndarray,
    outages: list[list[int]] | None,
    earlier: dict | None,
    raw_id: str,
) -> dict[str, Any]:
    receipts = records["ts_recv_ns"]
    return {
        "file": file_name,
        "sha256": tick_hash(records),
        "ticks": len(records),
        "first_recv_ns": int(receipts[0]) if len(records) else None,
        "last_recv_ns": int(receipts[-1]) if len(records) else None,
        "outages": outages,
        "merged": [*(earlier or {}).get("merged", []), raw_id],
    }


def _joined(earlier: dict | None, outages: tuple[Outage, ...] | None) -> list[list[int]] | None:
    """The day's outages as stored: the earlier entry's and these, in start order; None
    when either is not known."""
    before = [] if earlier is None else earlier.get("outages")
    if before is None or outages is None:
        return None
    spans = [*map(tuple, before), *((o.start_ns, o.end_ns) for o in outages)]
    return [list(span) for span in sorted(spans)]


def _not_recorded(root: Path, exchange: str, symbol: str, day: dt.date, notes: list[str]) -> str:
    waiting = raw_path(root, exchange, symbol, day).exists()
    advice = (
        f"its raw file is waiting: finalize {day} first (a recording finalizes when it stops)"
        if waiting
        else "record that day first"
    )
    found = "; ".join([*notes, f"no ticks of {exchange}:{symbol} were recorded on {day}"])
    return f"{found}: {advice}"


__all__ = [
    "RAW_SUFFIX",
    "TICK_INDEX_VERSION",
    "DayWritten",
    "Outage",
    "RecordedDay",
    "finalize_day",
    "finalize_raw",
    "load_day",
    "load_index",
    "read_day",
    "save_index",
    "tick_hash",
]
