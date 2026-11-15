"""The index of an instrument's recorded ticks (private to `recorder`): what it holds,
how a reader reads it, and how the recorder repairs it under its lock.

`index.json` records per IST day: the day file, its SHA-256 content hash, the tick
count, the first and last receive time, and the spans of the recordings that wrote the
day (`_tick_spans`). Only two things are corruption: a hash that does not match, and a
file that cannot be parsed. An OSError (a file another process holds open, a disk that
is gone) is raised naming the path, and nothing is moved. A reader never writes; a
repair (moving a damaged file aside, rebuilding an index) happens only under the
recorder's lock, after the index was read again and the file checked again. Recorded
ticks cannot be fetched again, so nothing is ever deleted for being damaged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.cache import INDEX_NAME, _quarantine, atomic_write
from ..errors import DataError
from ._tick_files import DAY_SUFFIX, TICK_DTYPE, recording_name
from ._tick_spans import Ending, Span, span_from_json, span_to_json
from ._types import LOGGER_NAME

#: The version of the index of an instrument's recorded ticks.
TICK_INDEX_VERSION = 2
_INDEX_KIND = "ticks"
# Mixed into every hash, so that a change of the stored form changes the hashes.
_HASH_HEADER = b"openalgo.backtest recorded ticks v2\n"
_HASH_PREFIX = "sha256:"

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class DayEntry:
    """What the index holds of one day: its file (a name in the instrument's folder),
    the hash of its records, the tick count, the first and last receive time (UTC
    nanoseconds, None for a day without a tick) and the recordings' spans."""

    file: str
    sha256: str
    ticks: int
    first_recv_ns: int | None
    last_recv_ns: int | None
    spans: tuple[Span, ...]


#: An instrument's index: its days by ISO date.
Days = dict[str, DayEntry]


def tick_hash(records: np.ndarray) -> str:
    """SHA-256 of recorded ticks, as 'sha256:<hex>'."""
    digest = hashlib.sha256(_HASH_HEADER)
    digest.update(np.ascontiguousarray(records, dtype=TICK_DTYPE).tobytes())
    return _HASH_PREFIX + digest.hexdigest()


def day_entry(file_name: str, records: np.ndarray, spans: tuple[Span, ...]) -> DayEntry:
    """The index entry of a day file holding `records` (in receive order)."""
    receipts = records["ts_recv_ns"]
    first, last = (int(receipts[0]), int(receipts[-1])) if len(records) else (None, None)
    return DayEntry(file_name, tick_hash(records), len(records), first, last, spans)


def read_file(path: Path) -> bytes:
    """The bytes of a file. Raises OSError naming the path when it cannot be read."""
    try:
        return path.read_bytes()
    except OSError as exc:
        if exc.filename is not None:
            raise
        raise OSError(exc.errno, exc.strerror or str(exc), str(path)) from exc


def load_records(path: Path) -> np.ndarray | str:
    """The records of a day file, or why it cannot be parsed (corruption). Raises
    OSError naming the path when it cannot be read."""
    blob = read_file(path)
    try:
        records = np.load(io.BytesIO(blob), allow_pickle=False)
    except (ValueError, EOFError) as exc:
        return f"it cannot be parsed ({exc})"
    if not (isinstance(records, np.ndarray) and records.dtype == TICK_DTYPE and records.ndim == 1):
        return "it holds no recorded ticks of this version"
    return records


def check_day(folder: Path, entry: DayEntry) -> np.ndarray | str:
    """The verified records of a day, or why its file is corrupt. Raises OSError naming
    the path when it cannot be read."""
    loaded = load_records(folder / entry.file)
    if isinstance(loaded, str):
        return loaded
    if tick_hash(loaded) != entry.sha256:
        return "its content does not match the hash in the index"
    return loaded


def read_index(folder: Path, exchange: str, symbol: str) -> Days:
    """An instrument's index as stored, for a reader; empty when there is none. Raises
    DataError when it cannot be parsed (nothing is changed), OSError naming the path
    when it cannot be read."""
    path = folder / INDEX_NAME
    try:
        blob = read_file(path)
    except FileNotFoundError:
        return {}
    days, problem = _parsed(blob, exchange, symbol)
    if problem is not None:
        raise DataError(
            f"the tick index {path} cannot be parsed ({problem}); nothing was changed: a "
            "finalize of the day moves it aside and rebuilds it from the day files"
        )
    return days


def load_index_locked(folder: Path, exchange: str, symbol: str, notes: list[str]) -> Days:
    """An instrument's index, under the recorder's lock. One that cannot be parsed is
    moved aside, and rebuilt from the day files present (as is a missing one beside day
    files), with a warning added to `notes`. Raises OSError naming the path when a file
    cannot be read, written or moved."""
    path = folder / INDEX_NAME
    try:
        blob = read_file(path)
    except FileNotFoundError:
        blob = None
    if blob is not None:
        days, problem = _parsed(blob, exchange, symbol)
        if problem is None:
            return days
        why = f"it could not be parsed ({problem}) and was moved to {_quarantine(path).name}"
    elif any(folder.glob(f"*{DAY_SUFFIX}")):
        why = "it was missing"
    else:
        return {}
    days = _rebuilt(folder, notes)
    save_index(folder, exchange, symbol, days)
    note = (
        f"the tick index {path} was rebuilt from the day files present: {why}; the spans and "
        "outages of its days are not known"
    )
    logger.warning(note)
    notes.append(note)
    return days


def save_index(folder: Path, exchange: str, symbol: str, days: Days) -> None:
    """Write an instrument's index. Raises OSError when it cannot be written."""
    stored = {
        "version": TICK_INDEX_VERSION,
        "kind": _INDEX_KIND,
        "exchange": exchange,
        "symbol": symbol,
        "days": {key: _entry_to_json(entry) for key, entry in sorted(days.items())},
    }
    payload = json.dumps(stored, indent=1, sort_keys=True).encode("utf-8")
    atomic_write(folder / INDEX_NAME, lambda handle: handle.write(payload))


def _parsed(blob: bytes, exchange: str, symbol: str) -> tuple[Days, str | None]:
    """(the days, None), or ({}, why the index cannot be parsed)."""
    try:
        stored = json.loads(blob.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError is a ValueError too
        return {}, f"it is not JSON ({exc})"
    header = {"version": TICK_INDEX_VERSION, "kind": _INDEX_KIND, "exchange": exchange}
    header["symbol"] = symbol
    if not isinstance(stored, dict) or any(stored.get(k) != v for k, v in header.items()):
        return {}, "it belongs to another instrument or format"
    if not isinstance(stored.get("days"), dict):
        return {}, "it holds no days"
    try:
        days = {key: _entry_from_json(key, entry) for key, entry in stored["days"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        return {}, f"an entry is malformed ({exc})"
    return days, None


def _entry_from_json(key: str, stored: Any) -> DayEntry:
    """A day's entry. Raises KeyError, TypeError or ValueError when it is malformed."""
    dt.date.fromisoformat(key)
    file, sha, ticks = stored["file"], stored["sha256"], stored["ticks"]
    # Only a plain day file name of this folder, so an edited index cannot point a read
    # (or a move aside) anywhere else.
    plain = isinstance(file, str) and Path(file).name == file
    if not (plain and file.startswith(f"{key}.") and file.endswith(DAY_SUFFIX)):
        raise ValueError(f"the file of {key} is not a day file of this folder")
    counted = isinstance(ticks, int) and not isinstance(ticks, bool) and ticks >= 0
    if not (isinstance(sha, str) and sha.startswith(_HASH_PREFIX) and counted):
        raise ValueError(f"the hash or count of {key} is malformed")
    spans = stored["spans"]
    if not isinstance(spans, list):
        raise TypeError(f"the spans of {key} are not a list")
    first, last = stored["first_recv_ns"], stored["last_recv_ns"]
    return DayEntry(file, sha, ticks, first, last, tuple(map(span_from_json, spans)))


def _entry_to_json(entry: DayEntry) -> dict[str, Any]:
    return {
        "file": entry.file,
        "sha256": entry.sha256,
        "ticks": entry.ticks,
        "first_recv_ns": entry.first_recv_ns,
        "last_recv_ns": entry.last_recv_ns,
        "spans": [span_to_json(span) for span in entry.spans],
    }


def _rebuilt(folder: Path, notes: list[str]) -> Days:
    """The days of an instrument's folder, from its day files: per day the most recently
    written file that can be parsed, hashed anew, its spans unknown. A file that cannot
    be parsed is moved aside and noted."""
    by_day: dict[str, list[Path]] = {}
    for path in folder.glob(f"*{DAY_SUFFIX}"):
        key = path.name.split(".", 1)[0]
        if _is_date(key):
            by_day.setdefault(key, []).append(path)
    days = {}
    for key, paths in sorted(by_day.items()):
        for path in sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True):
            loaded = load_records(path)
            if isinstance(loaded, str):
                notes.append(
                    f"the day file {path} {loaded}; it was moved to {_quarantine(path).name}"
                )
                continue
            days[key] = day_entry(path.name, loaded, _unknown_spans(loaded))
            break
    return days


def _unknown_spans(records: np.ndarray) -> tuple[Span, ...]:
    """A span of unknown times for each recording of the records, in first-seen order."""
    numbers, first = np.unique(records["recording"], return_index=True)
    in_order = numbers[np.argsort(first)].tolist()
    return tuple(Span(recording_name(n), None, None, Ending.UNKNOWN, None) for n in in_order)


def _is_date(text: str) -> bool:
    try:
        dt.date.fromisoformat(text)
    except ValueError:
        return False
    return True


__all__ = [
    "TICK_INDEX_VERSION",
    "DayEntry",
    "Days",
    "check_day",
    "day_entry",
    "load_index_locked",
    "load_records",
    "read_file",
    "read_index",
    "save_index",
    "tick_hash",
]
