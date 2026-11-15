"""The stored form of recorded ticks (private to `recorder`): the record layout, the raw
files a recording appends to, and where the files live in the package's cache.

Layout under the cache directory (`data.cache.resolve_cache_dir`), names made safe with
`data.cache.safe_name`:

    v2/ticks/recorder.lock                                      held while one writes
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.raw           while a recording runs
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.<id>.npy      the day, once finalized
    v2/ticks/exchange=<X>/symbol=<S>/index.json

A tick is one record of `TICK_DTYPE`, fixed size and little-endian, with its times in
UTC nanoseconds. A missing last trade time is `NO_EVENT_TIME` (0, which the live bars
read as missing too); a missing volume, bid or ask is NaN (`NO_VALUE`), never 0, which
is a value.

A raw file holds the ticks of one instrument and one IST day (the day of their receive
time): a header of `RAW_HEADER_SIZE` bytes (`RAW_MAGIC`, then the file's id in hex) and
the records, appended in the order the writer took them. A crash can leave a part of the
last record; a read cuts it off and reports how many bytes it cut.
"""

from __future__ import annotations

import datetime as dt
import math
import urllib.parse
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .._calendar_convert import SECONDS_PER_DAY, from_day
from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ..data.cache import FORMAT_VERSION, resolve_cache_dir, safe_name
from ..errors import DataError
from ._types import Tick

#: One recorded tick. `ts_recv_ns` is `Tick.ts_recv_ns` (when the writer took the tick
#: off its queue); `ts_queued_ns` when the feed's callback queued it, so the difference
#: is the time it waited; `ts_event_ns` the last trade time; prices rupees, volume units.
TICK_DTYPE = np.dtype(
    [
        ("ts_recv_ns", "<i8"),
        ("ts_queued_ns", "<i8"),
        ("ts_event_ns", "<i8"),
        ("ltp", "<f8"),
        ("volume", "<f8"),
        ("bid", "<f8"),
        ("ask", "<f8"),
    ]
)
#: A record's `ts_event_ns` when the feed gave no last trade time.
NO_EVENT_TIME = 0
#: A record's volume, bid or ask when the feed gave none.
NO_VALUE = math.nan

#: The cache's folder of recorded ticks, beside its history namespaces.
TICKS_FOLDER = "ticks"
#: The first bytes of a raw file; the digit is the version of the record layout.
RAW_MAGIC = b"OA-TICKS-RAW-v1\n"
# Hex characters of a raw file's id: enough that two files of one day never share one.
_RAW_ID_CHARS = 16
#: Bytes before the first record of a raw file: the magic, then the id.
RAW_HEADER_SIZE = len(RAW_MAGIC) + _RAW_ID_CHARS
RAW_SUFFIX = ".raw"
_NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
_EXCHANGE_PREFIX = "exchange="
_SYMBOL_PREFIX = "symbol="


def ticks_root(cache_dir: str | Path | None) -> Path:
    """The folder of recorded ticks in the cache directory (`resolve_cache_dir`)."""
    return resolve_cache_dir(cache_dir) / f"v{FORMAT_VERSION}" / TICKS_FOLDER


def instrument_folder(root: Path, exchange: str, symbol: str) -> Path:
    """The folder of one instrument's tick files."""
    return (
        root / f"{_EXCHANGE_PREFIX}{safe_name(exchange)}" / f"{_SYMBOL_PREFIX}{safe_name(symbol)}"
    )


def instrument_of(folder: Path) -> tuple[str, str]:
    """(exchange, symbol) of an instrument's folder: the inverse of `instrument_folder`
    (`safe_name` writes every other character as %XX of its UTF-8 bytes)."""
    exchange = folder.parent.name.removeprefix(_EXCHANGE_PREFIX)
    symbol = folder.name.removeprefix(_SYMBOL_PREFIX)
    return urllib.parse.unquote(exchange), urllib.parse.unquote(symbol)


def raw_path(root: Path, exchange: str, symbol: str, day: dt.date) -> Path:
    """The raw file of one instrument's ticks of an IST day."""
    return instrument_folder(root, exchange, symbol) / f"{day.isoformat()}{RAW_SUFFIX}"


def raw_files_of(root: Path, day: dt.date) -> list[Path]:
    """Every raw file of an IST day in the cache, in name order."""
    pattern = f"{_EXCHANGE_PREFIX}*/{_SYMBOL_PREFIX}*/{day.isoformat()}{RAW_SUFFIX}"
    return sorted(root.glob(pattern))


def ist_day(instant_ns: int) -> dt.date:
    """The IST date of an instant in UTC nanoseconds."""
    return from_day((instant_ns + IST_OFFSET_NANOS) // _NANOS_PER_DAY)


def tick_row(tick: Tick, queued_ns: int) -> tuple:
    """The fields of a tick's record, in `TICK_DTYPE` order."""
    return (
        tick.ts_recv_ns,
        queued_ns,
        NO_EVENT_TIME if tick.ts_event_ns is None else tick.ts_event_ns,
        tick.ltp,
        *(NO_VALUE if value is None else value for value in (tick.volume, tick.bid, tick.ask)),
    )


def records_of(rows: Iterable[tuple]) -> np.ndarray:
    """Records of `TICK_DTYPE` from rows made by `tick_row`."""
    return np.array(list(rows), dtype=TICK_DTYPE)


def ticks_of(records: np.ndarray, exchange: str, symbol: str) -> list[Tick]:
    """The ticks of records of `TICK_DTYPE`, in their order."""
    ticks = []
    for recv, _, event, ltp, volume, bid, ask in records.tolist():
        ticks.append(
            Tick(
                symbol,
                exchange,
                ltp,
                recv,
                None if event == NO_EVENT_TIME else event,
                *(None if math.isnan(value) else value for value in (volume, bid, ask)),
            )
        )
    return ticks


class RawFile:
    """A raw file open for appending, created by `create`. Only the writer that created
    it writes it."""

    def __init__(self, path: Path, handle: BinaryIO, raw_id: str) -> None:
        self.path = path
        self.raw_id = raw_id
        self._handle = handle

    @classmethod
    def create(cls, path: Path) -> RawFile:
        """Create a new raw file with its header written through to the system. Raises
        FileExistsError when the file exists, OSError when it cannot be made."""
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_id = uuid.uuid4().hex[:_RAW_ID_CHARS]
        # No `with` block: the file stays open across flushes until `close`.
        handle = open(path, "xb")  # noqa: SIM115
        try:
            handle.write(RAW_MAGIC + raw_id.encode("ascii"))
            handle.flush()
        except BaseException:
            handle.close()
            raise
        return cls(path, handle, raw_id)

    def append(self, records: np.ndarray) -> None:
        """Append records and hand them to the system, so a crash of the process does not
        lose them. Raises OSError when they cannot be written."""
        self._handle.write(np.ascontiguousarray(records, dtype=TICK_DTYPE).tobytes())
        self._handle.flush()

    def close(self) -> None:
        """Close the file; it stays for `finalize`."""
        self._handle.close()


@dataclass(frozen=True)
class RawContent:
    """What a raw file holds: its id, its whole records, and the bytes of a torn last
    record that were cut off (0 when there was none)."""

    raw_id: str
    records: np.ndarray
    torn_bytes: int


def read_raw(path: Path) -> RawContent:
    """Read a raw file. Raises DataError when its header is not a raw file's, OSError when
    it cannot be read."""
    blob = path.read_bytes()
    raw_id = blob[len(RAW_MAGIC) : RAW_HEADER_SIZE].decode("ascii", errors="replace")
    whole_header = len(blob) >= RAW_HEADER_SIZE and blob.startswith(RAW_MAGIC)
    if not (whole_header and all(character in "0123456789abcdef" for character in raw_id)):
        raise DataError(f"{path} is not a raw tick file of this version: its header differs")
    body = memoryview(blob)[RAW_HEADER_SIZE:]
    whole = len(body) - len(body) % TICK_DTYPE.itemsize
    records = np.frombuffer(body[:whole], dtype=TICK_DTYPE).copy()
    return RawContent(raw_id, records, len(body) - whole)


__all__ = [
    "NO_EVENT_TIME",
    "NO_VALUE",
    "RAW_HEADER_SIZE",
    "RAW_MAGIC",
    "RAW_SUFFIX",
    "TICKS_FOLDER",
    "TICK_DTYPE",
    "RawContent",
    "RawFile",
    "instrument_folder",
    "instrument_of",
    "ist_day",
    "raw_files_of",
    "raw_path",
    "read_raw",
    "records_of",
    "tick_row",
    "ticks_of",
    "ticks_root",
]
