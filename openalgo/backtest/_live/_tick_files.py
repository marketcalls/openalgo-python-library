"""The stored form of recorded ticks (private to `recorder`): the record layouts, where
every file lives, the recordings' numbers, and the one way a file is moved aside.

Layout under the cache directory (`data.cache.resolve_cache_dir`, which must be an
absolute folder here: recorded ticks cannot be fetched again), names made safe with
`data.cache.safe_name`:

    v2/ticks/recorder.lock                                 held while one writes
    v2/ticks/recordings.json                               the last recording number
    v2/ticks/raw/<YYYY-MM-DD>.<number>.p<part>.raw         a recording's raw file
    v2/ticks/raw/<YYYY-MM-DD>.<number>.stop.json           how it stopped
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.<number>.npy    finalized ticks
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.<number>.json   their sidecar

Every recording has a number, a counter for the whole cache taken under the recorder's
lock when it starts: recordings are ordered by it, never by the wall clock. A recording
covers one IST day and writes one raw file with all its instruments (another part only
when a torn end cannot be cut off); finalizing turns it into one `.npy` and one sidecar
per instrument, written once and never rewritten. Exchanges and symbols are upper case,
as OpenAlgo writes them, so one instrument has one folder on any disk.

A finalized tick is one record of `TICK_DTYPE`; a raw record (`RAW_DTYPE`) also carries
the index of its instrument in the raw file's header. `recording` is the recording's
number and `seq` the tick's place in that recording's receive order across all
instruments: the writer numbers each message from 1 as it takes it off the queue, so
(recording, seq) orders ticks exactly where receive times tie or the wall clock stepped
back. A missing last trade time is `NO_EVENT_TIME` (0, which the live bars read as
missing too); a missing volume, bid or ask is NaN (`NO_VALUE`). A volume of 0 is stored
as 0: the SDK fills a quote volume the server left out with 0, and the live bars apply
their own rule to it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import urllib.parse
import uuid
from pathlib import Path

import numpy as np

from .._calendar_convert import SECONDS_PER_DAY, from_day
from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ..data.cache import CACHE_ENV, FORMAT_VERSION, atomic_write, resolve_cache_dir, safe_name
from ..errors import ConfigError
from ._types import LOGGER_NAME, Tick

#: One finalized tick. `recording` is the number of the recording that took it, `seq`
#: its place in that recording's receive order; `ts_recv_ns` is `Tick.ts_recv_ns` (the
#: wall clock when the writer took it off its queue), `ts_queued_ns` when the feed's
#: callback queued it, `ts_event_ns` the last trade time; prices rupees, volume units.
TICK_DTYPE = np.dtype(
    [
        ("recording", "<u8"),
        ("seq", "<i8"),
        ("ts_recv_ns", "<i8"),
        ("ts_queued_ns", "<i8"),
        ("ts_event_ns", "<i8"),
        ("ltp", "<f8"),
        ("volume", "<f8"),
        ("bid", "<f8"),
        ("ask", "<f8"),
    ]
)
#: One raw tick: the index of its instrument in the raw file's header, then a tick.
RAW_DTYPE = np.dtype([("instrument", "<u4"), *TICK_DTYPE.descr])
#: A record's `ts_event_ns` when the feed gave no last trade time.
NO_EVENT_TIME = 0
#: A record's volume, bid or ask when the feed gave none.
NO_VALUE = math.nan

#: The cache's folder of recorded ticks, beside its history namespaces.
TICKS_FOLDER = "ticks"
RAW_FOLDER = "raw"
COUNTER_NAME = "recordings.json"
RAW_SUFFIX = ".raw"
STOP_SUFFIX = ".stop.json"
DAY_SUFFIX = ".npy"
SIDECAR_SUFFIX = ".json"
# Digits of a recording number in a file name: names of one day sort by number.
_NUMBER_DIGITS = 8
# Hex characters of the suffix that keeps a moved-aside file's name unique.
_ASIDE_CHARS = 8
_NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
_EXCHANGE_PREFIX = "exchange="
_SYMBOL_PREFIX = "symbol="

logger = logging.getLogger(LOGGER_NAME)


def ticks_root(cache_dir: str | os.PathLike | None) -> Path:
    """The folder of recorded ticks in the cache directory (`resolve_cache_dir`, a
    leading `~` expanded). Raises ConfigError when that is not an absolute folder: a
    relative one would move with the folder Python was started from, and recorded ticks
    cannot be fetched again."""
    given = resolve_cache_dir(cache_dir)
    folder = given.expanduser()
    if not folder.is_absolute():
        source = "cache_dir" if cache_dir is not None else CACHE_ENV
        raise ConfigError(
            f"recorded ticks are kept in an absolute folder only, and {source} is "
            f"{os.fspath(given)!r}: pass cache_dir= an absolute folder, or set {CACHE_ENV} "
            "to one. Recorded ticks cannot be fetched again, so they must not land in a "
            "folder that moves with the folder Python was started from"
        )
    return folder / f"v{FORMAT_VERSION}" / TICKS_FOLDER


def instrument_key(exchange: str, symbol: str) -> tuple[str, str]:
    """(exchange, symbol) in upper case, as OpenAlgo writes them."""
    return exchange.upper(), symbol.upper()


def instrument_folder(root: Path, exchange: str, symbol: str) -> Path:
    """The folder of one instrument's finalized files."""
    exchange, symbol = instrument_key(exchange, symbol)
    return (
        root / f"{_EXCHANGE_PREFIX}{safe_name(exchange)}" / f"{_SYMBOL_PREFIX}{safe_name(symbol)}"
    )


def instrument_folders(root: Path) -> list[Path]:
    """Every instrument's folder in the folder of recorded ticks, in name order."""
    found = root.glob(f"{_EXCHANGE_PREFIX}*/{_SYMBOL_PREFIX}*")
    return sorted(path for path in found if path.is_dir())


def instrument_of(folder: Path) -> tuple[str, str]:
    """(exchange, symbol) of an instrument's folder: the inverse of `instrument_folder`
    (`safe_name` writes every other character as %XX of its UTF-8 bytes)."""
    exchange = folder.parent.name.removeprefix(_EXCHANGE_PREFIX)
    symbol = folder.name.removeprefix(_SYMBOL_PREFIX)
    return urllib.parse.unquote(exchange), urllib.parse.unquote(symbol)


def number_text(number: int) -> str:
    """A recording number as it appears in file names."""
    return f"{number:0{_NUMBER_DIGITS}d}"


def raw_path(root: Path, day: dt.date, number: int, part: int) -> Path:
    """The raw file of a recording's part (0 first)."""
    return root / RAW_FOLDER / f"{day.isoformat()}.{number_text(number)}.p{part}{RAW_SUFFIX}"


def stop_path(root: Path, day: dt.date, number: int) -> Path:
    """The sidecar that says how a recording stopped, beside its raw file."""
    return root / RAW_FOLDER / f"{day.isoformat()}.{number_text(number)}{STOP_SUFFIX}"


def raw_parts_of(root: Path, day: dt.date) -> dict[int, list[Path]]:
    """The raw files of an IST day by recording number, each recording's parts in
    order."""
    parts: dict[int, list[tuple[int, Path]]] = {}
    for path in (root / RAW_FOLDER).glob(f"{day.isoformat()}.*{RAW_SUFFIX}"):
        pieces = path.name.split(".")
        if len(pieces) == 4 and pieces[1].isdecimal() and pieces[2][1:].isdecimal():
            parts.setdefault(int(pieces[1]), []).append((int(pieces[2][1:]), path))
    return {number: [path for _, path in sorted(of)] for number, of in sorted(parts.items())}


def finalized_paths(folder: Path, day: dt.date, number: int) -> tuple[Path, Path]:
    """(ticks, sidecar) of one recording's finalized ticks of an instrument."""
    stem = f"{day.isoformat()}.{number_text(number)}"
    return folder / f"{stem}{DAY_SUFFIX}", folder / f"{stem}{SIDECAR_SUFFIX}"


def finalized_numbers(folder: Path, day: dt.date) -> list[int]:
    """The recordings with a finalized file or sidecar of an instrument's IST day, in
    order."""
    numbers = set()
    for suffix in (DAY_SUFFIX, SIDECAR_SUFFIX):
        for path in folder.glob(f"{day.isoformat()}.*{suffix}"):
            pieces = path.name.split(".")
            if len(pieces) == 3 and pieces[1].isdecimal():
                numbers.add(int(pieces[1]))
    return sorted(numbers)


def next_recording_number(root: Path) -> int:
    """Take the next recording number for the cache, under the recorder's lock: one more
    than the last taken and than any number a file of the cache carries (so a lost
    counter file never reuses one). Raises OSError when the counter cannot be written."""
    counter = root / COUNTER_NAME
    try:
        last = json.loads(counter.read_bytes().decode("utf-8"))["last"]
        known = isinstance(last, int) and not isinstance(last, bool) and last >= 0
    except FileNotFoundError:
        last, known = 0, False
    except (ValueError, KeyError, TypeError):
        last, known = 0, False
    if not known and counter.exists():
        logger.warning(
            "the recording counter %s cannot be parsed: the next number is taken from the "
            "files of the cache, and the counter is written again",
            counter,
        )
    last = last if known else 0
    numbers = [last, *_numbers_in(root / RAW_FOLDER)]
    if not known:  # the counter is gone or unreadable: every finalized file is looked at
        numbers.extend(_numbers_in(root, recursive=True))
    number = max(numbers) + 1
    payload = json.dumps({"last": number}).encode("utf-8")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(counter, lambda handle: handle.write(payload))
    return number


def _numbers_in(folder: Path, *, recursive: bool = False) -> list[int]:
    paths = folder.rglob("????-??-??.*") if recursive else folder.glob("????-??-??.*")
    pieces = (path.name.split(".") for path in paths)
    return [int(piece[1]) for piece in pieces if len(piece) >= 3 and piece[1].isdecimal()]


def ist_day(instant_ns: int) -> dt.date:
    """The IST date of an instant in UTC nanoseconds."""
    return from_day((instant_ns + IST_OFFSET_NANOS) // _NANOS_PER_DAY)


def day_bounds(day: dt.date) -> tuple[int, int]:
    """[start, end) of an IST date in UTC nanoseconds."""
    midnight = (day - dt.date(1970, 1, 1)).days * _NANOS_PER_DAY - IST_OFFSET_NANOS
    return midnight, midnight + _NANOS_PER_DAY


def tick_row(tick: Tick, *, instrument: int, recording: int, seq: int, queued_ns: int) -> tuple:
    """The fields of a tick's raw record, in `RAW_DTYPE` order."""
    return (
        instrument,
        recording,
        seq,
        tick.ts_recv_ns,
        queued_ns,
        NO_EVENT_TIME if tick.ts_event_ns is None else tick.ts_event_ns,
        tick.ltp,
        *(NO_VALUE if value is None else value for value in (tick.volume, tick.bid, tick.ask)),
    )


def records_of(rows: list[tuple]) -> np.ndarray:
    """Raw records (`RAW_DTYPE`) from rows made by `tick_row`."""
    return np.array(rows, dtype=RAW_DTYPE)


def finalized_records(raw: np.ndarray) -> np.ndarray:
    """Raw records as finalized ones: the same fields without the instrument."""
    records = np.empty(len(raw), dtype=TICK_DTYPE)
    for name in TICK_DTYPE.names:
        records[name] = raw[name]
    return records


def ticks_of(records: np.ndarray, exchange: str, symbol: str) -> list[Tick]:
    """The ticks of finalized records (`TICK_DTYPE`), in their order."""
    ticks = []
    for _, _, recv, _, event, ltp, volume, bid, ask in records.tolist():
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


def set_aside(path: Path) -> Path:
    """Move a damaged file aside, under a name no reader looks at, and return where it
    went. Recorded ticks cannot be fetched again, so nothing is ever deleted. Raises
    OSError naming the path when it cannot be moved now (another process may hold it
    open)."""
    moved = path.with_name(f"{path.name}.damaged-{uuid.uuid4().hex[:_ASIDE_CHARS]}")
    try:
        os.replace(path, moved)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"{exc.strerror or exc}: the damaged file could not be moved aside now; close what "
            "holds it open and repair again",
            str(path),
        ) from exc
    return moved


__all__ = [
    "COUNTER_NAME",
    "DAY_SUFFIX",
    "NO_EVENT_TIME",
    "NO_VALUE",
    "RAW_DTYPE",
    "RAW_FOLDER",
    "RAW_SUFFIX",
    "SIDECAR_SUFFIX",
    "STOP_SUFFIX",
    "TICKS_FOLDER",
    "TICK_DTYPE",
    "day_bounds",
    "finalized_numbers",
    "finalized_paths",
    "finalized_records",
    "instrument_folder",
    "instrument_folders",
    "instrument_key",
    "instrument_of",
    "ist_day",
    "next_recording_number",
    "number_text",
    "raw_parts_of",
    "raw_path",
    "records_of",
    "set_aside",
    "stop_path",
    "tick_row",
    "ticks_of",
    "ticks_root",
]
