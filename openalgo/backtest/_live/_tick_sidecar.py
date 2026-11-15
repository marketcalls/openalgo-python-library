"""A finalized file's sidecar (private to `recorder`): what it records, and the one check
of a finalized file against it that finalize, the readers and repair all use.

Beside each `<day>.<number>.npy` (one recording's ticks of one instrument) a
`<day>.<number>.json` records the instrument, the day, the recording's number, the
file's SHA-256 content hash, the tick count, the first and last receive time and the
recording's span (its start and stop, how it stopped, its outages). The sidecar is
written after the ticks, so it is the mark that a finalized file is whole. Only two
things are damage: a hash that does not match, and a file that cannot be parsed (a
ticks file without a sidecar is not finalized yet). An OSError other than a missing
file is raised naming the path.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._tick_files import TICK_DTYPE
from ._tick_spans import Span, outages_to_json, span_from_json

#: The version of a finalized file's sidecar.
SIDECAR_VERSION = 3
_KIND = "recorded_ticks"
# Mixed into every hash, so that a change of the stored form changes the hashes.
_HASH_HEADER = b"openalgo.backtest recorded ticks v3\n"
_HASH_PREFIX = "sha256:"


@dataclass(frozen=True)
class Finalized:
    """What a sidecar records of one recording's finalized ticks of an instrument."""

    exchange: str
    symbol: str
    day: dt.date
    recording: int
    sha256: str
    ticks: int
    span: Span


class State(enum.Enum):
    """Where a recording's finalized file of an instrument stands."""

    VERIFIED = "verified"  # the ticks and the sidecar are there and agree
    ABSENT = "absent"  # no sidecar: not finalized (the ticks file may be half way)
    DAMAGED = "damaged"  # the sidecar or the ticks cannot be parsed, or disagree


@dataclass(frozen=True)
class Check:
    """The outcome of checking a finalized file: its state; the sidecar and the ticks
    when verified; why, when damaged."""

    state: State
    finalized: Finalized | None = None
    records: np.ndarray | None = None
    problem: str = ""


def tick_hash(records: np.ndarray) -> str:
    """SHA-256 of finalized ticks, as 'sha256:<hex>'."""
    digest = hashlib.sha256(_HASH_HEADER)
    digest.update(np.ascontiguousarray(records, dtype=TICK_DTYPE).tobytes())
    return _HASH_PREFIX + digest.hexdigest()


def sidecar_bytes(
    exchange: str, symbol: str, day: dt.date, records: np.ndarray, span: Span
) -> bytes:
    """The sidecar of finalized `records` (in receive order) of one recording."""
    receipts = records["ts_recv_ns"]
    document = {
        "version": SIDECAR_VERSION,
        "kind": _KIND,
        "exchange": exchange,
        "symbol": symbol,
        "day": day.isoformat(),
        "recording": span.recording,
        "sha256": tick_hash(records),
        "ticks": len(records),
        "first_recv_ns": int(receipts[0]) if len(records) else None,
        "last_recv_ns": int(receipts[-1]) if len(records) else None,
        "start_ns": span.start_ns,
        "stop_ns": span.stop_ns,
        "stopped_by": span.ended_by.value,
        "outages": outages_to_json(span.outages),
    }
    return json.dumps(document, indent=1, sort_keys=True).encode("utf-8")


def read_file(path: Path) -> bytes:
    """The bytes of a file. Raises OSError naming the path when it cannot be read."""
    try:
        return path.read_bytes()
    except OSError as exc:
        if exc.filename is not None:
            raise
        raise OSError(exc.errno, exc.strerror or str(exc), str(path)) from exc


def check(ticks_path: Path, sidecar_path: Path, day: dt.date, number: int) -> Check:
    """Check one recording's finalized file of an instrument against its sidecar.
    Raises OSError naming the path when a file cannot be read (other than missing)."""
    try:
        blob = read_file(sidecar_path)
    except FileNotFoundError:
        return Check(State.ABSENT)
    finalized = _parsed(blob, day, number)
    if isinstance(finalized, str):
        return Check(State.DAMAGED, problem=f"its sidecar {sidecar_path.name} {finalized}")
    try:
        records = _ticks(read_file(ticks_path))
    except FileNotFoundError:
        return Check(State.DAMAGED, finalized, problem="its ticks file is missing")
    if isinstance(records, str):
        return Check(State.DAMAGED, finalized, problem=f"its ticks file {records}")
    if tick_hash(records) != finalized.sha256:
        problem = "its ticks do not match the hash in its sidecar"
        return Check(State.DAMAGED, finalized, problem=problem)
    return Check(State.VERIFIED, finalized, records)


def _ticks(blob: bytes) -> np.ndarray | str:
    try:
        records = np.load(io.BytesIO(blob), allow_pickle=False)
    except (ValueError, EOFError) as exc:
        return f"cannot be parsed ({exc})"
    if not (isinstance(records, np.ndarray) and records.dtype == TICK_DTYPE and records.ndim == 1):
        return "holds no recorded ticks of this version"
    return records


def _parsed(blob: bytes, day: dt.date, number: int) -> Finalized | str:
    """The sidecar, or why it cannot be parsed."""
    try:
        stored = json.loads(blob.decode("utf-8"))
        if not isinstance(stored, dict):
            raise TypeError("it is not a JSON object")
        same = stored["version"] == SIDECAR_VERSION and stored["kind"] == _KIND
        if not (same and stored["day"] == day.isoformat() and stored["recording"] == number):
            raise ValueError("it belongs to another day, recording or format")
        sha, ticks = stored["sha256"], stored["ticks"]
        counted = isinstance(ticks, int) and not isinstance(ticks, bool) and ticks >= 0
        if not (isinstance(sha, str) and sha.startswith(_HASH_PREFIX) and counted):
            raise ValueError("its hash or tick count is malformed")
        return Finalized(
            exchange=_text(stored["exchange"]),
            symbol=_text(stored["symbol"]),
            day=day,
            recording=number,
            sha256=sha,
            ticks=ticks,
            span=span_from_json(stored, number),
        )
    except (ValueError, KeyError, TypeError) as exc:  # UnicodeDecodeError is a ValueError
        return f"cannot be parsed ({exc})"


def _text(value: Any) -> str:
    if not (isinstance(value, str) and value):
        raise ValueError(f"{value!r} is no name")
    return value


__all__ = [
    "SIDECAR_VERSION",
    "Check",
    "Finalized",
    "State",
    "check",
    "read_file",
    "sidecar_bytes",
    "tick_hash",
]
