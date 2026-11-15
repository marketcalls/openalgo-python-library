"""Finalizing a recording (private to `recorder`): its raw file becomes one finalized
file and sidecar per instrument, written once and never rewritten, and only then is the
raw file removed.

A recording writes two small sidecars beside its raw file, each with `atomic_write`:
the start sidecar when it starts (its number, identity, day, start and instruments: what
readers learn pending from), and the stop sidecar before it finalizes (the same, and its
stop time, why it stopped and its feed outages). Without a stop sidecar of its identity
it ended by a crash, at its greatest receive time, its outages not known.

For each instrument the recording's file is checked first. One that exists, verifies
and was written by this recording (its identity and start, `Finalized.written_by`) is
kept as it is; one that verifies but was written by another recording under the same
number (a counter restored from a copy of the cache) is a conflict, reported, and the
raw file kept; one that is absent is written (the ticks, then the sidecar) and checked;
one that is damaged is reported and its raw file kept, for `repair` to move it aside and
a later finalize to write it again. The raw file (all its parts), then the stop and the
start sidecar, are removed only once every instrument's file exists, verifies and is
the recording's, so a crash at any point is redone safely. Sidecars left once their raw
file is gone (a removal that failed) are removed by the next finalize after the same
check. A raw file and its sidecars are the only files the recorder ever deletes.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.cache import atomic_write
from ..errors import DataError
from ._tick_files import (
    finalized_paths,
    finalized_records,
    instrument_folder,
    is_identity,
    raw_parts_of,
    set_aside,
    sidecar_numbers,
    start_path,
    stop_path,
)
from ._tick_raw import RawContent, increasing, read_raw
from ._tick_sidecar import State, check, read_file, sidecar_bytes
from ._tick_spans import Outage, Span, Stop, outages_from_json, outages_to_json

_SIDECAR_VERSION = 4


@dataclass(frozen=True)
class FinalizedFile:
    """One recording's finalized ticks of an instrument: the ticks file, the instrument,
    the IST day, the recording's number and its tick count."""

    path: Path
    exchange: str
    symbol: str
    day: dt.date
    recording: int
    ticks: int


@dataclass(frozen=True)
class RecordingFinalized:
    """What finalizing one recording did: its finalized files; `rejected`, records its
    raw file held that failed validation (those of frames dropped whole after a power
    loss among them); `duplicates`, copies of a batch written twice, kept once;
    `torn_bytes`, bytes after its last readable frame; and whether its raw file was
    removed (False while a file of it is damaged or in conflict, or a removal failed)."""

    recording: int
    day: dt.date
    files: tuple[FinalizedFile, ...]
    rejected: int
    duplicates: int
    torn_bytes: int
    removed: bool


@dataclass(frozen=True)
class StartFacts:
    """What a recording records: its number and identity, its IST day, its start (UTC
    nanoseconds) and its instruments."""

    recording: int
    identity: str
    day: dt.date
    start_ns: int
    instruments: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StopFacts:
    """How a recording stopped: why, when (UTC nanoseconds) and its feed outages."""

    stopped_by: Stop
    stop_ns: int
    outages: tuple[Outage, ...]


def write_start(root: Path, start: StartFacts) -> None:
    """Write a recording's start sidecar. Raises OSError when it cannot be written."""
    _write(start_path(root, start.day, start.recording), _start_document(start))


def write_stop(root: Path, start: StartFacts, stop: StopFacts) -> None:
    """Write a recording's stop sidecar. Raises OSError when it cannot be written."""
    document = _start_document(start) | {
        "stopped_by": stop.stopped_by.value,
        "stop_ns": stop.stop_ns,
        "outages": outages_to_json(stop.outages),
    }
    _write(stop_path(root, start.day, start.recording), document)


def read_start(path: Path) -> StartFacts | str | None:
    """A start (or stop) sidecar's start facts; None when there is none; why, when it
    cannot be parsed. Raises OSError naming the path when it cannot be read."""
    try:
        stored = json.loads(read_file(path).decode("utf-8"))
        return _start_facts(stored)
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError) as exc:  # UnicodeDecodeError is a ValueError
        return f"it cannot be parsed ({exc})"


def finalize_pending(root: Path, day: dt.date, notes: list[str]) -> list[RecordingFinalized]:
    """Under the recorder's lock: finalize every recording of an IST day whose raw file
    is still there, in number order, then remove the sidecars a removal left behind.
    What was found on the way is added to `notes`. Raises OSError naming the path when a
    file cannot be read, written or moved."""
    raws = raw_parts_of(root, day)
    done = []
    for number, parts in raws.items():
        result = finalize_recording(root, day, number, parts, notes)
        if result is not None:
            done.append(result)
    for number in sidecar_numbers(root, day):
        if number not in raws:
            _clear_left_sidecars(root, day, number, notes)
    return done


def finalize_recording(
    root: Path, day: dt.date, number: int, parts: list[Path], notes: list[str]
) -> RecordingFinalized | None:
    """Finalize one recording from its raw file's parts (see the module docstring).
    Returns None when no part is a raw file of it (each is moved aside and noted)."""
    contents = _contents(parts, day, number, notes)
    if not contents:
        return None
    header = contents[0].header
    start = StartFacts(number, header.identity, day, header.start_ns, header.instruments)
    joined = np.concatenate([content.records for content in contents])
    once = increasing(joined)
    records = joined[once]
    span = _span(root, start, records, notes)
    files = []
    for index, (exchange, symbol) in enumerate(header.instruments):
        mine = finalized_records(records[records["instrument"] == index])
        written = _finalize_instrument(root, start, exchange, symbol, mine, span, notes)
        if written is not None:
            files.append(written)
    sidecars = [stop_path(root, day, number), start_path(root, day, number)]
    removed = len(files) == len(header.instruments) and _remove([*parts, *sidecars], notes)
    return RecordingFinalized(
        recording=number,
        day=day,
        files=tuple(files),
        rejected=sum(content.rejected for content in contents),
        duplicates=int((~once).sum()),
        torn_bytes=sum(content.torn_bytes for content in contents),
        removed=removed,
    )


def _contents(parts: list[Path], day: dt.date, number: int, notes: list[str]) -> list[RawContent]:
    """The parts that are this recording's raw files; any other is moved aside."""
    contents: list[RawContent] = []
    for path in parts:
        try:
            content = read_raw(path)
            header = content.header
            first = contents[0].header if contents else header
            same = (header.identity, header.instruments) == (first.identity, first.instruments)
            if (header.recording, header.day) != (number, day) or not same:
                raise DataError(f"{path} does not hold recording {number} of {day} its name says")
        except DataError as exc:
            notes.append(f"{exc}; it was moved to {set_aside(path).name}")
            continue
        contents.append(content)
    return contents


def _span(root: Path, start: StartFacts, records: np.ndarray, notes: list[str]) -> Span:
    """The recording's span, from its stop sidecar, or ended by a crash without one."""
    greatest = int(records["ts_recv_ns"].max()) if len(records) else start.start_ns
    stopped = _read_stop(stop_path(root, start.day, start.recording), start, notes)
    if stopped is None:
        return Span(
            start.recording, start.start_ns, max(greatest, start.start_ns), Stop.CRASH, None
        )
    stop_ns = max(stopped.stop_ns, greatest, start.start_ns)
    return Span(start.recording, start.start_ns, stop_ns, stopped.stopped_by, stopped.outages)


def _read_stop(path: Path, start: StartFacts, notes: list[str]) -> StopFacts | None:
    """The recording's stop sidecar, or None without one; one that cannot be parsed or
    is another recording's is noted, and the recording counts as crashed. Raises OSError
    naming the path when it cannot be read."""
    try:
        stored = json.loads(read_file(path).decode("utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as exc:
        notes.append(
            f"the stop sidecar {path} cannot be parsed ({exc}): the recording counts as crashed"
        )
        return None
    try:
        if _start_facts(stored) != start:
            raise ValueError("it belongs to another recording")
        stop_ns, outages = stored["stop_ns"], outages_from_json(stored["outages"])
        if not isinstance(stop_ns, int) or isinstance(stop_ns, bool) or outages is None:
            raise ValueError("its stop or outages are malformed")
        return StopFacts(Stop(stored["stopped_by"]), stop_ns, outages)
    except (ValueError, KeyError, TypeError) as exc:
        notes.append(
            f"the stop sidecar {path} cannot be used ({exc}): the recording counts as crashed"
        )
        return None


def _finalize_instrument(
    root: Path,
    start: StartFacts,
    exchange: str,
    symbol: str,
    records: np.ndarray,
    span: Span,
    notes: list[str],
) -> FinalizedFile | None:
    """The recording's finalized file of one instrument: kept when it verifies and is
    the recording's, written when absent, reported when damaged or another's (None)."""
    folder = instrument_folder(root, exchange, symbol)
    ticks_path, sidecar_path = finalized_paths(folder, start.day, start.recording)
    found = check(ticks_path, sidecar_path, start.day, start.recording)
    if found.state is State.ABSENT:
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = sidecar_bytes(exchange, symbol, start.day, start.identity, records, span)
        atomic_write(ticks_path, lambda handle: np.save(handle, records, allow_pickle=False))
        atomic_write(sidecar_path, lambda handle: handle.write(sidecar))
        found = check(ticks_path, sidecar_path, start.day, start.recording)
    if found.state is State.DAMAGED:
        notes.append(
            f"the finalized ticks {ticks_path} of recording {start.recording} are damaged "
            f"({found.problem}); the recording's raw file is kept: repair moves the damaged "
            "file aside, then finalize writes it again"
        )
        return None
    if not found.finalized.written_by(start.identity, start.start_ns, exchange, symbol):
        notes.append(
            f"conflict: {ticks_path} holds another recording numbered {start.recording} (its "
            "identity or start differs: was the cache's counter restored from a copy?); this "
            "recording's raw file is kept as it is, unfinalized"
        )
        return None
    ticks = found.finalized.ticks
    return FinalizedFile(ticks_path, exchange, symbol, start.day, start.recording, ticks)


def _clear_left_sidecars(root: Path, day: dt.date, number: int, notes: list[str]) -> None:
    """Remove the stop and start sidecars of a recording whose raw file is gone, once
    every instrument's file of it exists, verifies and is its own."""
    sidecars = [stop_path(root, day, number), start_path(root, day, number)]
    known = [facts for facts in map(read_start, sidecars) if isinstance(facts, StartFacts)]
    if not known:
        notes.append(f"the sidecars of recording {number} of {day} cannot be parsed; they are kept")
        return
    start = known[0]
    for exchange, symbol in start.instruments:
        ticks_path, sidecar_path = finalized_paths(
            instrument_folder(root, exchange, symbol), day, number
        )
        found = check(ticks_path, sidecar_path, day, number)
        mine = found.state is State.VERIFIED and found.finalized.written_by(
            start.identity, start.start_ns, exchange, symbol
        )
        if not mine:
            notes.append(
                f"recording {number} of {day} has no raw file left, but {ticks_path} is not its "
                "verified file; its sidecars are kept"
            )
            return
    _remove(sidecars, notes)


def _remove(paths: list[Path], notes: list[str]) -> bool:
    """Remove a finalized recording's raw file parts, then its sidecars; a removal that
    fails is noted (a later finalize removes them) and stops the rest."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            notes.append(
                f"{path} could not be removed ({exc}); its ticks are finalized, and a later "
                "finalize removes it"
            )
            return False
    return True


def _start_document(start: StartFacts) -> dict[str, Any]:
    return {
        "version": _SIDECAR_VERSION,
        "recording": start.recording,
        "identity": start.identity,
        "day": start.day.isoformat(),
        "start_ns": start.start_ns,
        "instruments": [list(instrument) for instrument in start.instruments],
    }


def _start_facts(stored: Any) -> StartFacts:
    """The start facts of a sidecar's document. Raises ValueError, KeyError or TypeError
    when they are malformed."""
    if not isinstance(stored, dict) or stored["version"] != _SIDECAR_VERSION:
        raise ValueError("it is no sidecar of this version")
    number, identity, start_ns = stored["recording"], stored["identity"], stored["start_ns"]
    counts = all(isinstance(v, int) and not isinstance(v, bool) for v in (number, start_ns))
    if not (counts and is_identity(identity)):
        raise ValueError("its recording, identity or start is malformed")
    instruments = tuple((str(exchange), str(symbol)) for exchange, symbol in stored["instruments"])
    return StartFacts(number, identity, dt.date.fromisoformat(stored["day"]), start_ns, instruments)


def _write(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=1, sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda handle: handle.write(payload))


__all__ = [
    "FinalizedFile",
    "RecordingFinalized",
    "StartFacts",
    "StopFacts",
    "finalize_pending",
    "finalize_recording",
    "read_start",
    "write_start",
    "write_stop",
]
