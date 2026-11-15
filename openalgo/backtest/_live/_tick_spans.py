"""What a recorded day says about its own coverage (private to `recorder`, which
re-exports the types): the span of each recording that wrote it, how it stopped, the
feed outages inside it, and a day read back with them.

A moment of a day is in one of three states: recorded (inside a recording's span, the
feed up), a feed outage (inside a span, the feed down), or not recorded (outside every
span: no recording ran). Inside the span of a recording that crashed the state is not
known: its outages were lost with it.
"""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class Stop(enum.Enum):
    """Why a recording stopped. A summary never says CRASH: only a span does, for a
    recording that did not write how it stopped (killed, crashed, the power gone)."""

    CLOSE = "close"  # the day's close from the calendar, the default stop
    STOP_AT = "stop_at"  # the `stop_at_ns` the call gave
    INTERRUPT = "interrupt"  # Ctrl+C (KeyboardInterrupt)
    WRITE_FAILED = "write_failed"  # a write of recorded ticks failed
    CRASH = "crash"  # it ended without saying how: at its greatest receive time


@dataclass(frozen=True, order=True)
class Outage:
    """A span in which the feed was down, UTC nanoseconds: from the feed's disconnect to
    its reconnect, or to the end of the recording when it was still down then."""

    start_ns: int
    end_ns: int

    def holds(self, instant_ns: int) -> bool:
        """Whether the feed was down at `instant_ns`."""
        return self.start_ns <= instant_ns < self.end_ns


class Coverage(enum.Enum):
    """What a recorded day says about a moment (see the module docstring)."""

    RECORDED = "recorded"
    OUTAGE = "outage"
    NOT_RECORDED = "not_recorded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Span:
    """The time one recording covered on its day, UTC nanoseconds, both ends included:
    from its start to its stop, or to its greatest receive time when that is later (a
    tick taken while it stopped). A recording that crashed (`ended_by` CRASH) stops at
    its greatest receive time (its start when it took none), and its `outages` are None:
    not known."""

    recording: int
    start_ns: int
    stop_ns: int
    ended_by: Stop
    outages: tuple[Outage, ...] | None

    def contains(self, instant_ns: int) -> bool:
        """Whether the span covers `instant_ns`."""
        return self.start_ns <= instant_ns <= self.stop_ns


@dataclass(frozen=True)
class RecordedDay:
    """One instrument's recorded ticks of an IST day.

    `records` are of `TICK_DTYPE`, in receive order (by recording number, each in its
    sequence); `spans` the recordings that wrote them, in the same order; `pending` is
    True when a raw file of the day that lists the instrument waits to be finalized (a
    recording runs, or one crashed): its ticks are not in `records` yet.
    """

    exchange: str
    symbol: str
    day: dt.date
    records: np.ndarray
    spans: tuple[Span, ...]
    pending: bool

    @property
    def outages(self) -> tuple[Outage, ...] | None:
        """Every feed outage of the day in start order, None when a span's are unknown."""
        if any(span.outages is None for span in self.spans):
            return None
        return tuple(sorted(outage for span in self.spans for outage in span.outages))

    def coverage(self, instant_ns: int) -> Coverage:
        """What the day says about the moment `instant_ns` (UTC nanoseconds)."""
        for span in self.spans:
            if span.contains(instant_ns):
                if span.outages is None:
                    return Coverage.UNKNOWN
                down = any(outage.holds(instant_ns) for outage in span.outages)
                return Coverage.OUTAGE if down else Coverage.RECORDED
        return Coverage.NOT_RECORDED


def outages_to_json(outages: tuple[Outage, ...] | None) -> list[list[int]] | None:
    """Outages as a sidecar stores them."""
    return None if outages is None else [[outage.start_ns, outage.end_ns] for outage in outages]


def outages_from_json(stored: Any) -> tuple[Outage, ...] | None:
    """Outages from a sidecar. Raises ValueError when they are malformed."""
    if stored is None:
        return None
    if not isinstance(stored, list):
        raise ValueError("the outages are not a list")
    return tuple(Outage(*_instants(pair)) for pair in stored)


def span_from_json(stored: Mapping[str, Any], recording: int) -> Span:
    """The span a finalized file's sidecar records. Raises ValueError, TypeError or
    KeyError when it is malformed."""
    start, stop = _instants([stored["start_ns"], stored["stop_ns"]])
    return Span(
        recording, start, stop, Stop(stored["stopped_by"]), outages_from_json(stored["outages"])
    )


def _instants(pair: Any) -> tuple[int, int]:
    whole = isinstance(pair, list) and len(pair) == 2
    if not (whole and all(isinstance(v, int) and not isinstance(v, bool) for v in pair)):
        raise ValueError(f"{pair!r} are no two instants")
    return pair[0], pair[1]


__all__ = [
    "Coverage",
    "Outage",
    "RecordedDay",
    "Span",
    "Stop",
    "outages_from_json",
    "outages_to_json",
    "span_from_json",
]
