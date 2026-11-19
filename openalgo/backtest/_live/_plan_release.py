"""Which of today's slices a read of the broker's candles releases to the kernel in the
catch-up (16-session-design 2.5 B and C, 3.1 step 11; 08-live-trading section 5 rules
2 and 3; unit 7). Part of `plan`.

A slice is settled at a read begun `CANDLE_SETTLE_S` or more after its close. The read
then releases the settled slices not caught up yet, oldest first, each one way:

- **with its candle**, when the read holds the slice's candle;
- **as a minute without a trade** (no bar), when the read holds no candle of the slice
  but one of a later slice: the broker has published past the slice;
- **marked missing**, when the read holds neither and began `CANDLE_LAST_READ_S` or more
  after the slice's close (the last read of the candle checks): a decision on a bar
  holding the slice is skipped until its candle is read (2.5 C);
- otherwise not at all: the broker's latest candle often lags its close by more than
  the settling time, so the slice and every later one wait for a later read.

A read that holds no candle of today at all (the broker serves none, or none yet) thus
marks every slice it gives up on missing, and `missing_note` says so for the start
banner. Times are UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .._calendar_convert import format_ist
from .._layout import NANOS_PER_SECOND
from .._messages import counted
from ._types import LiveBar
from .data_worker import CANDLE_READ_OFFSETS_S

#: Seconds after a slice's close from which a read of the broker's candles may hold the
#: slice's candle: the first read of the candle checks (08-live-trading section 5 rule 2).
CANDLE_SETTLE_S = CANDLE_READ_OFFSETS_S[0]
#: Seconds after a slice's close from which a read with no candle of the slice, and none
#: later, marks the slice missing: the candle checks' last read of a slice's close.
CANDLE_LAST_READ_S = CANDLE_READ_OFFSETS_S[-1]
#: How a slice is released (module docstring).
WITH_CANDLE = "candle"
NO_TRADE = "no_trade"
MISSING = "missing"

SETTLE_NS = round(CANDLE_SETTLE_S * NANOS_PER_SECOND)
_LAST_READ_NS = round(CANDLE_LAST_READ_S * NANOS_PER_SECOND)


@dataclass(frozen=True)
class Release:
    """One slice a read releases: its open (UTC nanoseconds), its `bar` (the broker's
    candle, None without one) and `kind`, `WITH_CANDLE`, `NO_TRADE` or `MISSING`."""

    opened: int
    bar: LiveBar | None
    kind: str


def releases(
    settled: Sequence[tuple[int, int]], candles: Mapping[int, LiveBar], read_ns: int
) -> tuple[Release, ...]:
    """The slices a read of the broker's candles begun at `read_ns` releases (module
    docstring), oldest first: of `settled`, the (open, close) of the slices settled at
    the read and not caught up yet, oldest first, those up to the first the read holds
    back. `candles` are the read's candles of the instrument by open, of any slice."""
    latest = max(candles, default=None)
    released = []
    for opened, closed in settled:
        bar = candles.get(opened)
        if bar is not None:
            kind = WITH_CANDLE
        elif latest is not None and latest > opened:
            kind = NO_TRADE
        elif read_ns >= closed + _LAST_READ_NS:
            kind = MISSING
        else:
            break
        released.append(Release(opened, bar, kind))
    return tuple(released)


def missing_note(
    name: str, symbol: str, exchange: str, marked: Sequence[int], *, none_today: bool
) -> str:
    """The warning for the slices opening at `marked` (UTC nanoseconds, oldest first, at
    least one) that a read of strategy `name`'s candles marked missing; `none_today`
    when the read held no candle of today at all."""
    span = format_ist(marked[0])
    if len(marked) > 1:
        span += f" to {format_ist(marked[-1])}"
    if none_today:
        reason = (
            "OpenAlgo returned no 1-minute candle of today (check the broker's login in "
            "OpenAlgo and that the broker serves today's intraday history)"
        )
    else:
        reason = f"no broker candle of them was read by {CANDLE_LAST_READ_S:g} s after their close"
    return (
        f"{name}: {counted(len(marked), 'minute')} of {exchange}:{symbol} ({span}) marked "
        f"missing, since {reason}: no decision is made on a bar holding one until its "
        "candle is read"
    )


__all__ = [
    "CANDLE_LAST_READ_S",
    "CANDLE_SETTLE_S",
    "MISSING",
    "NO_TRADE",
    "SETTLE_NS",
    "WITH_CANDLE",
    "Release",
    "missing_note",
    "releases",
]
