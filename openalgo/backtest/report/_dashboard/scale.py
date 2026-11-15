"""Axis arithmetic of the dashboard's charts: round tick steps, ticks, axes widened to
round ends, evenly spaced bands and histogram bins.

A round step is m x 10^k with m in (1, 2, 5): for a span cut into about `count` parts
it is the round step nearest the raw step span / count on a log scale, so the
thresholds between multiples are the geometric midpoints sqrt(2), sqrt(10) and
sqrt(50).
"""

import math
from typing import NamedTuple

import numpy as np

#: Multiples of a power of ten that make a round step, and the geometric midpoints
#: between neighbours (sqrt(1 x 2), sqrt(2 x 5), sqrt(5 x 10)) where the choice turns.
ROUND_MULTIPLES = (1.0, 2.0, 5.0, 10.0)
_MIDPOINTS = (math.sqrt(2.0), math.sqrt(10.0), math.sqrt(50.0))
#: Steps tried for a value axis that must keep a minimum gap between labels: the round
#: multiples and 2.5, smallest first.
AXIS_MULTIPLES = (1.0, 2.0, 2.5, 5.0)
#: Times an axis is widened to round ends before it is taken as it is.
WIDEN_PASSES = 10
#: Decimals kept on a tick, so that 0.1 + 0.2 reads 0.3.
TICK_DECIMALS = 12


def round_step(low: float, high: float, count: int) -> float:
    """The round step nearest (high - low) / count on a log scale; 0 for an empty
    span."""
    raw = abs(high - low) / max(count, 1)
    if raw == 0 or not math.isfinite(raw):
        return 0.0
    power = 10.0 ** math.floor(math.log10(raw))
    error = raw / power
    multiple = next(
        (m for m, turn in zip(ROUND_MULTIPLES, _MIDPOINTS, strict=False) if error < turn),
        ROUND_MULTIPLES[-1],
    )
    return multiple * power


def ticks(low: float, high: float, count: int) -> list[float]:
    """Multiples of the round step for `count` parts that lie in [low, high]."""
    step = round_step(low, high, count)
    if step == 0:
        return [round(low, TICK_DECIMALS)]
    first, last = math.ceil(low / step - 1e-9), math.floor(high / step + 1e-9)
    return [round(i * step, TICK_DECIMALS) for i in range(first, last + 1)]


def widen(low: float, high: float, count: int = 10) -> tuple[float, float]:
    """The axis [low, high] widened outward to multiples of its round step, repeated
    until the step no longer changes. A zero span is returned as it is."""
    previous = None
    for _ in range(WIDEN_PASSES):
        step = round_step(low, high, count)
        if step == 0 or step == previous:
            break
        low, high = math.floor(low / step) * step, math.ceil(high / step) * step
        previous = step
    return low, high


class Linear(NamedTuple):
    """A linear map from a value domain [low, high] onto pixels [start, end]."""

    low: float
    high: float
    start: float
    end: float

    def __call__(self, value: float) -> float:
        span = self.high - self.low
        share = 0.5 if span == 0 else (value - self.low) / span
        return self.start + share * (self.end - self.start)


def bands(count: int, length: float, padding: float) -> tuple[list[float], float]:
    """`count` evenly spaced bands across `length` pixels: the start of each band and
    their width. `padding` is the share of a step left empty between bands and, as
    much again, half at each end."""
    step = length / max(1.0, count - padding + 2 * padding)
    offset = (length - step * (count - padding)) / 2
    return [offset + step * i for i in range(count)], step * (1 - padding)


def bins(values: np.ndarray, low: float, high: float, count: int) -> list[tuple[float, float, int]]:
    """A histogram of `values` over [low, high]: edges at the round ticks for `count`
    parts that fall strictly inside, so the first and last bins may be narrower. Each
    bin is (left, right, how many values): left-closed, and the last one closed on
    both sides."""
    edges = [low, *(t for t in ticks(low, high, count) if low < t < high), high]
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    return [(edges[i], edges[i + 1], int(counts[i])) for i in range(len(edges) - 1)]


def quartiles(values: np.ndarray) -> tuple[float, float, float, float, float]:
    """Lowest, first quartile, median, third quartile and highest of `values`, the
    quartiles by linear interpolation between order statistics."""
    known = np.sort(values[np.isfinite(values)])
    q1, median, q3 = np.quantile(known, [0.25, 0.5, 0.75])
    return float(known[0]), float(q1), float(median), float(q3), float(known[-1])


__all__ = [
    "AXIS_MULTIPLES",
    "Linear",
    "bands",
    "bins",
    "quartiles",
    "round_step",
    "ticks",
    "widen",
]
