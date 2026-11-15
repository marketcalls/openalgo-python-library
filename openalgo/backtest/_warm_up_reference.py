"""The reference bars every indicator's warm-up is measured on (decision D25; the
measurement is in `_warm_up_measure`).

The warm-up of a `self.I` indicator is measured on bars the engine makes itself, never on
the run's bars, so it is a function of the indicator call alone (the function, its
parameters and the bar fields it reads): the same for any symbol, interval, `start`,
`end` or price history. Live trading keeps the same window (08-live-trading section
11). The bars must therefore hold the shapes that make an indicator slow to settle on
real prices: long trends (a parabolic stop-and-reverse keeps its state through a whole
trend), quiet and wide ranges, sharp reversals, and opening gaps.

`REFERENCE_BARS` bars, made from integers so that they are the same bit for bit on every
platform and NumPy version (no random generator, no transcendental function). The
splitmix64 mixer of a number gives each draw. The bars are cut into regimes of 32 to
1,024 bars, each of one kind:

- a trend up (30 to 150 percent over the regime, spread evenly over its bars) or down
  (20 to 60 percent), turned back when the price has left Rs 400 to Rs 2,500;
- a range: the price is pulled back towards the regime's first close by 1/128 of the
  distance a bar;
- a reversal: a trend of 20 to 60 percent for half the regime and back for the other
  half (down first at or above Rs 1,000.00, up first below).

Every bar also moves by a uniform draw of up to 0.5 to 5 percent of the price (the
regime's volatility: a large share to a volatile small one), and one bar in `JUMP_ODDS`
opens with a jump of 2 to 8 percent that the price keeps (an opening gap); every other
open is the previous close moved by up to 0.25 percent. A move is a percentage of the
close at the start of its block of `BLOCK_BARS` bars, so the walk is close to a
multiplicative one without a logarithm. Prices are in paise, divided by 100 at the end
(one correctly rounded division), and never below `FLOOR_PAISE`. Volume is 1,00,000 to
3,00,000 units a bar, three times that on a jump. Bar k opens at 09:15 IST on day k from
2000-01-03 and closes a regular NSE session later.

Whether the bars are hard enough is tested: the warm-up each common indicator measures
here reproduces its full-history values on a battery of differently shaped series
(test/backtest/test_warm_up_battery.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._layout import IST_OFFSET_NANOS, NANOS_PER_MINUTE
from ._series import FIELDS, TIMES
from .data._sessions import NANOS_PER_DAY

# Bars of the reference series: the longest warm-up it can measure is half of it, 32,768
# bars, enough for an exponential average of about 3,800 bars (it needs about 8.5 times
# its length to come within the tolerance).
REFERENCE_BARS = 65_536
# The splitmix64 mixer (its published increment and multipliers).
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
SPLITMIX_SHIFTS = (30, 27, 31)
# The streams, interleaved: stream s of draw k is mixer output STREAMS k + s. Bars draw
# from the first seven, regimes from the last four (draw k of a regime stream is the
# k-th regime's).
MOVE, OPENING, RANGE, VOLUME, JUMP, JUMP_SIZE, SIGN = range(7)
LENGTH, KIND, SIZE, VOLATILITY = range(7, 11)
STREAMS = 11
# Prices in paise; moves in basis points (1/100 of a percent), and a trend's drift a bar
# in millionths.
PAISE_PER_RUPEE = 100
BASIS_POINTS = 10_000
MILLIONTHS = 1_000_000
MILLIONTHS_PER_BASIS_POINT = MILLIONTHS // BASIS_POINTS
FIRST_CLOSE_PAISE = 100_000
LOWEST_LEVEL_PAISE = 40_000
HIGHEST_LEVEL_PAISE = 250_000
FLOOR_PAISE = 5_000
# The regimes (module docstring). A size is drawn as a whole number below SIZE_DRAWS
# and brought to the kind's bounds.
SHORTEST_REGIME, LONGEST_REGIME = 32, 1_024
TREND_UP, TREND_DOWN, RANGE_BOUND, REVERSAL = range(4)
KINDS = 4
SIZE_DRAWS = 1 << 30
RISE_BASIS_POINTS = (3_000, 15_000)
FALL_BASIS_POINTS = (2_000, 6_000)
RANGE_PULL_BARS = 128
VOLATILITY_BASIS_POINTS = (50, 500)
BLOCK_BARS = 64
# Openings: one bar in JUMP_ODDS jumps by 2 to 6 percent; the others open within 0.25
# percent of the previous close.
JUMP_ODDS = 48
JUMP_BASIS_POINTS = (200, 800)
OPENING_BASIS_POINTS = 25
# Volume in units: 1,00,000 to 3,00,000 a bar, JUMP_VOLUME times that on a jump.
LEAST_VOLUME, MOST_VOLUME = 100_000, 300_000
JUMP_VOLUME = 3
# Bar k opens at FIRST_OPEN_IST on day k and closes a regular NSE session later.
FIRST_OPEN_IST = "2000-01-03 09:15"
SESSION_MINUTES = 375


def _mixed(numbers: np.ndarray) -> np.ndarray:
    """The splitmix64 output for each state number (uint64, wrapping arithmetic)."""
    first, second, third = (np.uint64(shift) for shift in SPLITMIX_SHIFTS)
    value = (numbers + np.uint64(1)) * np.uint64(SPLITMIX_INCREMENT)
    value = (value ^ (value >> first)) * np.uint64(SPLITMIX_MULTIPLIER_1)
    value = (value ^ (value >> second)) * np.uint64(SPLITMIX_MULTIPLIER_2)
    return value ^ (value >> third)


def _uniform(stream: int, low: int, high: int, count: int = REFERENCE_BARS) -> np.ndarray:
    """`count` whole numbers from `low` to `high`, both included, from `stream`."""
    numbers = np.arange(count, dtype=np.uint64) * np.uint64(STREAMS) + np.uint64(stream)
    spread = np.uint64(high - low + 1)
    return (_mixed(numbers) % spread).astype(np.int64) + low


@dataclass(frozen=True)
class _Regime:
    """One regime: its first bar, its bar count, its kind, its first close in paise, its
    drift a bar in millionths of the price (signed; for a reversal, of its first half)
    and its volatility in basis points."""

    first: int
    bars: int
    kind: int
    anchor: int
    drift: int
    volatility: int

    def blocks(self) -> list[tuple[int, int]]:
        """[start, stop) of each block of at most BLOCK_BARS bars, a reversal's halves
        apart (its drift turns at the half)."""
        stop = self.first + self.bars
        turns = [self.first + (self.bars + 1) // 2] if self.kind == REVERSAL else []
        edges = sorted({*range(self.first, stop, BLOCK_BARS), *turns, stop})
        return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]

    def drift_at(self, bar: int, level: int) -> int:
        """The drift of bar `bar` (from the regime's first) in paise at `level`."""
        if self.kind == RANGE_BOUND:
            return (self.anchor - level) // RANGE_PULL_BARS
        rate = -self.drift if self.kind == REVERSAL and 2 * bar >= self.bars else self.drift
        return level * rate // MILLIONTHS


def _regime(
    first: int, bars: int, kind: int, size_draw: int, level: int, volatility: int
) -> _Regime:
    """The regime starting at bar `first` from `level` paise (module docstring): a trend
    is turned back outside the price band, and a reversal goes down first at or above
    the first close."""
    if kind in (TREND_UP, TREND_DOWN):
        if level > HIGHEST_LEVEL_PAISE:
            kind = TREND_DOWN
        elif level < LOWEST_LEVEL_PAISE:
            kind = TREND_UP
    low, high = RISE_BASIS_POINTS if kind == TREND_UP else FALL_BASIS_POINTS
    size = low + size_draw % (high - low + 1)
    span = bars // 2 if kind == REVERSAL else bars
    falls = kind == TREND_DOWN or (kind == REVERSAL and level >= FIRST_CLOSE_PAISE)
    drift = size * MILLIONTHS_PER_BASIS_POINT // max(span, 1)
    return _Regime(first, bars, kind, level, -drift if falls else drift, volatility)


def _regimes() -> list[tuple[int, int, int, int, int]]:
    """(first bar, bars, kind, SIZE draw, volatility in basis points) of every regime,
    from the draws alone; the walk turns each into a `_Regime` at its first close."""
    most = REFERENCE_BARS // SHORTEST_REGIME + 1
    lengths = _uniform(LENGTH, SHORTEST_REGIME, LONGEST_REGIME, most).tolist()
    kinds = _uniform(KIND, 0, KINDS - 1, most).tolist()
    sizes = _uniform(SIZE, 0, SIZE_DRAWS - 1, most).tolist()
    volatilities = _uniform(VOLATILITY, *VOLATILITY_BASIS_POINTS, most).tolist()
    shapes, first = [], 0
    for length, kind, size, volatility in zip(lengths, kinds, sizes, volatilities, strict=True):
        if first >= REFERENCE_BARS:
            break
        bars = min(length, REFERENCE_BARS - first)
        shapes.append((first, bars, kind, size, volatility))
        first += bars
    return shapes


def _closes_and_jumps() -> tuple[np.ndarray, np.ndarray]:
    """(the close of every bar in paise, the opening jump of every bar in paise)."""
    moves = _uniform(MOVE, -BASIS_POINTS, BASIS_POINTS)
    jumped = _uniform(JUMP, 0, JUMP_ODDS - 1) == 0
    jump_sizes = np.where(_uniform(SIGN, 0, 1) == 0, -1, 1) * _uniform(
        JUMP_SIZE, *JUMP_BASIS_POINTS
    )
    closes = np.empty(REFERENCE_BARS, dtype=np.int64)
    jumps = np.zeros(REFERENCE_BARS, dtype=np.int64)
    level = FIRST_CLOSE_PAISE
    for first, bars, kind, size, volatility in _regimes():
        regime = _regime(first, bars, kind, size, level, volatility)
        for start, stop in regime.blocks():
            part = slice(start, stop)
            noise = moves[part] * volatility * level // (BASIS_POINTS * BASIS_POINTS)
            jumps[part] = np.where(jumped[part], jump_sizes[part] * level // BASIS_POINTS, 0)
            drift = regime.drift_at(start - first, level)
            path = np.cumsum(noise + jumps[part] + drift)
            closes[part] = np.maximum(level + path, FLOOR_PAISE)
            level = int(closes[part][-1])
    return closes, jumps


def reference_columns() -> dict[str, np.ndarray]:
    """The reference bars as store columns (`_series`): `ts_open` and `ts_close` int64
    UTC nanoseconds, prices in rupees, volume in units. Each call builds them afresh
    (about 20 ms); `_warm_up_measure.ReferenceNeeds` keeps one copy per run or sweep."""
    closes, jumps = _closes_and_jumps()
    before = np.concatenate(([FIRST_CLOSE_PAISE], closes[:-1]))
    nudge = _uniform(OPENING, -OPENING_BASIS_POINTS, OPENING_BASIS_POINTS)
    opens = np.maximum(before + jumps + before * nudge // BASIS_POINTS, FLOOR_PAISE)
    spans = _uniform(RANGE, 0, 2 * VOLATILITY_BASIS_POINTS[1]) * before // BASIS_POINTS
    highs = np.maximum(opens, closes) + spans // 2
    lows = np.maximum(np.minimum(opens, closes) - (spans - spans // 2), FLOOR_PAISE // 2)
    first = int(pd.Timestamp(FIRST_OPEN_IST).value) - IST_OFFSET_NANOS
    ts_open = first + np.arange(REFERENCE_BARS, dtype=np.int64) * NANOS_PER_DAY
    prices = {"open": opens, "high": highs, "low": lows, "close": closes}
    columns = {name: values / PAISE_PER_RUPEE for name, values in prices.items()}
    volume = _uniform(VOLUME, LEAST_VOLUME, MOST_VOLUME)
    columns["volume"] = np.where(jumps != 0, volume * JUMP_VOLUME, volume).astype(float)
    columns["ts_open"] = ts_open
    columns["ts_close"] = ts_open + SESSION_MINUTES * NANOS_PER_MINUTE
    return {name: columns[name] for name in (*TIMES, *FIELDS)}


__all__ = ["REFERENCE_BARS", "reference_columns"]
