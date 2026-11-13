"""The reference bars every indicator's warm-up is measured on (decision D25).

The warm-up of a `self.I` indicator is the window-stability length of
`_causality_checks.settle_length`: the shortest rolling window, doubling from 16 bars,
whose values at sampled bars agree with the values computed on all the bars within the
window tolerance. It is measured here, on bars the engine makes itself, never on the
run's bars. So the warm-up is a function of the indicator call alone (the function, its
parameters and the bar fields it reads): the same for any symbol, interval, `start`,
`end` or price history, counted in the bars of the indicator's own series, and no
decision depends on a bar after it. Live trading keeps the same window (08-live-trading
section 11).

The reference bars: `REFERENCE_BARS` of them, made from integers so that they are the
same bit for bit on every platform and NumPy version (no random generator, no
transcendental function). The splitmix64 mixer of each bar's number gives four
pseudo-random streams: the close's move, the open's gap from the previous close, the
bar's range and its volume. The close is a walk in paise that starts at Rs 1,000.00,
moves up to 2 percent a bar (a spread of about 1.2 percent a bar, the daily volatility
of a large NSE share) and is pulled back towards Rs 1,000.00 by 1/256 of its distance a
bar, so it stays between about Rs 660 and Rs 1,490; prices are paise divided by 100,
one correctly rounded division. The 256-bar pull leaves the walk room to trend over
hundreds of bars, so a long average starts far from the price, as on real shares. Bar
k opens at 09:15 IST on day k from 2000-01-03.

An indicator that no window shorter than `REFERENCE_BARS` settles (a cumulative series
such as on-balance volume) has no warm-up length: no history makes it ready.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._causality_checks import WINDOW_TOLERANCE, Outputs, settle_length
from ._layout import IST_OFFSET_NANOS, NANOS_PER_MINUTE
from ._series import FIELDS, TIMES

# Bars of the reference series: the longest warm-up it can measure is half of it, 8,192
# bars, four times the 2,048 of a 200-bar EMA, the longest lookback in common use.
REFERENCE_BARS = 16_384
# The splitmix64 mixer (its published increment and multipliers).
SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
SPLITMIX_SHIFTS = (30, 27, 31)
# The four streams, interleaved: stream s of bar k is mixer output 4 k + s.
MOVE, GAP, RANGE, VOLUME = range(4)
STREAMS = 4
# The walk of the close, in paise (see the module docstring).
FIRST_CLOSE_PAISE = 100_000
LARGEST_MOVE_PAISE = 2_000
REVERSION_BARS = 256
LARGEST_GAP_PAISE = 100
LARGEST_RANGE_PAISE = 1_000
PAISE_PER_RUPEE = 100
# Volume in units: from 1,00,000 to 2,00,000 a bar.
LEAST_VOLUME = 100_000
VOLUME_SPREAD = 100_000
# Bar k opens at FIRST_OPEN_IST on day k and closes a regular NSE session later.
FIRST_OPEN_IST = "2000-01-03 09:15"
SESSION_MINUTES = 375
NANOS_PER_DAY = 24 * 60 * NANOS_PER_MINUTE


def _mixed(numbers: np.ndarray) -> np.ndarray:
    """The splitmix64 output for each state number (uint64, wrapping arithmetic)."""
    first, second, third = (np.uint64(shift) for shift in SPLITMIX_SHIFTS)
    value = (numbers + np.uint64(1)) * np.uint64(SPLITMIX_INCREMENT)
    value = (value ^ (value >> first)) * np.uint64(SPLITMIX_MULTIPLIER_1)
    value = (value ^ (value >> second)) * np.uint64(SPLITMIX_MULTIPLIER_2)
    return value ^ (value >> third)


def _uniform(stream: int, low: int, high: int) -> np.ndarray:
    """One whole number from `low` to `high`, both included, per bar, from `stream`."""
    numbers = np.arange(REFERENCE_BARS, dtype=np.uint64) * np.uint64(STREAMS) + np.uint64(stream)
    spread = np.uint64(high - low + 1)
    return (_mixed(numbers) % spread).astype(np.int64) + low


def _closes() -> np.ndarray:
    """The walk of the close in paise."""
    moves = _uniform(MOVE, -LARGEST_MOVE_PAISE, LARGEST_MOVE_PAISE).tolist()
    closes, level = [], FIRST_CLOSE_PAISE
    for move in moves:
        level += move - (level - FIRST_CLOSE_PAISE) // REVERSION_BARS
        closes.append(level)
    return np.asarray(closes, dtype=np.int64)


def reference_columns() -> dict[str, np.ndarray]:
    """The reference bars as store columns (`_series`): `ts_open` and `ts_close` int64
    UTC nanoseconds, prices in rupees, volume in units. Each call builds them afresh
    (about 4 ms); `ReferenceNeeds` keeps one copy per run or sweep."""
    closes = _closes()
    opens = np.concatenate(([FIRST_CLOSE_PAISE], closes[:-1])) + _uniform(
        GAP, -LARGEST_GAP_PAISE, LARGEST_GAP_PAISE
    )
    spans = _uniform(RANGE, 0, LARGEST_RANGE_PAISE)
    highs = np.maximum(opens, closes) + spans // 2
    lows = np.minimum(opens, closes) - (spans - spans // 2)
    first = int(pd.Timestamp(FIRST_OPEN_IST).value) - IST_OFFSET_NANOS
    ts_open = first + np.arange(REFERENCE_BARS, dtype=np.int64) * NANOS_PER_DAY
    prices = {"open": opens, "high": highs, "low": lows, "close": closes}
    columns = {name: values / PAISE_PER_RUPEE for name, values in prices.items()}
    columns["volume"] = _uniform(VOLUME, LEAST_VOLUME, LEAST_VOLUME + VOLUME_SPREAD).astype(float)
    columns["ts_open"] = ts_open
    columns["ts_close"] = ts_open + SESSION_MINUTES * NANOS_PER_MINUTE
    return {name: columns[name] for name in (*TIMES, *FIELDS)}


class ReferenceNeeds:
    """Warm-ups measured on the reference bars, each once per indicator label (a run's
    or a sweep's; the bars are built on first use). `tolerance` is the window
    tolerance, relative."""

    def __init__(self, tolerance: float = WINDOW_TOLERANCE) -> None:
        self.tolerance = tolerance
        self._columns: dict[str, np.ndarray] | None = None
        self._times: pd.DatetimeIndex | None = None
        self._found: dict[str, int | None] = {}

    def columns(self) -> dict[str, np.ndarray]:
        """The reference bars as store columns (`reference_columns`)."""
        if self._columns is None:
            self._columns = reference_columns()
        return self._columns

    def bars(self, start: int, stop: int) -> dict[str, Any]:
        """Bars [start, stop) of the reference as a recipe reads them (`_views.Recipe`):
        the price and volume fields and "times", their IST opens."""
        columns = self.columns()
        if self._times is None:
            utc = pd.to_datetime(columns["ts_open"], utc=True)
            self._times = pd.DatetimeIndex(utc.tz_convert("Asia/Kolkata"), name="time")
        window: dict[str, Any] = {name: columns[name][start:stop] for name in FIELDS}
        window["times"] = self._times[start:stop]
        return window

    def need(self, label: str, outputs: Outputs) -> int | None:
        """The warm-up in bars of the indicator labelled `label`, whose causal outputs on
        any bars `outputs` computes; None when no window settles on the reference bars
        (or the indicator cannot be computed on all of them)."""
        if label not in self._found:
            self._found[label] = self._measure(outputs)
        return self._found[label]

    def _measure(self, outputs: Outputs) -> int | None:
        def compute(start: int, stop: int) -> list[np.ndarray]:
            return outputs(self.bars(start, stop))

        try:
            whole = compute(0, REFERENCE_BARS)
        except ValueError:
            return None  # stated by the caller as an indicator no window settles
        return settle_length(compute, whole, REFERENCE_BARS, self.tolerance)


__all__ = ["REFERENCE_BARS", "ReferenceNeeds", "reference_columns"]
