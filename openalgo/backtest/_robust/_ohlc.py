"""What the bar generators share: the sessions of a series of bars, prices put on the
tick grid, a bar made valid, and a copy of bars with new prices.

Bars are in the canonical layout (`_layout.BAR_COLUMNS`): `ts_open` and `ts_close` UTC
nanoseconds, prices in rupees. A session is an IST calendar day: Indian sessions never
cross midnight IST, so the bars of one session share their IST date.

A price on the tick grid is a whole number of ticks, at least one tick (a generator never
makes a price of zero or below), rounded to `GRID_DECIMALS` places so that 503.15 is
stored as 503.15 and not as the 503.15000000000003 that 10,063 x 0.05 gives. A valid bar
has its high the largest and its low the smallest of its four prices.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .._layout import IST_OFFSET_NANOS
from ..data._sessions import NANOS_PER_DAY
from ..errors import DataError

#: The decimals a price on the tick grid is rounded to: any tick of a hundredth of a
#: paisa or more (four decimals of a rupee) stays exact, and eight decimals are far
#: above the error of a float of a few lakh rupees.
GRID_DECIMALS = 8
#: The price columns, in the order a generator handles them.
PRICES = ("open", "high", "low", "close")


def session_days(ts_open: np.ndarray) -> np.ndarray:
    """The IST day of each bar (days since 1970-01-01), from its UTC nanosecond open."""
    return (np.asarray(ts_open, dtype=np.int64) + IST_OFFSET_NANOS) // NANOS_PER_DAY


def openers(days: np.ndarray) -> np.ndarray:
    """Whether each bar opens its session: the first bar, and every bar whose IST day
    differs from the bar before it."""
    first = np.ones(len(days), dtype=bool)
    first[1:] = days[1:] != days[:-1]
    return first


def on_grid(prices: np.ndarray, tick: float) -> np.ndarray:
    """`prices` rounded to the nearest whole number of `tick` rupees, at least one tick
    (module docstring)."""
    ticks = np.maximum(np.round(np.asarray(prices, dtype=float) / tick), 1.0)
    return np.round(ticks * tick, GRID_DECIMALS)


def valid(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, tick: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four prices on the tick grid with the high the largest and the low the smallest
    of each bar (module docstring)."""
    o, h, low, c = (on_grid(values, tick) for values in (opens, highs, lows, closes))
    every = np.vstack((o, h, low, c))
    return o, every.max(axis=0), every.min(axis=0), c


def with_prices(
    bars: pd.DataFrame, prices: tuple[np.ndarray, ...], extra: dict[str, np.ndarray] | None = None
) -> pd.DataFrame:
    """A copy of `bars` (its times, attrs and other columns) with the open, high, low and
    close of `prices`, and the columns of `extra` replaced (volume and open interest that
    travel with their bars)."""
    copy = bars.copy()
    for name, values in zip(PRICES, prices, strict=True):
        copy[name] = values
    for name, values in (extra or {}).items():
        copy[name] = values
    return copy


def same_grid(frames: Sequence[pd.DataFrame], symbols: Sequence[str]) -> np.ndarray:
    """The bar opens every one of `frames` shares, UTC nanoseconds. Raises DataError
    when two instruments' bars are not at the same times, which a generator that moves
    several instruments' bars together needs."""
    first = frames[0]["ts_open"].to_numpy(dtype=np.int64)
    for symbol, frame in zip(symbols[1:], frames[1:], strict=True):
        opens = frame["ts_open"].to_numpy(dtype=np.int64)
        if len(opens) != len(first) or not np.array_equal(opens, first):
            raise DataError(
                f"the bars of {symbols[0]} and {symbol} are not at the same times "
                f"({len(first)} and {len(opens)} bars): several instruments are permuted "
                "together only on one grid of bars, so give bars at the same times for "
                "every symbol"
            )
    return first


__all__ = [
    "GRID_DECIMALS",
    "PRICES",
    "on_grid",
    "openers",
    "same_grid",
    "session_days",
    "valid",
    "with_prices",
]
