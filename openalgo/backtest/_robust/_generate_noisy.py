"""Noisy bars (design 17, section 2.6): a copy of the bars with every price moved a
little, a pure function of the bars, the settings and a seed.

Each price of each bar moves with its own probability (`Noise.chances`, open, high, low
and close; by default 0.40, 0.60, 0.50 and 0.75) by a uniform draw within plus or minus
`Noise.fraction` (0.20 by default) of that bar's average true range of `Noise.period`
bars (14), computed once on the base bars as they are (`ranges`). The bar is then made
valid: its prices on the tick grid, its high the largest and its low the smallest of the
four (`_ohlc.valid`). Times, volume and open interest stay as they are, history included
(the indicators warm up on noisy bars too).

The average true range is `openalgo.ta.atr` (Wilder's smoothing); the bars before it is
defined (the first `period` - 1) take its first value, and a series too short to have
one takes the mean of its bars' high less low.

Draws: run r of a stream reads 8 draws a bar over every instrument's bars, in
registration order (draws r x 8 x bars onwards, `_random.block`): for each bar, whether
each of its four prices moves (a fraction below its chance), then how far (a fraction
mapped onto plus or minus the range).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalgo.indicators import ta

from . import _random
from ._ohlc import PRICES, valid, with_prices

#: The draws a bar takes: for each of its four prices, whether and how far it moves.
DRAWS_PER_BAR = 2 * len(PRICES)


@dataclass(frozen=True)
class Noise:
    """How bars are made noisy: `chances` that the open, high, low and close move
    (fractions), `fraction` of the average true range a move is at most, and `period`,
    the bars of that average."""

    chances: tuple[float, float, float, float] = (0.40, 0.60, 0.50, 0.75)
    fraction: float = 0.20
    period: int = 14


def ranges(bars: pd.DataFrame, period: int) -> np.ndarray:
    """The average true range of each bar in rupees (module docstring)."""
    high, low, close = (bars[name].to_numpy(dtype=float) for name in ("high", "low", "close"))
    if len(close) >= period:
        values = np.asarray(ta.atr(high, low, close, period), dtype=float)
        known = np.flatnonzero(np.isfinite(values))
        if len(known):
            values[: known[0]] = values[known[0]]
            return values
    return np.full(len(close), float(np.mean(high - low)) if len(close) else 0.0)


def noisy_one(
    bars: pd.DataFrame, tick: float, noise: Noise, draws: np.ndarray, atr: np.ndarray
) -> pd.DataFrame:
    """`bars` moved by `draws` (8 fractions a bar, as the module docstring orders them)
    within `noise.fraction` of `atr`, made valid on the grid of `tick` rupees."""
    count = len(bars)
    whether, how_far = draws.reshape(2, len(PRICES), count)
    reach = noise.fraction * atr
    moved = []
    for k, name in enumerate(PRICES):
        shift = np.where(whether[k] < noise.chances[k], (2.0 * how_far[k] - 1.0) * reach, 0.0)
        moved.append(bars[name].to_numpy(dtype=float) + shift)
    return with_prices(bars, valid(*moved, tick))


def noisy(
    frames: Sequence[pd.DataFrame],
    ticks: Sequence[float],
    noise: Noise,
    seed: int,
    run: int,
    atrs: Sequence[np.ndarray] | None = None,
) -> list[pd.DataFrame]:
    """Run `run` of the noisy copies of `frames` (canonical layout) from the stream
    `seed` (module docstring), each on the grid of its tick in `ticks`. `atrs`, the
    average true range of each frame (`ranges`), saves computing them again."""
    atrs = atrs if atrs is not None else [ranges(frame, noise.period) for frame in frames]
    sizes = [len(frame) for frame in frames]
    draws = _random.fractions(_random.block(seed, run, 1, DRAWS_PER_BAR * sum(sizes))[0])
    copies, offset = [], 0
    for frame, tick, atr, size in zip(frames, ticks, atrs, sizes, strict=True):
        part = draws[offset : offset + DRAWS_PER_BAR * size]
        copies.append(noisy_one(frame, tick, noise, part, atr))
        offset += DRAWS_PER_BAR * size
    return copies


__all__ = ["DRAWS_PER_BAR", "Noise", "noisy", "noisy_one", "ranges"]
