"""Permuted bars (design 17, section 2.5): another price path with the same moves in
another order, a pure function of the bars and a seed.

A bar's move is four numbers kept together, so the bar stays a valid OHLC: its open
against the close before it, and its high, low and close against its own open, as price
ratios (prices stay positive whatever the order). Volume and open interest travel with
their bar's move; the bars' times stay where they are. Prices are rebuilt by products of
the ratios in bar order: a product is rounded the same way on every platform, which a
logarithm and an exponential are not.

    within a session    the moves of the bars after the session's first are permuted
                        among those bars (an intraday path in another order)
    across sessions     the move of each session's first bar, the overnight gap with the
                        opening bar's own range, is permuted among the sessions as one
                        more set (so on daily bars, one bar a session, the days are
                        permuted)

The history before `start` and the run are permuted apart, each in its own sessions, so
the history keeps its own moves and still ends at its last close, and the run's moves
stay in the run; the run's first bar opens a set of its own even when `start` falls
inside a session. The first bar of all keeps its open: the path starts where the real one
does. Each segment's moves sum to the same, so a permuted path ends at the real last
close. Prices are rebuilt from the moves and put on the tick grid (`_ohlc.valid`).

Several instruments are permuted with the same order of positions, which keeps their
moves at the same times together (their correlation); they must share one grid of bar
times (`_ohlc.same_grid`).

Draws: run r of a stream reads one 64-bit key per bar (draws r x bars onwards,
`_random.block`); the bars of a set are put in the order of their keys.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import _random
from ._ohlc import openers, same_grid, session_days, valid, with_prices


def source_positions(ts_open: np.ndarray, first_run_ns: int, keys: np.ndarray) -> np.ndarray:
    """For each bar position, the position whose move it takes (module docstring), from
    the bars' UTC nanosecond opens, the open of the run's first bar and one key per bar."""
    days = session_days(ts_open)
    in_run = np.asarray(ts_open) >= first_run_ns
    first = openers(days)
    first[1:] |= in_run[1:] != in_run[:-1]  # the run's first bar opens a set of its own
    moved_first = first.copy()
    moved_first[0] = False  # the first bar of all keeps its open: there is no close before it
    source = np.arange(len(days))
    for segment in (~in_run, in_run):
        sessions = np.flatnonzero(moved_first & segment)
        source[sessions] = sessions[np.argsort(keys[sessions], kind="stable")]
        inside = np.flatnonzero(~first & segment)
        source[inside] = inside[np.lexsort((keys[inside], days[inside]))]
    return source


def _moves(bars: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """(open over the close before, high, low and close over the open) of each bar, price
    ratios; the first bar's open is measured against itself."""
    o, h, low, c = (bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close"))
    before = np.concatenate((o[:1], c[:-1]))
    return o / before, h / o, low / o, c / o


def rebuilt(bars: pd.DataFrame, source: np.ndarray, tick: float) -> pd.DataFrame:
    """`bars` with each position taking the move, volume and open interest of
    `source[position]`, its prices rebuilt from the first open and put on the grid of
    `tick` rupees."""
    gap, high, low, close = (move[source] for move in _moves(bars))
    first = float(bars["open"].iloc[0])
    closes = first * np.cumprod(gap * close)
    opens = np.concatenate(([first], closes[:-1])) * gap
    prices = valid(opens, opens * high, opens * low, closes, tick)
    extra = {name: bars[name].to_numpy()[source] for name in ("volume", "oi") if name in bars}
    return with_prices(bars, prices, extra)


def permuted(
    frames: Sequence[pd.DataFrame],
    symbols: Sequence[str],
    ticks: Sequence[float],
    first_run_ns: int,
    seed: int,
    run: int,
) -> list[pd.DataFrame]:
    """Run `run` of the permuted bars of the stream `seed`: every instrument's `frames`
    (canonical layout, history first) permuted with one order of positions (module
    docstring), each on the grid of its tick in `ticks`. Raises DataError when the
    instruments' bars are not at the same times."""
    opens = same_grid(frames, symbols)
    keys = _random.block(seed, run, 1, len(opens))[0]
    source = source_positions(opens, first_run_ns, keys)
    return [rebuilt(frame, source, tick) for frame, tick in zip(frames, ticks, strict=True)]


__all__ = ["permuted", "rebuilt", "source_positions"]
