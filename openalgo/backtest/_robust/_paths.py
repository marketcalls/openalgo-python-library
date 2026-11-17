"""Equity paths of trades booked one after another, and the fans that draw them.

A path books trades' net P&L in rupees one after another on the start capital at fixed
rupee size (trades are sized in units, not as a share of equity), as
`metrics.monte_carlo(resample="trades")` does. Its wealth after k trades is the equity
over the capital; a path whose equity reaches zero or below is ruined there and its
wealth stays 0 (a cumulative return of -100 percent) to the end. Its maximum drawdown is
the deepest fall of the wealth below its running peak, the capital included, as a
fraction (zero or negative): the kernel's close drawdown of the same path.

A fan summarises many paths at no more than `FAN_POINTS` evenly spaced numbers of
trades: the percentile bands of `BAND_LEVELS` (5 to 95 and 25 to 75 percent, the median)
of the cumulative return, `SAMPLED_PATHS` runs drawn thin, and the run's own path (design
17, section 2: about 100 sampled paths keep 1,000 runs light on the page).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from . import _random

#: The most points a fan is drawn at: the kernel's band points (`MAX_BAND_POINTS` of
#: `oa_bt::metrics::montecarlo`), so every fan of a report has the same resolution.
FAN_POINTS = 201
#: The percentile levels of a fan's bands, fractions (design 17, section 2).
BAND_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
#: The runs a fan draws as thin paths (design 17, section 2).
SAMPLED_PATHS = 100
#: The draws a check computes at once, whatever its number of runs: about 16 MB of
#: 64-bit values, so a long trade list does not need memory in proportion to its runs.
DRAWS_PER_CHUNK = 2_000_000


def fan_steps(trades: int) -> np.ndarray:
    """The numbers of trades a fan is drawn at: every one from 0 for fewer than
    `FAN_POINTS`, else `FAN_POINTS` evenly spaced from 0 to `trades`, rounded."""
    if trades < FAN_POINTS:
        return np.arange(trades + 1)
    last = FAN_POINTS - 1
    return np.round(np.arange(FAN_POINTS) * trades / last).astype(np.int64)


@dataclass(frozen=True)
class Walk:
    """Paths of trades (module docstring), one row per run: `wealth` (runs, trades + 1),
    the wealth after each number of trades from 0 (1 at the start); `final_return` and
    `max_drawdown` (fractions) and `ruined` (bool) per run."""

    wealth: np.ndarray
    final_return: np.ndarray
    max_drawdown: np.ndarray
    ruined: np.ndarray


def walk(pnl: np.ndarray, capital: float) -> Walk:
    """The paths of `pnl`, (runs, trades) net P&L in rupees booked in column order on
    `capital` rupees."""
    runs, count = pnl.shape
    equity = capital + np.cumsum(pnl, axis=1)
    below_zero = equity <= 0.0
    ruined = below_zero.any(axis=1)
    ruin_at = np.where(ruined, below_zero.argmax(axis=1), count)
    alive = np.arange(count)[None, :] < ruin_at[:, None]
    wealth = np.ones((runs, count + 1))
    wealth[:, 1:] = np.where(alive, equity / capital, 0.0)
    peaks = np.maximum.accumulate(wealth, axis=1)
    drawdown = np.minimum((wealth / peaks - 1.0).min(axis=1), 0.0)
    return Walk(wealth, wealth[:, -1] - 1.0, drawdown, ruined)


def bands(cumulative: np.ndarray) -> np.ndarray:
    """The `BAND_LEVELS` percentiles of (runs, points) cumulative returns at each
    point, (points, levels), by linear interpolation between order statistics (the
    kernel's quantile definition)."""
    return np.quantile(cumulative, BAND_LEVELS, axis=0, method="linear").T


@dataclass(frozen=True, eq=False)
class Fan:
    """A fan of paths (module docstring): `steps` the numbers of trades of each point,
    `bands` (points, levels) cumulative returns at `BAND_LEVELS`, `samples` (paths,
    points) the thin paths, `real` (points,) the run's own path or None, `runs` how many
    paths the bands summarise. Cumulative returns are fractions of the capital."""

    steps: np.ndarray
    bands: np.ndarray
    samples: np.ndarray
    real: np.ndarray | None
    runs: int


@dataclass(frozen=True, eq=False)
class PathSet:
    """Paths measured run by run: `final_return`, `max_drawdown` (fractions) and
    `ruined` per run, and `at_steps` (runs, points), each run's cumulative return at the
    fan's steps."""

    final_return: np.ndarray
    max_drawdown: np.ndarray
    ruined: np.ndarray
    at_steps: np.ndarray


def measure(chunks: Iterable[np.ndarray], capital: float, steps: np.ndarray) -> PathSet:
    """The paths of every (runs, trades) chunk of net P&L, in order, booked on
    `capital` rupees."""
    walks = [walk(rows, capital) for rows in chunks]
    return PathSet(
        np.concatenate([w.final_return for w in walks]),
        np.concatenate([w.max_drawdown for w in walks]),
        np.concatenate([w.ruined for w in walks]),
        np.concatenate([w.wealth[:, steps] - 1.0 for w in walks]),
    )


def own_path(pnl: np.ndarray, capital: float, steps: np.ndarray) -> np.ndarray:
    """The cumulative return of the run's own trades, in exit order, at `steps`."""
    return walk(pnl[None, :], capital).wealth[0, steps] - 1.0


def fan(paths: PathSet, steps: np.ndarray, real: np.ndarray | None) -> Fan:
    """The fan of `paths`: their bands, their first `SAMPLED_PATHS` runs drawn thin, and
    the run's own path `real`."""
    samples = paths.at_steps[:SAMPLED_PATHS]
    return Fan(steps, bands(paths.at_steps), samples, real, len(paths.final_return))


def resampled_trades(pnl: np.ndarray, capital: float, seed: int, runs: int) -> np.ndarray:
    """The first `runs` paths of `metrics.monte_carlo(resample="trades", seed=seed)`, as
    (runs, trades) net P&L in rupees, drawn as the kernel draws them: each run takes as
    many draws from the stream as it books, so a run ruined after k trades leaves the
    next run to start k draws on. The draws after a run's ruin are left in its row; a
    walk puts its wealth at 0 there whatever they are."""
    count = len(pnl)
    rows = np.empty((runs, count))
    offset = 0
    for run in range(runs):
        drawn = pnl[_random.below(_random.outputs(seed, offset, count), count)]
        rows[run] = drawn
        hit = np.flatnonzero(capital + np.cumsum(drawn) <= 0.0)
        offset += int(hit[0]) + 1 if len(hit) else count
    return rows


__all__ = [
    "BAND_LEVELS",
    "DRAWS_PER_CHUNK",
    "FAN_POINTS",
    "SAMPLED_PATHS",
    "Fan",
    "PathSet",
    "Walk",
    "bands",
    "fan",
    "fan_steps",
    "measure",
    "own_path",
    "resampled_trades",
    "walk",
]
