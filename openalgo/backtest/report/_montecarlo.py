"""Monte Carlo by bootstrap of a `Performance`: many alternative runs drawn with
replacement from the run's own session returns, or from its closed trades, and the
spread of their outcomes. The kernel draws and measures the paths (`oa_bt::metrics::
montecarlo`); the same inputs and seed give the same simulations, bit for bit.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from .._native import native
from ..errors import ConfigError
from ._kernel import call
from ._performance import Performance
from ._periods import NANOS_PER_DAY

#: Paths drawn by default: enough for stable 5th and 95th percentiles.
DEFAULT_SIMULATIONS = 1_000
#: The seed used when none is given. Any fixed value would do: it only makes a report
#: reproduce the same simulations every time it is built.
DEFAULT_SEED = 2_026
#: What a study can resample.
RESAMPLE = ("sessions", "trades")
#: The figures of each path.
FIGURES = ("final_return", "cagr", "max_drawdown", "sharpe")
#: Percentile levels of the summary table.
PERCENTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class MonteCarlo:
    """A Monte Carlo study of one run.

    resample: "sessions" or "trades"; simulations: paths drawn; seed: the generator's.
    runs: one row per path: final_return (a fraction), cagr (a fraction per year, over
        the run's elapsed time), max_drawdown (a fraction, zero or negative, on the
        path's closes) and sharpe (annualised by sessions per year, or for trades by the
        run's trades per year).
    percentiles: the figures of `runs` at each of `PERCENTILE_LEVELS` (rows, fractions
        from 0 to 1), by linear interpolation between order statistics; NaN figures
        are left out.
    band: the cumulative return of the paths after each number of draws (index
        "draws"), at the 5th, 25th, 50th, 75th and 95th percentiles (columns 0.05 to
        0.95), at no more than 201 evenly spaced numbers of draws.
    actual: the same four figures for the run's own sequence, measured the same way.
    actual_path: the cumulative return of the run's own sequence after each number of
        draws (index "draws", from 0); for trades in exit order on the start equity.
    bust, goal: the thresholds asked for (a maximum drawdown, zero or negative, and a
        final return), or None.
    bust_probability: the share of paths whose maximum drawdown reached `bust`;
    goal_probability: the share whose final return reached `goal`; NaN when not asked.
    """

    resample: str
    simulations: int
    seed: int
    runs: pd.DataFrame
    percentiles: pd.DataFrame
    band: pd.DataFrame
    actual: Mapping[str, float]
    actual_path: pd.Series
    bust: float | None
    goal: float | None
    bust_probability: float
    goal_probability: float


def _whole(value: object, name: str, least: int) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < least:
        raise ConfigError(f"{name} is a whole number, {least} or more, got {value!r}")
    return int(value)


def _threshold(value: object, name: str, low: float, high: float) -> float | None:
    if value is None:
        return None
    number = float(value) if isinstance(value, numbers.Real) else math.nan
    if not low <= number <= high:
        raise ConfigError(f"{name} is a fraction from {low} to {high}, got {value!r}")
    return number


def _closed_pnl_by_exit(trades: pd.DataFrame) -> np.ndarray:
    """Net P&L of closed trades in exit order (then entry order, then input order)."""
    if trades.empty:
        return np.empty(0)
    closed = trades.loc[~pd.isna(trades["exit_time"])]
    ordered = closed.sort_values(["exit_time", "entry_time"], kind="stable")
    return ordered["net_pnl"].to_numpy(dtype=float)


def _trade_path(pnl: np.ndarray, capital: float) -> np.ndarray:
    """Returns of booking `pnl` one after another on `capital`: P&L over the equity
    before it, up to the trade that loses the capital."""
    before = capital + np.concatenate(([0.0], np.cumsum(pnl)[:-1]))
    alive = np.flatnonzero(before <= 0)
    end = alive[0] if len(alive) else len(pnl)
    return pnl[:end] / before[:end]


def monte_carlo(
    performance: Performance,
    *,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    resample: str = "sessions",
    bust: float | None = None,
    goal: float | None = None,
) -> MonteCarlo:
    """Implementation of `openalgo.backtest.metrics.monte_carlo` (documented there)."""
    simulations = _whole(simulations, "simulations", 1)
    seed = _whole(seed, "seed", 0)
    if resample not in RESAMPLE:
        raise ConfigError(f"resample is one of {RESAMPLE}, got {resample!r}")
    bust = _threshold(bust, "bust", -1.0, 0.0)
    goal = _threshold(goal, "goal", -1.0, math.inf)
    m = performance.metrics
    elapsed = int((m["end"] - m["start"]).value)
    rate, capital = float(m["risk_free_rate"]), float(m["start_equity"])
    if resample == "sessions":
        own = performance.session_returns.to_numpy(dtype=float)
        per_year = float(m["sessions_per_year"])
        spec = (simulations, seed, per_year, rate, elapsed)
        out = call("bootstrap_sessions", own, spec)
    else:
        pnl = _closed_pnl_by_exit(performance.trades)
        years = elapsed / NANOS_PER_DAY / native.METRICS_DAYS_PER_YEAR
        if years <= 0:
            raise ConfigError("the run has no elapsed time over which to count trades a year")
        per_year = len(pnl) / years
        own = _trade_path(pnl, capital)
        spec = (simulations, seed, per_year, rate, elapsed)
        out = call("bootstrap_trades", pnl, capital, spec)
    return _study(out, resample, simulations, seed, bust, goal, own, (per_year, rate, elapsed))


def _study(
    out: dict,
    resample: str,
    simulations: int,
    seed: int,
    bust: float | None,
    goal: float | None,
    own: np.ndarray,
    annual: tuple[float, float, int],
) -> MonteCarlo:
    runs = pd.DataFrame({name: out[name] for name in FIGURES})
    runs.index.name = "simulation"
    levels = np.array(PERCENTILE_LEVELS)
    table = {name: call("percentiles", runs[name].to_numpy(), levels) for name in FIGURES}
    percentiles = pd.DataFrame(table, index=pd.Index(levels, name="percentile"))
    band = pd.DataFrame(
        out["band"],
        index=pd.Index(out["band_steps"], name="draws"),
        columns=list(out["band_probabilities"]),
    )
    actual = call("path", own, *annual)
    actual = {"final_return": actual["total_return"], **{k: actual[k] for k in FIGURES[1:]}}
    # The sequence ends at its ruin (if any), where the wealth is 0.
    wealth = np.maximum(np.cumprod(1.0 + own), 0.0)
    actual_path = pd.Series(
        np.concatenate(([0.0], wealth - 1.0)),
        index=pd.RangeIndex(len(own) + 1, name="draws"),
        name="cumulative_return",
    )
    final, deepest = runs["final_return"].to_numpy(), runs["max_drawdown"].to_numpy()
    return MonteCarlo(
        resample=resample,
        simulations=simulations,
        seed=seed,
        runs=runs,
        percentiles=percentiles,
        band=band,
        actual=MappingProxyType(actual),
        actual_path=actual_path,
        bust=bust,
        goal=goal,
        bust_probability=math.nan if bust is None else float(np.mean(deepest <= bust)),
        goal_probability=math.nan if goal is None else float(np.mean(final >= goal)),
    )


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SIMULATIONS",
    "FIGURES",
    "PERCENTILE_LEVELS",
    "RESAMPLE",
    "MonteCarlo",
    "monte_carlo",
]
