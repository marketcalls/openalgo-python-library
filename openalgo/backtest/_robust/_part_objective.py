"""The run's objective measured on a part of its trades (random out of sample, design 17
section 2.2), where a part is a set of whole sessions and the closed trades entered in
them.

A part's session P&L is the net P&L of the trades entered in each of its sessions (0
for a session with none), in rupees; over the capital it is the part's session return.
The objectives a part can be measured by, each annualised where its size matters so a
part of 30 percent of the sessions compares with one of 70 percent:

    sharpe, sortino   the Sharpe and Sortino ratios of the part's session returns,
                      annualised by the run's sessions per year with its risk-free rate:
                      the kernel's definitions (`oa_bt::metrics::ratios`; mean excess
                      return over the sample deviation, or over the downside deviation)
                      with every sum exactly rounded (`math.fsum`), so a part that makes
                      exactly nothing has a ratio of exactly 0 and the sign of a ratio
                      is never a rounding residue
    net_profit        the part's net P&L per year, rupees: its total over its sessions
                      times the sessions per year
    expectancy        the mean net P&L of its trades, rupees
    profit_factor     the sum of its winning trades' net P&L over the magnitude of the
                      sum of its losing ones (infinite with wins and no loss)
    win_rate          its winning trades over its trades

A part with too little to define the figure (no trade, one session) has NaN. Any other
objective, a function of a whole `Result` among them, cannot be measured on a part.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._trades import TradeRun

#: The objectives a part of the trades can be measured by (module docstring).
MEASURED = ("sharpe", "sortino", "net_profit", "expectancy", "profit_factor", "win_rate")


@dataclass(frozen=True)
class Part:
    """A part of a run: the P&L of each of its sessions and of each of its trades,
    rupees."""

    session_pnl: np.ndarray
    trade_pnl: np.ndarray


def _sharpe(excess: np.ndarray, mean: float) -> float:
    """Mean excess return over its sample deviation (divisor n - 1), per session; NaN
    when the excess returns do not vary."""
    if (excess == excess[0]).all():
        return math.nan
    deviation = math.sqrt(math.fsum((excess - mean) ** 2) / (len(excess) - 1))
    return mean / deviation


def _sortino(excess: np.ndarray, mean: float) -> float:
    """Mean excess return over the downside deviation (the root mean square of the
    excess returns below zero), per session; infinite with no downside and a positive
    mean, NaN with neither."""
    downside = math.sqrt(math.fsum(np.minimum(excess, 0.0) ** 2) / len(excess))
    if downside > 0:
        return mean / downside
    return math.inf if mean > 0 else math.nan


def _ratio(name: str, trades: TradeRun) -> Callable[[Part], float]:
    per_year = trades.sessions_per_year
    # The annual risk-free rate as a rate per session, as the kernel converts it.
    per_session = (1.0 + trades.risk_free_rate) ** (1.0 / per_year) - 1.0
    ratio = _sharpe if name == "sharpe" else _sortino

    def measure(part: Part) -> float:
        if len(part.session_pnl) < 2:
            return math.nan
        excess = part.session_pnl / trades.capital - per_session
        mean = math.fsum(excess) / len(excess)
        return ratio(excess, mean) * math.sqrt(per_year)

    return measure


def _net_profit(trades: TradeRun) -> Callable[[Part], float]:
    def measure(part: Part) -> float:
        if not len(part.session_pnl):
            return math.nan
        return math.fsum(part.session_pnl) / len(part.session_pnl) * trades.sessions_per_year

    return measure


def _expectancy(part: Part) -> float:
    return math.fsum(part.trade_pnl) / len(part.trade_pnl) if len(part.trade_pnl) else math.nan


def _profit_factor(part: Part) -> float:
    wins = math.fsum(part.trade_pnl[part.trade_pnl > 0])
    losses = -math.fsum(part.trade_pnl[part.trade_pnl < 0])
    if losses > 0:
        return wins / losses
    return math.inf if wins > 0 else math.nan


def _win_rate(part: Part) -> float:
    return float(np.mean(part.trade_pnl > 0)) if len(part.trade_pnl) else math.nan


def part_measure(objective: Any, trades: TradeRun) -> Callable[[Part], float] | None:
    """How `objective` is measured on a part of `trades`, or None when it cannot be."""
    if not isinstance(objective, str) or objective not in MEASURED:
        return None
    if objective in ("sharpe", "sortino"):
        return _ratio(objective, trades)
    if objective == "net_profit":
        return _net_profit(trades)
    return {"expectancy": _expectancy, "profit_factor": _profit_factor, "win_rate": _win_rate}[
        objective
    ]


def objective_name(objective: Any) -> str:
    """The objective's name as a report shows it: a metric's key, or a function's
    name."""
    return objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")


__all__ = ["MEASURED", "Part", "objective_name", "part_measure"]
