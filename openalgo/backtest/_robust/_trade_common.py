"""What the trade-list checks share: the settings every one of them takes, the trade
count needed to judge, and the summary block of resampled paths (trade resample and
variance, design 17 sections 2.11 and 2.13).

The summary block of a set of paths (`_paths.walk`, fractions of the start capital):

    profitable_share        paths that end above the capital
    ruin_odds               paths whose maximum drawdown reaches `ruin` (a fraction, 0.5
                            is a fall of half from a peak)
    average_profit_rupees   the mean of the paths' final P&L
    average_drawdown        the mean of their maximum drawdowns (zero or negative)
    profit_low_rupees, profit_high_rupees
                            the 95 percent interval of the final P&L: its 2.5th and
                            97.5th percentiles
    drawdown_low, drawdown_high
                            the same interval of the maximum drawdown (low is deeper)
    return_over_drawdown    the mean final return over the magnitude of the mean
                            maximum drawdown (infinite when no path ever falls and the
                            mean return is positive)

Percentiles interpolate linearly between order statistics, as every percentile of the
metrics package does.
"""

from __future__ import annotations

import math

import numpy as np

from ..report._kernel import call
from . import _words as words
from ._blocks import Tile, TilesBlock
from ._setting import Setting, between, whole
from ._trades import TradeRun
from ._verdict import Outcome, not_judged

#: Closed trades below which a trade-list check is not judged (design 17, section 4).
MIN_TRADES = 30
#: The 95 percent interval: its lower and upper percentile levels.
INTERVAL = (0.025, 0.975)

RUNS = Setting(
    "runs",
    None,
    whole(1),
    "a whole number, 1 or more, or None",
    "alternative runs; None takes the report's runs",
)
ENOUGH_TRADES = Setting(
    "min_trades",
    MIN_TRADES,
    whole(2),
    "a whole number, 2 or more",
    "closed trades needed to judge; fewer is too few trades to judge",
)
RUIN = Setting(
    "ruin",
    0.5,
    between(0.0, 1.0, low_open=True, high_open=False),
    "a fraction above 0 and at most 1",
    "the maximum drawdown, a fraction of a peak, that counts a path as ruined",
)


def too_few(trades: TradeRun, needed: int, rule: str) -> Outcome | None:
    """The "not judged" outcome of a run with fewer than `needed` closed trades, else
    None."""
    if trades.closed >= needed:
        return None
    why = f"Too few trades to judge: {words.whole(trades.closed, 'closed trade')}, {needed} needed."
    return not_judged(rule, why)


def path_figures(
    final_return: np.ndarray, max_drawdown: np.ndarray, capital: float, ruin: float
) -> dict[str, float]:
    """The summary block of paths with these final returns and maximum drawdowns
    (module docstring)."""
    profit_low, profit_high = call("percentiles", final_return, np.array(INTERVAL))
    drawdown_low, drawdown_high = call("percentiles", max_drawdown, np.array(INTERVAL))
    mean_return, mean_drawdown = float(np.mean(final_return)), float(np.mean(max_drawdown))
    if mean_drawdown < 0:
        ratio = mean_return / -mean_drawdown
    else:
        ratio = math.inf if mean_return > 0 else math.nan
    return {
        "runs": float(len(final_return)),
        "profitable_share": float(np.mean(final_return > 0)),
        "ruin_odds": float(np.mean(max_drawdown <= -ruin)),
        "average_profit_rupees": mean_return * capital,
        "average_drawdown": mean_drawdown,
        "profit_low_rupees": float(profit_low) * capital,
        "profit_high_rupees": float(profit_high) * capital,
        "drawdown_low": float(drawdown_low),
        "drawdown_high": float(drawdown_high),
        "return_over_drawdown": ratio,
    }


def _tone(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


def path_tiles(title: str, subtitle: str, figures: dict[str, float], ruin: float) -> TilesBlock:
    """The summary block as tiles."""
    profit = figures["average_profit_rupees"]
    low, high = figures["profit_low_rupees"], figures["profit_high_rupees"]
    tiles = (
        Tile("Profitable", words.percent(figures["profitable_share"]), "of the paths"),
        Tile("Ruin odds", words.percent(figures["ruin_odds"]), f"drawdown {words.percent(-ruin)}"),
        Tile("Average profit", words.rupees(profit), "final P&L", _tone(profit)),
        Tile("Profit, low", words.rupees(low), "2.5th percentile", _tone(low)),
        Tile("Profit, high", words.rupees(high), "97.5th percentile", _tone(high)),
        Tile("Return / drawdown", words.number(figures["return_over_drawdown"]), "averages"),
        Tile("Average drawdown", words.percent(figures["average_drawdown"]), "maximum", "neg"),
        Tile("Drawdown, deep", words.percent(figures["drawdown_low"]), "2.5th percentile", "neg"),
        Tile("Drawdown, mild", words.percent(figures["drawdown_high"]), "97.5th percentile", "neg"),
    )
    return TilesBlock(title, subtitle, tiles)


__all__ = [
    "ENOUGH_TRADES",
    "INTERVAL",
    "MIN_TRADES",
    "RUIN",
    "RUNS",
    "path_figures",
    "path_tiles",
    "too_few",
]
