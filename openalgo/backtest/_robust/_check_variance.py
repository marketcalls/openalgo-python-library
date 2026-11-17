"""Variance (design 17, section 2.13): does it survive a worse win rate?

The closed trades resampled as the trade resample does (`runs` paths of as many trades
as the run had, booked on the start capital), but each drawn trade is a winner (net P&L
above zero) with a set probability, drawn from the winners, else from the others (net
P&L of zero or below). Two sets of paths use the same draws: one at the run's own win
rate, one at a win rate `points` percentage points lower (not below zero), so the
difference between them is the lower win rate alone. Figures: the summary block of each
(`_trade_common.path_figures`). Pass: at the lower rate, at least `pass_share` of the
paths profitable. A run whose every trade won has no other trade to draw and is not
judged.

Draws: path r reads 2 x trades draws from r x 2 x trades onwards of the check's stream;
the even ones decide winner or not (a fraction below the rate is a winner), the odd ones
pick the trade within its group.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import _random
from . import _words as words
from ._blocks import FanBlock
from ._paths import DRAWS_PER_CHUNK, PathSet, fan, fan_steps, measure, own_path
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUIN, RUNS, path_figures, path_tiles, too_few
from ._trades import TradeRun
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: Percentage points per unit fraction.
POINTS_PER_UNIT = 100.0

SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    RUIN,
    Setting(
        "points",
        5.0,
        between(0.0, 100.0, low_open=True, high_open=True),
        "a number of percentage points above 0 and below 100",
        "how far the win rate is lowered",
    ),
    Setting(
        "pass_share",
        0.8,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of paths that must end profitable at the lower win rate",
    ),
)


def rule(settings: Mapping[str, Any]) -> str:
    """The pass rule, in words."""
    return (
        f"at least {words.percent(settings['pass_share'], 0)} of paths profitable with the "
        f"win rate {settings['points']:g} points lower"
    )


def _drawn(trades: TradeRun, seed: int, runs: int, rates: tuple[float, ...]) -> list[PathSet]:
    """The paths at each of `rates`, from the same draws."""
    count = trades.closed
    winners, others = trades.pnl[trades.pnl > 0], trades.pnl[trades.pnl <= 0]
    steps = fan_steps(count)
    chunks: list[list[np.ndarray]] = [[] for _ in rates]
    for first, size in _random.run_chunks(runs, 2 * count, DRAWS_PER_CHUNK):
        bits = _random.block(seed, first, size, 2 * count)
        chance = _random.fractions(bits[:, 0::2])
        pick = bits[:, 1::2]
        won = winners[_random.below(pick, len(winners))] if len(winners) else np.zeros(pick.shape)
        other = others[_random.below(pick, len(others))]
        for rows, rate in zip(chunks, rates, strict=True):
            rows.append(np.where(chance < rate, won, other))
    return [measure(rows, trades.capital, steps) for rows in chunks]


def run(
    trades: TradeRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring; `objective` is not used."""
    text = rule(settings)
    short = too_few(trades, settings["min_trades"], text)
    if short is not None:
        return short
    if not (trades.pnl <= 0).any():
        why = "Every closed trade won, so no other trade is left to draw at a lower win rate."
        return not_judged(text, why)
    own_rate = trades.win_rate
    lower_rate = max(own_rate - settings["points"] / POINTS_PER_UNIT, 0.0)
    own, lower = _drawn(trades, seed, runs, (own_rate, lower_rate))
    ruin = settings["ruin"]
    own_figures = path_figures(own.final_return, own.max_drawdown, trades.capital, ruin)
    lower_figures = path_figures(lower.final_return, lower.max_drawdown, trades.capital, ruin)
    summary = {"win_rate": own_rate, "lower_win_rate": lower_rate}
    summary |= {f"own_{key}": value for key, value in own_figures.items()}
    summary |= {f"lower_{key}": value for key, value in lower_figures.items()}
    profitable = lower_figures["profitable_share"]
    verdict = PASSED if profitable >= settings["pass_share"] else FAILED
    figure = (
        f"{words.percent(profitable)} of paths profitable at a {words.percent(lower_rate)} win "
        f"rate ({words.percent(own_figures['profitable_share'])} at the run's "
        f"{words.percent(own_rate)})"
    )
    winners = int((trades.pnl > 0).sum())
    note = (
        f"{words.whole(runs, 'path')} of {trades.closed} trades at each rate; each drawn trade "
        f"is a winner with the rate's probability, drawn from the {winners} winners, else "
        f"from the {trades.closed - winners} other trades (net P&L of zero or below). Both "
        "rates use the same draws."
    )
    table = pd.concat([_rows(own, own_rate, trades), _rows(lower, lower_rate, trades)])
    blocks = _blocks(
        trades, (own, lower), (own_rate, lower_rate), (own_figures, lower_figures), ruin
    )
    return Outcome(verdict, figure, text, note, table, summary, blocks)


def _rows(paths: PathSet, rate: float, trades: TradeRun) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "win_rate": rate,
            "final_return": paths.final_return,
            "max_drawdown": paths.max_drawdown,
            "profit_rupees": paths.final_return * trades.capital,
            "ruined": paths.ruined,
        }
    )
    table.index.name = "run"
    return table


def _blocks(trades: TradeRun, paths: tuple, rates: tuple, figures: tuple, ruin: float) -> tuple:
    steps = fan_steps(trades.closed)
    real = own_path(trades.pnl, trades.capital, steps)
    names = ("At the run's win rate", "At the lower win rate")
    fans = tuple(
        FanBlock(name, f"Win rate {words.percent(rate)}", fan(paths_at, steps, real))
        for name, rate, paths_at in zip(names, rates, paths, strict=True)
    )
    tiles = tuple(
        path_tiles(name, f"Win rate {words.percent(rate)}", block, ruin)
        for name, rate, block in zip(names, rates, figures, strict=True)
    )
    return (fans, tiles)


__all__ = ["SETTINGS", "rule", "run"]
