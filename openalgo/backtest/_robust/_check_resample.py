"""Trade resample (design 17, section 2.11): how wide is the range of outcomes?

`runs` paths, each drawing as many closed trades as the run had, with replacement, and
booking their net P&L one after another on the start capital: the Monte Carlo of
`metrics.monte_carlo(resample="trades")`, with the check's seed. Its runs give the
summary block (`_trade_common.path_figures`): the share of paths profitable, the ruin
odds (paths whose maximum drawdown reaches `ruin`), the average profit and drawdown,
their 95 percent intervals and the return over drawdown. Pass: at least `pass_share` of
the paths profitable and ruin odds at most `max_ruin`.

The fan's bands are the Monte Carlo's own; its thin paths are its first runs, drawn
again the way the kernel draws them (`_paths.resampled_trades`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..report._montecarlo import monte_carlo
from . import _words as words
from ._blocks import FanBlock
from ._paths import BAND_LEVELS, SAMPLED_PATHS, Fan, own_path, resampled_trades, walk
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUIN, RUNS, path_figures, path_tiles, too_few
from ._trades import TradeRun
from ._verdict import FAILED, PASSED, Outcome

SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    RUIN,
    Setting(
        "pass_share",
        0.95,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of paths that must end profitable",
    ),
    Setting(
        "max_ruin",
        0.05,
        between(0.0, 1.0, low_open=False, high_open=True),
        "a fraction from 0 and below 1",
        "the most ruin odds allowed",
    ),
)


def rule(settings: Mapping[str, Any]) -> str:
    """The pass rule, in words."""
    return (
        f"at least {words.percent(settings['pass_share'], 0)} of paths profitable and at "
        f"most {words.percent(settings['max_ruin'], 0)} reaching a "
        f"{words.percent(settings['ruin'], 0)} drawdown"
    )


def _fan(trades: TradeRun, band: Any, seed: int, runs: int) -> Fan:
    """The Monte Carlo's bands, its first runs thin and the run's own path."""
    steps = band.index.to_numpy(dtype=np.int64)
    rows = resampled_trades(trades.pnl, trades.capital, seed, min(SAMPLED_PATHS, runs))
    samples = walk(rows, trades.capital).wealth[:, steps] - 1.0
    real = own_path(trades.pnl, trades.capital, steps)
    return Fan(steps, band.loc[:, list(BAND_LEVELS)].to_numpy(dtype=float), samples, real, runs)


def run(
    trades: TradeRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring; `objective` is not used."""
    text = rule(settings)
    short = too_few(trades, settings["min_trades"], text)
    if short is not None:
        return short
    study = monte_carlo(trades.performance, simulations=runs, seed=seed, resample="trades")
    final = study.runs["final_return"].to_numpy(dtype=float)
    drawdown = study.runs["max_drawdown"].to_numpy(dtype=float)
    summary = path_figures(final, drawdown, trades.capital, settings["ruin"])
    profitable, ruin_odds = summary["profitable_share"], summary["ruin_odds"]
    passed = profitable >= settings["pass_share"] and ruin_odds <= settings["max_ruin"]
    figure = (
        f"{words.percent(profitable)} of paths profitable, ruin odds {words.percent(ruin_odds)}"
    )
    note = (
        f"{words.whole(runs, 'path')} of {words.whole(trades.closed, 'trade')} drawn with "
        f"replacement from the closed trades and booked on {words.rupees(trades.capital)} "
        f"(metrics.monte_carlo with resample='trades' and seed {seed})."
    )
    table = study.runs.rename_axis("run").copy()
    table["profit_rupees"] = final * trades.capital
    table["ruined"] = drawdown <= -settings["ruin"]
    blocks = (
        (
            FanBlock(
                "Resampled equity",
                f"{words.whole(runs, 'path')}",
                _fan(trades, study.band, seed, runs),
            ),
        ),
        (path_tiles("Summary", "Over the paths", summary, settings["ruin"]),),
    )
    return Outcome(PASSED if passed else FAILED, figure, text, note, table, summary, blocks)


__all__ = ["SETTINGS", "rule", "run"]
