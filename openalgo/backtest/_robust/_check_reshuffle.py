"""Trade reshuffle (design 17, section 2.10): how deep can the drawdown get in another
order?

The closed trades' net P&L, reordered at random without replacement `runs` times and
booked one after another on the start capital (`_paths`), so every order ends at the
same P&L unless it loses the whole capital first. Figures: the maximum drawdown that 50,
95 and 99 percent of the orders stay within (the 50th, 5th and 1st percentiles of their
maximum drawdowns, which are zero or negative), beside the run's own order's. Pass:
informational unless `max_drawdown` (a fraction of a peak, 0.25 is 25 percent) is given;
then the drawdown `level` of the orders stay within must be no deeper than it.

Draws: order r sorts the trades by draws r x trades onwards of the check's stream
(`_random.orders`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..report._kernel import call
from . import _random
from . import _words as words
from ._blocks import LIMIT, REAL, FanBlock, HistogramBlock, Mark, Tile, TilesBlock
from ._paths import DRAWS_PER_CHUNK, fan, fan_steps, measure, own_path, walk
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES, RUNS, too_few
from ._trades import TradeRun
from ._verdict import FAILED, INFORMATIONAL, PASSED, Outcome

#: The shares of orders whose drawdown the figures state (design 17, section 2.10).
SHARES = (0.50, 0.95, 0.99)

SETTINGS = (
    RUNS,
    ENOUGH_TRADES,
    Setting(
        "max_drawdown",
        None,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1, or None",
        "the deepest drawdown allowed, a fraction of a peak; None judges nothing",
    ),
    Setting(
        "level",
        0.95,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "the share of orders whose drawdown the limit is held against",
    ),
)


def rule(settings: Mapping[str, Any]) -> str:
    """The pass rule, in words."""
    limit = settings["max_drawdown"]
    if limit is None:
        return "informational: give reshuffle={'max_drawdown': 0.25} to judge"
    return (
        f"the drawdown {words.percent(settings['level'], 0)} of orders stay within is no "
        f"deeper than {words.percent(-limit)}"
    )


def within(drawdowns: np.ndarray, share: float) -> float:
    """The maximum drawdown `share` of the orders stay within: its (1 - share)
    percentile, drawdowns being zero or negative."""
    return float(call("percentiles", drawdowns, np.array([1.0 - share]))[0])


def run(
    trades: TradeRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring; `objective` is not used."""
    text = rule(settings)
    short = too_few(trades, settings["min_trades"], text)
    if short is not None:
        return short
    count, steps = trades.closed, fan_steps(trades.closed)
    paths = measure(
        lambda first, size: trades.pnl[_random.orders(seed, first, size, count)],
        _random.run_chunks(runs, count, DRAWS_PER_CHUNK),
        trades.capital,
        steps,
    )
    real = own_path(trades.pnl, trades.capital, steps)
    real_drawdown = float(walk(trades.pnl[None, :], trades.capital).max_drawdown[0])
    summary = {
        f"drawdown_{round(share * 100)}": within(paths.max_drawdown, share) for share in SHARES
    }
    level, limit = settings["level"], settings["max_drawdown"]
    at_level = within(paths.max_drawdown, level)
    summary |= {
        "runs": float(runs),
        "drawdown_at_level": at_level,
        "real_drawdown": real_drawdown,
        "final_pnl_rupees": math.fsum(trades.pnl),
        "ruined_share": float(paths.ruined.mean()),
    }
    verdict = INFORMATIONAL if limit is None else PASSED if at_level >= -limit else FAILED
    figure = (
        f"{words.percent(level, 0)} of orders stay within {words.percent(at_level)} "
        f"(the run's order {words.percent(real_drawdown)})"
    )
    note = (
        f"{words.whole(runs, 'order')} of the {trades.closed} closed trades, drawn without "
        f"replacement and booked on {words.rupees(trades.capital)}; every order ends at "
        f"{words.rupees(math.fsum(trades.pnl))} of net P&L"
    )
    ruined = int(paths.ruined.sum())
    if ruined:
        note += f", except {words.whole(ruined, 'order')} that lose the whole capital first"
    table = pd.DataFrame(
        {
            "final_return": paths.final_return,
            "max_drawdown": paths.max_drawdown,
            "ruined": paths.ruined,
        }
    )
    table.index.name = "run"
    marks = (Mark(real_drawdown, "the run's order", REAL),)
    if limit is not None:
        marks += (Mark(-limit, "limit", LIMIT),)
    blocks = _blocks(fan(paths, steps, real), paths.max_drawdown, marks, summary, runs)
    return Outcome(verdict, figure, text, note + ".", table, summary, blocks)


def _blocks(shown_fan, drawdowns: np.ndarray, marks: tuple, summary: dict, runs: int) -> tuple:
    tiles = tuple(
        Tile(
            f"{round(share * 100)}% within",
            words.percent(summary[f"drawdown_{round(share * 100)}"]),
            "of the orders",
            "neg",
        )
        for share in SHARES
    ) + (
        Tile("Run's order", words.percent(summary["real_drawdown"]), "maximum drawdown", "neg"),
        Tile("Final P&L", words.rupees(summary["final_pnl_rupees"]), "every order"),
    )
    return (
        (
            FanBlock(
                "Reshuffled equity", f"{words.whole(runs, 'order')} of the same trades", shown_fan
            ),
            HistogramBlock("Maximum drawdown", "Of every order", drawdowns, True, marks),
        ),
        (TilesBlock("Summary", "Maximum drawdown over the orders", tiles),),
    )


__all__ = ["SETTINGS", "SHARES", "rule", "run", "within"]
