"""Out-of-sample runs stitched into one equity curve (design 17, sections 2.1 and 2.3).

Each window is an ordinary backtest that opens with the run's capital. The stitched curve
books their P&L one after another on that capital, at fixed rupee size, as the runs were
sized: a window's equity at each session close moves by its own P&L from the capital,
on top of the P&L of the windows before it. A position open at a window's end is marked
there, as any backtest marks it, and is not carried into the next window, which starts
flat. The stitched curve is measured as any run is (`metrics.analyse`, with the run's
capital as its opening wealth and the first window's opening as its opening time), at
session closes, with every window's closed trades.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..metrics import analyse
from ..report._performance import Performance
from ._rerun_one import RunDetail


def stitched_equity(details: Sequence[RunDetail], capital: float) -> pd.Series:
    """The stitched equity at each session close of `details`' windows, in order, in
    rupees (module docstring)."""
    pieces, carried = [], 0.0
    for detail in details:
        pieces.append(detail.session_equity + carried)
        carried += float(detail.session_equity.iloc[-1]) - capital
    return pd.concat(pieces)


def stitched(details: Sequence[RunDetail], capital: float, exchange: str) -> Performance:
    """The performance of the stitched curve of `details` (module docstring), on
    `exchange`'s calendar. Raises ConfigError when the windows overlap."""
    trades = [detail.trades for detail in details if len(detail.trades)]
    return analyse(
        stitched_equity(details, capital),
        exchanges=exchange,
        trades=pd.concat(trades, ignore_index=True) if trades else None,
        opening_wealth=capital,
        opening_time=details[0].opening,
    )


__all__ = ["stitched", "stitched_equity"]
