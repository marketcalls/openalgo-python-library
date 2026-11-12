"""The R-multiples section of the full report (12-position-sizing 6, FR-SIZE-09): the
summary of the closed trades' R and its histogram.

It is shown when the trades table has an `r_multiple` column (a backtest result's
trades do) and at least one closed trade has an initial risk; the definitions are those
of `openalgo.backtest._result_r`.
"""

import numpy as np
import pandas as pd

from .._result_r import r_summary
from . import _charts as charts
from . import _format as fmt
from . import _sections as sections
from ._performance import Performance

#: The bin width of a histogram whose trades all have the same R: one R.
R_HISTOGRAM_FLOOR = 1.0


def _r(value: float) -> str:
    return fmt.MISSING if fmt.is_missing(value) else f"{fmt.number(value)}R"


def _summary_table(summary: dict) -> str:
    rows = (
        ("Trades with an initial risk", f"{summary['trades_with_risk']} of {summary['trades']}"),
        ("Mean R (expectancy)", _r(summary["mean_r"])),
        ("Median R", _r(summary["median_r"])),
        ("Standard deviation of R", _r(summary["std_r"])),
        ("Best R", _r(summary["best_r"])),
        ("Worst R", _r(summary["worst_r"])),
        ("Total R", _r(summary["total_r"])),
        ("Trades above 0R", fmt.percent(summary["positive_share"])),
    )
    body = "".join(
        f"<tr><th scope='row'>{label}</th><td class='num'>{value}</td></tr>"
        for label, value in rows
    )
    return f"<div class='scroll'><table class='data'><tbody>{body}</tbody></table></div>"


def section(performance: Performance) -> list[str]:
    """The R-multiples section, or none when no closed trade has an initial risk."""
    trades = performance.trades
    if "r_multiple" not in trades.columns:
        return []
    closed = trades.loc[~pd.isna(trades["exit_time"]), "r_multiple"].to_numpy(dtype=float)
    known = closed[np.isfinite(closed)]
    if not len(known):
        return []
    chart = charts.histogram(
        known,
        label="R-multiples of closed trades",
        edge=_r,
        unit="trade(s)",
        floor=R_HISTOGRAM_FLOOR,
    )
    lede = (
        "Each closed trade's net P&amp;L over its initial risk, the loss its opening fill "
        "would take at the stop its entry set. Trades opened without a stop have no R."
    )
    body = _summary_table(r_summary(closed)) + chart
    return [sections.section("r-multiples", "R-multiples", body, lede)]


__all__ = ["R_HISTOGRAM_FLOOR", "section"]
