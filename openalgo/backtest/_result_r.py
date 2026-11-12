"""R-multiples of a run (12-position-sizing 6, FR-SIZE-09).

A closed trade's R is its net P&L (after charges) over its initial risk: what its
opening fill would have lost at the stop its entry set (units times the distance from
the fill price to that stop, times the multiplier), recorded by the kernel when the
position opened (`rust/oa_bt/src/engine/initial_risk.rs`). A later add, trail or
breakeven move does not change it. A trade opened with no stop (none given, only a
trail or breakeven that arms later, or the next contract of a futures roll) has no
initial risk and no R: NaN, and it is left out of the summary.

The summary describes the distribution of R over the trades that have one: their count,
mean (the expectancy in R), median, sample standard deviation, best, worst and total, the
share above zero, and the distribution in one-R bins named by their lower edge (a trade
of R 2.4 counts in bin 2, one of R -0.5 in bin -1).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# The statistics of the summary, in order.
STATISTICS = ("mean_r", "median_r", "std_r", "best_r", "worst_r", "total_r", "positive_share")


def r_values(net_pnl: Any, initial_risk: Any) -> np.ndarray:
    """Each trade's R: net P&L over initial risk (rupees both), NaN where the initial
    risk is missing or not positive."""
    pnl = pd.to_numeric(pd.Series(net_pnl, dtype=object), errors="coerce").to_numpy(float)
    risk = pd.to_numeric(pd.Series(initial_risk, dtype=object), errors="coerce").to_numpy(float)
    usable = np.isfinite(risk) & (risk > 0)
    return np.divide(pnl, risk, out=np.full(len(pnl), math.nan), where=usable)


def _statistics(known: np.ndarray) -> dict[str, float]:
    """The statistics of the R values `known`; NaN for none (and a standard deviation
    of one value)."""
    if not len(known):
        return dict.fromkeys(STATISTICS, math.nan)
    return {
        "mean_r": float(np.mean(known)),
        "median_r": float(np.median(known)),
        "std_r": float(np.std(known, ddof=1)) if len(known) > 1 else math.nan,
        "best_r": float(np.max(known)),
        "worst_r": float(np.min(known)),
        "total_r": float(np.sum(known)),
        "positive_share": float(np.mean(known > 0)),
    }


def r_summary(r: Any) -> dict[str, Any]:
    """The summary of the R of every closed trade (see the module docstring): `trades`,
    `trades_with_risk`, the `STATISTICS` (`positive_share` a fraction) and
    `distribution` ({lower edge of a one-R bin: trades})."""
    every = np.asarray(r, dtype=float)
    known = every[np.isfinite(every)]
    edges, counts = np.unique(np.floor(known).astype(int), return_counts=True)
    return {
        "trades": int(len(every)),
        "trades_with_risk": int(len(known)),
        **_statistics(known),
        "distribution": {int(edge): int(count) for edge, count in zip(edges, counts, strict=True)},
    }


__all__ = ["STATISTICS", "r_summary", "r_values"]
