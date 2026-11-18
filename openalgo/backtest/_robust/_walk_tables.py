"""The tables of a walk-forward (design 17, section 2.3): its folds as a DataFrame
(`bt.walk_forward`'s `WalkForwardResult.folds` and the walk-forward check's table) and as
the report's fold table, the stitched row first, with a small equity line a row.

A fold's figures: its four window dates (the first and last session of each window), the
combination chosen in sample, its in-sample and out-of-sample objective per year
(`_walk.annual`) and their efficiency (out over in, from the objective's break-even),
and its out-of-sample net P&L in rupees, maximum drawdown (a fraction, zero or
negative), P&L over the drawdown in rupees (`max_drawdown_amount`; infinite with no
drawdown and a profit), closed trades and win rate. A fold that chose nothing has NaN
figures.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import _words as words
from ._bar_common import break_even
from ._blocks import SparkTableBlock
from ._search_common import params_text
from ._walk import FoldOutcome, WalkOutcome


def _ratio(value: float, inside: float, level: float | None) -> float:
    if level is None or not inside > level or not math.isfinite(value):
        return math.nan
    return (value - level) / (inside - level)


def _pnl_over_drawdown(net: float, drawdown_rupees: float) -> float:
    """Net P&L over the magnitude of the maximum drawdown in rupees (zero or negative)."""
    if drawdown_rupees < 0:
        return net / -drawdown_rupees
    return math.inf if drawdown_rupees == 0 and net > 0 else math.nan


def _row(fold: FoldOutcome, outcome: WalkOutcome, dates: Sequence[dt.date]) -> dict[str, Any]:
    windows = fold.fold
    row: dict[str, Any] = {
        "fold": windows.number,
        "in_sample_start": dates[windows.in_sample.first],
        "in_sample_end": dates[windows.in_sample.last],
        "out_of_sample_start": dates[windows.out_of_sample.first],
        "out_of_sample_end": dates[windows.out_of_sample.last],
    }
    for name in outcome.grid.names:
        row[name] = None if fold.chosen is None else outcome.grid.combos[fold.chosen][name]
    inside, outside = fold.in_annual(outcome.objective), fold.out_annual(outcome.objective)
    row.update(in_sample=inside, out_of_sample=outside)
    row["efficiency"] = _ratio(outside, inside, break_even(outcome.objective))
    detail = fold.out.detail if fold.traded else None
    figures = detail.figures if detail is not None else {}
    net = fold.out.net_profit if fold.traded else math.nan
    row.update(
        net_profit=net,
        max_drawdown=figures.get("max_drawdown", math.nan),
        pnl_over_drawdown=_pnl_over_drawdown(net, figures.get("max_drawdown_amount", math.nan)),
        trades=figures.get("trades", math.nan),
        win_rate=figures.get("win_rate", math.nan),
    )
    return row


def fold_frame(outcome: WalkOutcome, dates: Sequence[dt.date]) -> pd.DataFrame:
    """A row per fold (module docstring), indexed by fold number."""
    frame = pd.DataFrame([_row(fold, outcome, dates) for fold in outcome.folds])
    return frame.set_index("fold")


def drift_words(outcome: WalkOutcome) -> str:
    """Each parameter's drift, in words: "fast 25.0%, slow 0.0%"."""
    return ", ".join(f"{name} {words.percent(value)}" for name, value in outcome.drift().items())


def _cells(label: str, dates_text: str, chosen: str, figures: dict[str, float]) -> tuple:
    return (
        label,
        dates_text,
        chosen,
        words.rupees(figures["net_profit"]),
        words.percent(figures["max_drawdown"]),
        words.number(figures["pnl_over_drawdown"]),
        words.number(figures["trades"], 0),
        words.percent(figures["win_rate"]),
    )


def fold_table(outcome: WalkOutcome, dates: Sequence[dt.date], capital: float) -> SparkTableBlock:
    """The report's fold table: the stitched row, then a row per fold, each with its
    out-of-sample equity line."""
    frame = fold_frame(outcome, dates)
    rows, sparks = [], []
    performance = outcome.performance
    if performance is not None:
        metrics = performance.metrics
        net = float(metrics["net_profit"])
        stitched = {
            "net_profit": net,
            "max_drawdown": float(metrics["max_drawdown"]),
            "pnl_over_drawdown": _pnl_over_drawdown(net, float(metrics["max_drawdown_amount"])),
            "trades": float(metrics["trades"]),
            "win_rate": float(metrics["win_rate"]),
        }
        span = f"{frame['out_of_sample_start'].iloc[0]} to {frame['out_of_sample_end'].iloc[-1]}"
        rows.append(_cells("Stitched", span, "as each fold chose", stitched))
        sparks.append(performance.equity.to_numpy(dtype=float) - capital)
    for fold in outcome.folds:
        row = frame.loc[fold.fold.number]
        span = f"{row['out_of_sample_start']} to {row['out_of_sample_end']}"
        chosen = "none" if fold.chosen is None else params_text(outcome.grid.swept(fold.chosen))
        rows.append(_cells(f"Fold {fold.fold.number}", span, chosen, row.to_dict()))
        detail = fold.out.detail if fold.traded else None
        sparks.append(
            np.array([]) if detail is None else detail.session_equity.to_numpy() - capital
        )
    columns = (
        "Run",
        "Out of sample",
        "Parameters",
        "Net P&L",
        "Drawdown",
        "P&L / drawdown",
        "Trades",
        "Win rate",
    )
    return SparkTableBlock(
        "By fold",
        "The stitched run first; each fold's out-of-sample equity from zero",
        columns,
        tuple(rows),
        3,
        "Equity",
        tuple(sparks),
    )


__all__ = ["drift_words", "fold_frame", "fold_table"]
