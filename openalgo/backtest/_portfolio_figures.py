"""The account's figures of a portfolio run: its performance (the metrics package over the
account's equity and every strategy's trades) and each strategy's contribution to it.

The account's equity at a round is `account_capital - sum(capital) + sum(strategy
equity)`, each strategy's equity at its last figure, so the strategies' net P&L adds up to
the account's. A strategy's equity at an instant is its last equity point at or before it,
its capital before its first. Amounts are rupees.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._portfolio_tables import STRATEGY
from ._reject_reasons import ACCOUNT_LOSS_GATE, ACCOUNT_MARGIN
from .metrics import analyse
from .report._performance import Performance
from .result import Result

PERCENT = 100.0
#: The refusals that count as orders the account refused a strategy (an order a breach
#: cancelled was working, not refused).
REFUSALS = (ACCOUNT_MARGIN, ACCOUNT_LOSS_GATE)
#: The columns of `contribution`, in order.
CONTRIBUTION_COLUMNS = (
    "net_pnl_rupees",
    "share_pct",
    "drawdown_pnl_rupees",
    "peak_margin_rupees",
    "orders_refused",
)


def account_performance(
    equity: pd.Series, funds: float, results: Mapping[str, Result], benchmark: Any
) -> Performance:
    """Every metric of the account (`metrics.analyse`): its equity in rupees by IST time,
    opened with `funds` at the earliest open of a strategy's first slice, with every
    strategy's closed and open trades (a `strategy` column naming each), annualised by the
    calendars of the strategies' exchanges, and `benchmark` (closes, or None)."""
    tables = [(name, result._metrics_trades()) for name, result in results.items()]
    tables = [(name, table) for name, table in tables if table is not None]
    trades = None
    if tables:
        trades = pd.concat(
            [table.assign(**{STRATEGY: name}) for name, table in tables], ignore_index=True
        )
    openings = [result._opening_time() for result in results.values()]
    known = [opening for opening in openings if opening is not None]
    exchanges = list(
        dict.fromkeys(result.manifest["arguments"]["exchange"] for result in results.values())
    )
    return analyse(
        equity.rename("equity"),
        exchanges=exchanges,
        trades=trades,
        benchmark=benchmark,
        opening_wealth=funds,
        opening_time=min(known) if known else None,
    )


def _as_of(series: pd.Series, instant: pd.Timestamp | None, before: float) -> float:
    """`series` at its last point at or before `instant` (`before` when there is none, or
    for `instant` None: the opening)."""
    if instant is None or not len(series) or instant < series.index[0]:
        return before
    return float(series.asof(instant))


def _margin_at(result: Result, instant: pd.Timestamp | None) -> float:
    """A strategy's margin in rupees at the account's round `instant`, as the account
    counted it. The account takes a strategy's figures after its fills at a slice's open
    (18-multi-strategy 2.3), while its equity rows are its slice closes: the margin is
    that of its last slice close at or before `instant`, or, when it filled after that
    close and by `instant` (at the open of a longer slice still open at `instant`, as a
    daily strategy's beside minute ones), that of the close of the slice it filled in,
    which counts a fill at that close as well. 0 before it has a figure, and for
    `instant` None."""
    margins = result.equity["margin_rupees"]
    if instant is None or not len(margins):
        return 0.0
    closed = int(margins.index.searchsorted(instant, side="right"))
    fills = result.fills["time"]
    since = fills <= instant
    if closed:
        since &= fills > margins.index[closed - 1]
    if since.any() and closed < len(margins):
        return float(margins.iloc[closed])
    return _as_of(margins, instant, 0.0)


def worst_drawdown(equity: pd.Series, funds: float) -> tuple[pd.Timestamp | None, Any]:
    """(the instant the account's maximum drawdown started from, None for the opening;
    the instant of its trough) on `equity` opened with `funds`; (None, None) when the
    account never fell below a peak. The trough is the first lowest point against the
    running peak, and the start the last point at that peak before it."""
    values = equity.to_numpy(dtype=float)
    if not len(values):
        return None, None
    peaks = np.maximum.accumulate(np.concatenate(([funds], values)))[1:]
    falls = values / peaks - 1.0
    trough = int(np.argmin(falls))
    if not falls[trough] < 0.0:
        return None, None
    at_peak = np.flatnonzero(values[: trough + 1] == peaks[trough])
    start = equity.index[at_peak[-1]] if len(at_peak) else None
    return start, equity.index[trough]


def contribution(
    account: pd.DataFrame,
    funds: float,
    results: Mapping[str, Result],
    refused: pd.DataFrame,
) -> pd.DataFrame:
    """Each strategy's part in the account, indexed by strategy name (call order):
    `net_pnl_rupees` (its final equity less its capital), `share_pct` (that in percent of
    the account's net P&L; NaN when the account's is 0), `drawdown_pnl_rupees` (its P&L
    from the start to the trough of the account's maximum drawdown; NaN when the account
    never fell below a peak), `peak_margin_rupees` (its margin when the account's margin
    was at its highest, first such round, as `_margin_at` reads it) and `orders_refused`
    (its orders the account refused, `account_margin` or `account_loss_gate`). `account`
    is `PortfolioResult.account`."""
    equity = account["equity_rupees"]
    total = float(equity.iloc[-1]) - funds if len(equity) else 0.0
    start, trough = worst_drawdown(equity, funds)
    margins = account["margin_rupees"].to_numpy()
    busiest = account.index[int(np.argmax(margins))] if len(margins) else None
    rows = []
    for name, result in results.items():
        capital = float(result.manifest["arguments"]["capital"])
        own = result.equity["equity_rupees"]
        net = (float(own.iloc[-1]) if len(own) else capital) - capital
        inside = np.nan
        if trough is not None:
            inside = _as_of(own, trough, capital) - _as_of(own, start, capital)
        mine = refused[(refused[STRATEGY] == name) & refused["reason"].isin(REFUSALS)]
        rows.append(
            {
                "net_pnl_rupees": net,
                # + 0.0: no P&L is a share of 0, not -0, of an account that lost.
                "share_pct": net / total * PERCENT + 0.0 if total else np.nan,
                "drawdown_pnl_rupees": inside,
                "peak_margin_rupees": _margin_at(result, busiest),
                "orders_refused": len(mine),
            }
        )
    index = pd.Index(list(results), name=STRATEGY)
    return pd.DataFrame(rows, index=index, columns=list(CONTRIBUTION_COLUMNS))


__all__ = [
    "CONTRIBUTION_COLUMNS",
    "REFUSALS",
    "account_performance",
    "contribution",
    "worst_drawdown",
]
