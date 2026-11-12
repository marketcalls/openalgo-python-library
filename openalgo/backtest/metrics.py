"""Performance metrics of a backtest, built in: returns, risk, drawdowns, trades, costs.

    from openalgo.backtest import metrics

    perf = metrics.analyse(equity, exchanges="NSE", trades=trades, risk_free_rate=0.065)
    perf.metrics["sharpe"], perf.metrics["max_drawdown"], perf.monthly_returns

The per-bar and per-trade work runs in the native kernel; this module checks the
inputs, takes sessions per year from the session calendar, and returns a
`Performance`. `openalgo.backtest.report.tearsheet` renders one as an HTML report.

Inputs
    equity: a pandas Series of equity in rupees, indexed by time (a time-zone-aware
        DatetimeIndex in any zone, or int64 UTC nanoseconds), strictly increasing. The
        first point is the starting equity, before any P&L of the run; each later
        point is the marked equity at that instant (bars are stamped at their close).
        At least two points.
    exchanges: the OpenAlgo exchanges the strategy trades ("NSE", or ["NFO", "MCX"]).
        Their union calendar gives the sessions per year.
    trades: a DataFrame, one row per round trip, with the columns
        entry_time, exit_time   times as above; exit_time NaT (or NA) for a trade still
                                open at the end
        entry_price, exit_price rupees per unit; exit_price may be NaN when open
        quantity                units, positive (never lots); price x quantity is the
                                traded value
        side                    "BUY" for a long trade, "SELL" for a short one
        gross_pnl, net_pnl, costs   rupees; net_pnl = gross_pnl - costs to the paisa
        Optional cost columns brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst
        and dp_charge (rupees) are summed into `Performance.cost_breakdown`. Other
        columns (symbol, exit_reason, ...) are kept for the report. Trade statistics
        count closed trades only; an open trade counts for exposure and turnover.
    cash_flows: a Series of deposits (positive) and withdrawals (negative) in rupees,
        indexed by time, each after the first equity point and at or before the last.
        A flow is taken to arrive at the next equity point; it changes the equity but
        is not a return.
    benchmark: a Series of positive index or price levels indexed by time (NIFTY 50
        closes, or a total-return index). It is read as of the start and as of each
        session's last equity point, never back-filled: a session whose previous
        benchmark value is unknown has no benchmark return.
    risk_free_rate: the annual risk-free rate as a fraction (0.065 is 6.5 percent).
        It is converted to the per-session rate (1 + rate) ^ (1 / sessions per year) - 1
        before it is subtracted from session returns.
    calendar: the session calendar; the shared built-in one by default.

Conventions
    Session returns: the IST date of each point after the first is its session date,
    and the return of a session is the time-weighted wealth at its last point over the
    wealth at the previous session's last point (the start for the first), minus 1.
    Annualisation: CAGR uses elapsed calendar time (365.25 days a year). Volatility,
    Sharpe, Sortino and alpha use sessions per year: the sessions of the traded
    exchanges' union calendar in the whole calendar years the run touches, scaled to
    365.25 days, never a fixed 252 or 365. A year with no holiday list is reported in
    `annualisation_template_years` and announced with a CalendarWarning.
    Ruin: once equity before cash flows reaches zero or below, the wealth index is 0
    and later sessions have no return; total return and CAGR are -1.
    Undefined figures are NaN; a positive figure over a zero denominator is infinite.

Several strategies on fixed allocations: each strategy's equity is analysed on its own
allocation; the combined account is `analyse(combine_equity({...}), trades=...)` with
the trades tables concatenated, and `return_correlation` compares the strategies.

Errors: ConfigError for any input that cannot be used, with a message that names the
column and row; CalendarError for an unknown exchange.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from .calendar import Calendar
from .report import _analysis, _combined
from .report._definitions import DEFINITIONS, Definition, as_text
from .report._performance import Performance


def analyse(
    equity: pd.Series,
    *,
    exchanges: str | Sequence[str],
    trades: pd.DataFrame | None = None,
    cash_flows: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    calendar: Calendar | None = None,
) -> Performance:
    """Every metric of one equity series, with its trades, flows and benchmark.

    The inputs are described in the module docstring and every key of the returned
    `Performance.metrics` in `DEFINITIONS`. Raises ConfigError for unusable input
    (naive timestamps, unsorted points, a trade whose net P&L is not gross P&L minus
    costs, a rate written in percent) and CalendarError for an unknown exchange.
    """
    return _analysis.analyse(
        equity,
        exchanges=exchanges,
        trades=trades,
        cash_flows=cash_flows,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        calendar=calendar,
    )


def combine_equity(equities: Mapping[str, pd.Series]) -> pd.Series:
    """The combined equity, in rupees, of strategies run on fixed capital allocations.

    Each strategy trades its own allocation, so the combined account is the sum of
    their equities on the union of their timestamps. A strategy's equity at an instant
    is its last value at or before it, and its first value (its allocation) before its
    first point. `equities` maps a strategy name to an equity Series indexed as for
    `analyse`. Returns a Series indexed by IST time. Raises ConfigError for an empty
    mapping or a malformed series.
    """
    return _combined.combine_equity(equities)


def return_correlation(strategies: Mapping[str, Performance | pd.Series]) -> pd.DataFrame:
    """Pearson correlation matrix of the strategies' session returns.

    `strategies` maps a name to its `Performance` (or to a Series of session returns
    indexed by IST date). Returns are aligned on the union of the session dates, and a
    strategy with no point on a date counts a return of 0 there (its equity did not
    move). A strategy whose returns never vary has NaN in its row and column. Raises
    ConfigError for fewer than two strategies.
    """
    return _combined.return_correlation(strategies)


__doc__ += as_text()

__all__ = [
    "DEFINITIONS",
    "Definition",
    "Performance",
    "analyse",
    "combine_equity",
    "return_correlation",
]
