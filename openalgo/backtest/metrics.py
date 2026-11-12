"""Performance metrics of a backtest, built in: returns, risk, drawdowns, trades, costs,
the distribution and tails of session returns, rolling statistics, a horizon summary
and Monte Carlo resampling.

    from openalgo.backtest import metrics

    perf = metrics.analyse(
        equity, exchanges="NSE", trades=trades, risk_free_rate=0.065, opening_wealth=500_000
    )
    perf.metrics["sharpe"], perf.metrics["max_drawdown"], perf.monthly_returns
    perf.drawdown_details(5), perf.rolling, perf.horizons
    study = metrics.monte_carlo(perf, simulations=1_000, seed=7)

The per-bar and per-trade work runs in the native kernel; this module checks the
inputs, takes sessions per year from the session calendar, and returns a
`Performance`. `openalgo.backtest.report.tearsheet` renders one as an HTML report.
Every definition cites its source (a paper or a textbook) in the kernel's module docs
(`oa_bt::metrics`).

Inputs
    equity: a pandas Series of equity in rupees, indexed by time (a time-zone-aware
        DatetimeIndex in any zone, or int64 UTC nanoseconds), strictly increasing. Each
        point is the marked equity at that instant (bars are stamped at their close),
        except that without an opening wealth the first point is taken as the starting
        equity, before any P&L of the run. At least one point with an opening wealth,
        two without.
    opening_wealth: the capital before the first equity point, in rupees: the wealth
        the run opens with. Pass it whenever the first point can already hold P&L, as
        the close of a backtest's first slice can (`Result.performance` passes the
        run's capital). The first point is then a mark: its P&L counts in the total
        return and CAGR, its session has a return from the opening wealth, and a fall
        below the opening wealth is a drawdown. None (the default) keeps the reading
        above: the first point is the start and belongs to no session.
    opening_time: the instant the run opened with its opening wealth (a time-zone-aware
        time or int64 UTC nanoseconds), before the first point; only with an opening
        wealth. It is the start: elapsed time and CAGR are counted from it, the first
        bar runs from it to the first point (so exposure counts the first slice), and a
        drawdown that starts at the opening is timed from it. `Result.performance`
        passes the open of the run's first slice. None (the default) times the opening
        at the first point.
    rolling_window: sessions in each window of `Performance.rolling`, 2 or more; by
        default half a year of sessions of the traded calendar.
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
        benchmark value is unknown has no benchmark return. With an opening wealth the
        start is read as the benchmark's last value stamped before the first point.
    risk_free_rate: the annual risk-free rate as a fraction (0.065 is 6.5 percent).
        It is converted to the per-session rate (1 + rate) ^ (1 / sessions per year) - 1
        before it is subtracted from session returns.
    calendar: the session calendar; the shared built-in one by default.

Conventions
    Session returns: the IST date of each mark is its session date (every point with an
    opening wealth, every point after the first without one), and the return of a
    session is the time-weighted wealth at its last mark over the wealth at the
    previous session's last mark (the start for the first: the opening wealth, else the
    first point), minus 1. Total return, drawdowns and the monthly and yearly tables
    are built from the same start.
    Annualisation: CAGR uses elapsed calendar time (365.25 days a year). Volatility,
    Sharpe, Sortino and alpha use sessions per year: the sessions of the traded
    exchanges' union calendar in the whole calendar years the run touches, scaled to
    365.25 days, never a fixed 252 or 365. A year with no holiday list is reported in
    `annualisation_template_years` and announced with a CalendarWarning.
    Ruin: once equity before cash flows reaches zero or below, the wealth index is 0
    and later sessions have no return; total return and CAGR are -1.
    Drawdowns: every drawdown figure is a fall, zero or negative, as a fraction
    (max_drawdown, `Performance.drawdown`, episode depths, the conditional drawdown, the
    horizon and Monte Carlo drawdowns) or in rupees (max_drawdown_amount). Durations
    and the ulcer index are zero or positive.
    Undefined figures are NaN; a positive figure over a zero denominator is infinite.

Several strategies on fixed allocations: each strategy's equity is analysed on its own
allocation, its opening wealth. The combined account opens with the sum of their
capitals, and its trades are their trades tables concatenated:

    combined = combine_equity(equities, opening_wealth=capitals)
    analyse(combined, exchanges=..., opening_wealth=sum(capitals.values()), trades=...)

`return_correlation` compares the strategies.

Errors: ConfigError for any input that cannot be used, with a message that names the
column and row; CalendarError for an unknown exchange.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from .calendar import Calendar
from .report import _analysis, _combined, _montecarlo
from .report._definitions import DEFINITIONS, GROUPS, Definition, as_text
from .report._montecarlo import MonteCarlo
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
    opening_wealth: float | None = None,
    opening_time: object = None,
    rolling_window: int | None = None,
) -> Performance:
    """Every metric of one equity series, with its trades, flows and benchmark.

    The inputs are described in the module docstring and every key of the returned
    `Performance.metrics` in `DEFINITIONS`. `opening_wealth` (rupees) is the capital
    before the first point; None takes the first point as the starting equity.
    `opening_time` is the instant the run opened with it; None times the opening at the
    first point. `rolling_window` is the sessions per window of the rolling statistics.
    Raises ConfigError for unusable input (naive timestamps, unsorted points, a trade
    whose net P&L is not gross P&L minus costs, a rate written in percent, an opening
    wealth that is not a positive number, an opening time without one or not before the
    first point, a window below two sessions) and CalendarError for an unknown exchange.
    """
    return _analysis.analyse(
        equity,
        exchanges=exchanges,
        trades=trades,
        cash_flows=cash_flows,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        calendar=calendar,
        opening_wealth=opening_wealth,
        opening_time=opening_time,
        rolling_window=rolling_window,
    )


def monte_carlo(
    performance: Performance,
    *,
    simulations: int = _montecarlo.DEFAULT_SIMULATIONS,
    seed: int = _montecarlo.DEFAULT_SEED,
    resample: str = "sessions",
    bust: float | None = None,
    goal: float | None = None,
) -> MonteCarlo:
    """Monte Carlo by bootstrap (Efron 1979): `simulations` alternative runs, each
    drawing with replacement as many items as the run had.

    resample="sessions" draws session returns, which compound into the path.
    resample="trades" draws the closed trades' net P&L in rupees and books them one
    after another on the start equity at fixed rupee size (trades are sized in units,
    not as a share of equity); a path whose equity reaches zero is ruined there.

    Each path is measured like the run: final return, CAGR over the run's elapsed time,
    maximum drawdown on the path's closes and Sharpe (annualised by sessions per year,
    or for trades by the run's trades per year). `bust` (a maximum drawdown, -1 to 0)
    and `goal` (a final return) add the share of paths that reached them. The same
    inputs and `seed` give the same simulations on every platform. Returns a
    `MonteCarlo`. Raises ConfigError for a count or seed that is not a whole number
    (simulations at least 1, seed 0 or more), another resample, a threshold out of
    range, or nothing to draw (no session, or no closed trade).
    """
    return _montecarlo.monte_carlo(
        performance, simulations=simulations, seed=seed, resample=resample, bust=bust, goal=goal
    )


def combine_equity(
    equities: Mapping[str, pd.Series], *, opening_wealth: Mapping[str, float] | None = None
) -> pd.Series:
    """The combined equity, in rupees, of strategies run on fixed capital allocations.

    Each strategy trades its own allocation, so the combined account is the sum of
    their equities on the union of their timestamps. A strategy's equity at an instant
    is its last value at or before it. Before its first point it is its capital when
    `opening_wealth` maps every strategy name to its capital in rupees, and otherwise
    its first value (which is then its starting equity). `equities` maps a strategy
    name to an equity Series indexed as for `analyse`. Returns a Series indexed by IST
    time; with capitals, analyse it with `opening_wealth=sum(capitals.values())`, the
    combined capital. Raises ConfigError for an empty mapping, a malformed series, or
    capitals that do not name exactly the strategies or are not positive numbers.
    """
    return _combined.combine_equity(equities, opening_wealth)


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
    "GROUPS",
    "Definition",
    "MonteCarlo",
    "Performance",
    "analyse",
    "combine_equity",
    "monte_carlo",
    "return_correlation",
]
