"""Compute a `Performance` from an equity series and what goes with it.

The per-bar and per-trade work runs in the native kernel (`oa_bt::metrics`); this
module converts the inputs, asks the calendar for sessions per year, and assembles
the metrics mapping in the order of `DEFINITIONS`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .._layout import NANOS_PER_SECOND
from ..calendar import Calendar, default_calendar
from ..errors import ConfigError
from . import _extended as extended
from . import _periods as periods
from . import _rolling as rolling
from . import _tables_data as tables_data
from ._definitions import DEFINITIONS
from ._inputs import (
    TradeTable,
    benchmark_columns,
    check_opening_wealth,
    flow_columns,
    series_columns,
    trade_table,
    utc_nanos,
)
from ._kernel import call
from ._performance import Performance


@dataclass(frozen=True)
class EquityPoints:
    """The analysed equity series and where it starts.

    ts: int64 UTC nanoseconds; values: rupees; opening: the opening wealth in rupees
    (the capital before the first point), or None when the first point is the start;
    opening_ns: the opening time in UTC nanoseconds (the instant the run opened with its
    capital), or None when the opening is timed at the first point.
    """

    ts: np.ndarray
    values: np.ndarray
    opening: float | None
    opening_ns: int | None = None

    @property
    def start_equity(self) -> float:
        """Rupees at the start: the opening wealth, else the first point."""
        return float(self.values[0]) if self.opening is None else self.opening

    @property
    def start_ns(self) -> int:
        """UTC nanoseconds of the start: the opening time, else the first point."""
        return int(self.ts[0]) if self.opening_ns is None else self.opening_ns

    @property
    def marks(self) -> np.ndarray:
        """Timestamps of the points that fall in a session: every point when there is
        an opening wealth, every point after the first (the start) when there is none."""
        return self.ts if self.opening is not None else self.ts[1:]

    @property
    def bar_edges(self) -> np.ndarray:
        """The instants that bound the bars: the opening time (when given) and every
        point."""
        if self.opening_ns is None:
            return self.ts
        return np.concatenate(([self.opening_ns], self.ts)).astype(np.int64)


def exchange_list(exchanges: str | Sequence[str]) -> list[str]:
    """The exchange codes as a list without repeats. Raises ConfigError when empty."""
    codes = [exchanges] if isinstance(exchanges, str) else list(exchanges)
    if not codes or not all(isinstance(code, str) for code in codes):
        raise ConfigError(
            "exchanges must name the OpenAlgo exchanges the strategy trades, for example "
            "'NSE' or ['NFO', 'MCX']: their calendars give the sessions per year"
        )
    return list(dict.fromkeys(codes))


def risk_free(rate: object) -> float:
    """The annual risk-free rate as a fraction. Raises ConfigError outside (-1, 1)."""
    try:
        value = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = math.nan
    if not -1.0 < value < 1.0:
        raise ConfigError(
            f"risk_free_rate is an annual fraction between -1 and 1 (0.065 is 6.5 "
            f"percent), got {rate!r}"
        )
    return value


def opening_instant(value: object, opening: float | None, ts: np.ndarray) -> int | None:
    """The opening time in UTC nanoseconds, or None. Raises ConfigError for a time
    without an opening wealth, a value that is not a time-zone-aware time (or int64
    UTC nanoseconds), or a time that is not before the first equity point."""
    if value is None:
        return None
    if opening is None:
        raise ConfigError(
            "opening_time is the instant the run opened with its opening_wealth: pass "
            "opening_wealth with it"
        )
    nanos, missing = utc_nanos([value], what="opening_time")
    if missing.any() or (len(ts) and int(nanos[0]) >= int(ts[0])):
        raise ConfigError(
            f"opening_time must be a time before the first equity point, got {value!r}"
        )
    return int(nanos[0])


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
    """Implementation of `openalgo.backtest.metrics.analyse` (documented there)."""
    codes = exchange_list(exchanges)
    rate = risk_free(risk_free_rate)
    opening = None if opening_wealth is None else check_opening_wealth(opening_wealth)
    window = None if rolling_window is None else rolling.check_window(rolling_window)
    cal = default_calendar() if calendar is None else calendar
    ts, values = series_columns(equity, what="equity")
    points = EquityPoints(ts, values, opening, opening_instant(opening_time, opening, ts))
    curve = call("curve", ts, values, *flow_columns(cash_flows), opening, points.opening_ns)
    days = np.asarray(curve["session_days"], dtype=np.int64)
    annual, years = periods.annualisation(cal, codes, points.start_ns, int(ts[-1]))
    table = trade_table(trades)
    figures = _run_figures(points, curve, periods.missing_sessions(cal, codes, days))
    figures |= _return_figures(points, curve, annual, years, rate)
    figures |= _drawdown_figures(points, curve)
    figures |= _trade_figures(table, points)
    returns = pd.Series(curve["session_returns"], index=periods.date_index(days), name="return")
    monthly, yearly = periods.period_returns(returns)
    streaks = _extended_figures(figures, returns, (monthly, yearly), table, points, annual.value)
    benchmark_equity = bench_returns = None
    if benchmark is not None:
        extra, benchmark_equity, bench_returns = _benchmark(
            benchmark, points, days, returns, annual.value, rate
        )
        figures |= extra
    window = rolling.default_window(annual.value) if window is None else window
    figures["rolling_window_sessions"] = window
    index = periods.ist_times(ts)
    return Performance(
        metrics=MappingProxyType({key: figures[key] for key in DEFINITIONS if key in figures}),
        equity=pd.Series(values, index=index, name="equity"),
        drawdown=pd.Series(curve["drawdown"], index=index, name="drawdown"),
        session_returns=returns,
        monthly_returns=monthly,
        yearly_returns=yearly,
        trades=table.frame,
        cost_breakdown=MappingProxyType(table.cost_breakdown()),
        benchmark_equity=benchmark_equity,
        exchanges=tuple(codes),
        wealth=pd.Series(curve["wealth"], index=index, name="wealth"),
        drawdown_episodes=tables_data.drawdown_episodes(
            curve["drawdown_episodes"], ts, points.start_ns
        ),
        rolling=rolling.rolling_frame(returns, bench_returns, window, annual.value, rate),
        benchmark_returns=bench_returns,
        horizons=rolling.horizon_frame(
            returns, bench_returns, points.start_ns, int(ts[-1]), annual.value, rate
        ),
        losing_streaks=tables_data.losing_streaks(*streaks),
    )


def _extended_figures(
    figures: dict,
    returns: pd.Series,
    calendar_tables: tuple[pd.DataFrame, pd.Series],
    table: TradeTable,
    points: EquityPoints,
    sessions_per_year: float,
) -> tuple[list[int], list[int]]:
    """Add the distribution, trade-ratio and composite figures to `figures`, which
    already holds the core ones; return the losing streaks of sessions and of trades.
    `calendar_tables` are the monthly and yearly returns."""
    rate = figures["risk_free_rate"]
    session_extra, session_streaks = extended.session_figures(
        returns, sessions_per_year, rate, figures["cagr"], *calendar_tables
    )
    trade_extra, trade_streaks = extended.trade_ratio_figures(table, points.start_equity)
    figures |= session_extra | trade_extra
    figures |= extended.composite_figures(figures)
    return session_streaks, trade_streaks


def _run_figures(points: EquityPoints, curve: dict, missing: int) -> dict:
    ts = points.ts
    return {
        "start": periods.ist_instant(points.start_ns),
        "end": periods.ist_instant(ts[-1]),
        "elapsed_days": float(ts[-1] - points.start_ns) / periods.NANOS_PER_DAY,
        "sessions": len(curve["session_returns"]),
        "missing_sessions": missing,
        "start_equity": points.start_equity,
        "end_equity": float(points.values[-1]),
        "net_cash_flows": float(np.sum(curve["flow"])),
        "net_profit": float(curve["pnl"][-1]),
    }


def _return_figures(points: EquityPoints, curve: dict, annual, years: str, rate: float) -> dict:
    growth = float(curve["wealth"][-1])
    stats = call("returns", curve["session_returns"], annual.value, rate)
    cagr = call("cagr", growth, int(points.ts[-1]) - points.start_ns)
    deepest = curve["max_drawdown"]
    return {
        "total_return": growth - 1.0,
        "cagr": cagr,
        "volatility": stats["volatility"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "calmar": call("calmar", cagr, deepest[3] if deepest else 0.0),
        "risk_free_rate": rate,
        "risk_free_per_session": stats["risk_free_per_period"],
        "sessions_per_year": annual.value,
        "annualisation_years": years,
        "annualisation_coverage": annual.coverage,
        "annualisation_template_years": ", ".join(f"{c} {y}" for c, y in annual.template_only),
    }


def _drawdown_figures(points: EquityPoints, curve: dict) -> dict:
    ts = points.ts

    def instant(position: int | None) -> pd.Timestamp | None:
        return None if position is None else periods.ist_instant(ts[position])

    def peak_instant(position: int | None) -> pd.Timestamp | None:
        # The kernel gives no position for a peak at the opening wealth: it is reported
        # at the start (the opening time, else the first point), where it is timed.
        return periods.ist_instant(points.start_ns) if position is None else instant(position)

    figures: dict[str, Any] = {
        "max_drawdown": 0.0,
        "max_drawdown_amount": curve["max_drawdown_amount"],
        "max_drawdown_peak": None,
        "max_drawdown_trough": None,
        "max_drawdown_recovery": None,
        "max_drawdown_days": 0.0,
        "longest_drawdown_days": 0.0,
        "longest_drawdown_start": None,
        "longest_drawdown_end": None,
        "longest_drawdown_recovered": None,
        "ruined": curve["ruin"] is not None,
        "ruin_time": instant(curve["ruin"]),
    }
    if curve["max_drawdown"] is not None:
        peak, trough, recovery, depth, duration = curve["max_drawdown"]
        figures["max_drawdown"] = depth
        figures["max_drawdown_peak"] = peak_instant(peak)
        figures["max_drawdown_trough"] = instant(trough)
        figures["max_drawdown_recovery"] = instant(recovery)
        figures["max_drawdown_days"] = duration / periods.NANOS_PER_DAY
    if curve["longest_drawdown"] is not None:
        peak, _, recovery, _, duration = curve["longest_drawdown"]
        figures["longest_drawdown_days"] = duration / periods.NANOS_PER_DAY
        figures["longest_drawdown_start"] = peak_instant(peak)
        figures["longest_drawdown_end"] = instant(len(ts) - 1 if recovery is None else recovery)
        figures["longest_drawdown_recovered"] = recovery is not None
    return figures


def _trade_figures(table: TradeTable, points: EquityPoints) -> dict:
    closed = table.closed
    stats = call(
        "trades",
        closed(table.entry_ts),
        closed(table.exit_ts),
        closed(table.gross_pnl),
        closed(table.net_pnl),
        closed(table.costs),
    )
    entries = np.sum(np.abs(table.entry_price) * table.quantity)
    exits = np.sum(np.abs(closed(table.exit_price)) * closed(table.quantity))
    turnover = float(entries + exits)
    mean_equity = float(np.mean(points.values))
    return {
        "exposure": call("exposure", points.bar_edges, table.entry_ts, table.holding_exits()),
        "trades": stats["trades"],
        "open_trades": int(np.sum(table.is_open)),
        "winning_trades": stats["wins"],
        "losing_trades": stats["losses"],
        **{key: stats[key] for key in _TRADE_KEYS},
        "average_holding_seconds": stats["average_holding"] / NANOS_PER_SECOND,
        "cost_share_of_gross_pnl": stats["cost_share_of_gross"],
        "turnover": turnover,
        "turnover_ratio": turnover / mean_equity if mean_equity > 0 else math.nan,
    }


# Trade statistics whose kernel name is their metrics key.
_TRADE_KEYS = (
    "win_rate",
    "average_win",
    "average_loss",
    "profit_factor",
    "expectancy",
    "payoff_ratio",
    "largest_win",
    "largest_loss",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "gross_pnl",
    "total_costs",
    "net_pnl",
)
# Benchmark figures: metrics key and kernel name.
_BENCHMARK_KEYS = {
    "alpha": "alpha",
    "beta": "beta",
    "correlation": "correlation",
    "r_squared": "r_squared",
    "tracking_error": "tracking_error",
    "information_ratio": "information_ratio",
    "treynor_ratio": "treynor",
}


def _benchmark(
    benchmark: pd.Series,
    points: EquityPoints,
    days: np.ndarray,
    returns: pd.Series,
    sessions_per_year: float,
    rate: float,
) -> tuple[dict, pd.Series, pd.Series]:
    """Benchmark figures, the benchmark scaled to the starting equity, and its session
    returns on the strategy's session dates.

    The benchmark is read as of the start and as of each session's last equity point,
    so each of its returns spans the same instants as the strategy's return; a session
    whose previous benchmark value is unknown has no benchmark return. Without an
    opening wealth the start is the first point, read at its instant. With one, the
    first point is already a mark after the opening, so the start is read at the
    opening time when it is given and otherwise as the last benchmark value stamped
    before that point. The base is the first value known in the run: as of the start,
    or else as of the first session end it covers (daily closes stamped after an
    intraday start, for example).
    """
    bench_ts, bench_values = benchmark_columns(benchmark)
    ends = periods.session_end_times(points.marks, days)
    first = int(points.ts[0])
    if points.opening is None:
        start = first
    elif points.opening_ns is not None:
        start = points.opening_ns
    else:
        # At or before one nanosecond earlier is strictly before the first point.
        start = first - 1
    path = periods.benchmark_path(bench_ts, bench_values, start, ends)
    bench_returns = path[1:] / path[:-1] - 1.0
    stats = call("benchmark", returns.to_numpy(), bench_returns, sessions_per_year, rate)
    known = np.flatnonzero(np.isfinite(path))
    base = path[known[0]] if len(known) else math.nan
    figures = {
        "benchmark_total_return": float(path[-1] / base - 1.0),
        "benchmark_sessions": stats["sessions"],
        **{key: stats[name] for key, name in _BENCHMARK_KEYS.items()},
    }
    by_session = pd.Series(bench_returns, index=returns.index, name="benchmark")
    scaled = points.start_equity * path / base
    if points.opening is not None:
        # The opening has no row of its own: its base value is not in the series.
        equity = pd.Series(scaled[1:], index=periods.ist_times(ends), name="benchmark")
        return figures, equity, by_session
    index = periods.ist_times(np.concatenate(([first], ends)))
    return figures, pd.Series(scaled, index=index, name="benchmark"), by_session


__all__ = ["analyse", "exchange_list", "opening_instant", "risk_free"]
