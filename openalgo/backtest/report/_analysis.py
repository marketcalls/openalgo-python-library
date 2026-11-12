"""Compute a `Performance` from an equity series and what goes with it.

The per-bar and per-trade work runs in the native kernel (`oa_bt::metrics`); this
module converts the inputs, asks the calendar for sessions per year, and assembles
the metrics mapping in the order of `DEFINITIONS`.
"""

import math
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .._layout import NANOS_PER_SECOND
from ..calendar import Calendar, default_calendar
from ..errors import ConfigError
from . import _periods as periods
from ._definitions import DEFINITIONS
from ._inputs import TradeTable, benchmark_columns, flow_columns, series_columns, trade_table
from ._kernel import call
from ._performance import Performance


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
    """Implementation of `openalgo.backtest.metrics.analyse` (documented there)."""
    codes = exchange_list(exchanges)
    rate = risk_free(risk_free_rate)
    cal = default_calendar() if calendar is None else calendar
    ts, values = series_columns(equity, what="equity")
    curve = call("curve", ts, values, *flow_columns(cash_flows))
    days = np.asarray(curve["session_days"], dtype=np.int64)
    annual, years = periods.annualisation(cal, codes, int(ts[0]), int(ts[-1]))
    table = trade_table(trades)
    figures = _run_figures(ts, values, curve, periods.missing_sessions(cal, codes, days))
    figures |= _return_figures(ts, curve, annual, years, rate)
    figures |= _drawdown_figures(ts, curve)
    figures |= _trade_figures(table, ts, values)
    returns = pd.Series(curve["session_returns"], index=periods.date_index(days), name="return")
    benchmark_equity = None
    if benchmark is not None:
        extra, benchmark_equity = _benchmark(
            benchmark, ts, values, days, returns, annual.value, rate
        )
        figures |= extra
    monthly, yearly = periods.period_returns(returns)
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
    )


def _run_figures(ts: np.ndarray, values: np.ndarray, curve: dict, missing: int) -> dict:
    return {
        "start": periods.ist_instant(ts[0]),
        "end": periods.ist_instant(ts[-1]),
        "elapsed_days": float(ts[-1] - ts[0]) / periods.NANOS_PER_DAY,
        "sessions": len(curve["session_returns"]),
        "missing_sessions": missing,
        "start_equity": float(values[0]),
        "end_equity": float(values[-1]),
        "net_cash_flows": float(np.sum(curve["flow"])),
        "net_profit": float(curve["pnl"][-1]),
    }


def _return_figures(ts: np.ndarray, curve: dict, annual, years: str, rate: float) -> dict:
    growth = float(curve["wealth"][-1])
    stats = call("returns", curve["session_returns"], annual.value, rate)
    cagr = call("cagr", growth, int(ts[-1] - ts[0]))
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


def _drawdown_figures(ts: np.ndarray, curve: dict) -> dict:
    def instant(position: int | None) -> pd.Timestamp | None:
        return None if position is None else periods.ist_instant(ts[position])

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
        figures["max_drawdown_peak"] = instant(peak)
        figures["max_drawdown_trough"] = instant(trough)
        figures["max_drawdown_recovery"] = instant(recovery)
        figures["max_drawdown_days"] = duration / periods.NANOS_PER_DAY
    if curve["longest_drawdown"] is not None:
        peak, _, recovery, _, duration = curve["longest_drawdown"]
        figures["longest_drawdown_days"] = duration / periods.NANOS_PER_DAY
        figures["longest_drawdown_start"] = instant(peak)
        figures["longest_drawdown_end"] = instant(len(ts) - 1 if recovery is None else recovery)
        figures["longest_drawdown_recovered"] = recovery is not None
    return figures


def _trade_figures(table: TradeTable, ts: np.ndarray, values: np.ndarray) -> dict:
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
    mean_equity = float(np.mean(values))
    return {
        "exposure": call("exposure", ts, table.entry_ts, table.holding_exits()),
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


def _benchmark(
    benchmark: pd.Series,
    ts: np.ndarray,
    values: np.ndarray,
    days: np.ndarray,
    returns: pd.Series,
    sessions_per_year: float,
    rate: float,
) -> tuple[dict, pd.Series]:
    """Benchmark figures and the benchmark scaled to the starting equity.

    The benchmark is read as of the start and as of each session's last equity point,
    so each of its returns spans the same instants as the strategy's return; a session
    whose previous benchmark value is unknown has no benchmark return. Its base is its
    first value known in the run: as of the start, or else as of the first session end
    it covers (daily closes stamped after an intraday start, for example).
    """
    bench_ts, bench_values = benchmark_columns(benchmark)
    ends = periods.session_end_times(ts, days)
    path = periods.benchmark_path(bench_ts, bench_values, int(ts[0]), ends)
    bench_returns = path[1:] / path[:-1] - 1.0
    stats = call("benchmark", returns.to_numpy(), bench_returns, sessions_per_year, rate)
    known = np.flatnonzero(np.isfinite(path))
    base = path[known[0]] if len(known) else math.nan
    figures = {
        "benchmark_total_return": float(path[-1] / base - 1.0),
        "benchmark_sessions": stats["sessions"],
        "alpha": stats["alpha"],
        "beta": stats["beta"],
        "correlation": stats["correlation"],
    }
    index = periods.ist_times(np.concatenate(([ts[0]], ends)))
    scaled = pd.Series(values[0] * path / base, index=index, name="benchmark")
    return figures, scaled


__all__ = ["analyse", "exchange_list", "risk_free"]
