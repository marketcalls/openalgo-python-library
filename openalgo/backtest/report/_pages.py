"""The sections of each report mode, in page order.

full     every section: summary, equity with the longest drawdowns shaded,
         cumulative return, session returns, monthly and yearly returns, return
         distribution, drawdowns, rolling statistics, horizons, benchmark (when
         given), Monte Carlo, trade P&L distribution, trades, losing streaks, costs,
         R-multiples (when a closed trade has an initial risk), strategies (when
         given), assumptions and notes
basic    summary with the basic facts, equity, monthly and yearly returns, drawdowns,
         trades, costs, assumptions and notes
metrics  every metric with its definition, and the assumptions
charts   every chart and nothing else
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .._messages import counted
from . import _charts as charts
from . import _charts_series as series
from . import _charts_simulation as simulation
from . import _format as fmt
from . import _r_multiples as r_multiples
from . import _sections as sections
from . import _tables as tables
from . import _tables_data as tables_data
from . import _tables_extra as extra
from ._combined import return_correlation
from ._montecarlo import DEFAULT_SEED, DEFAULT_SIMULATIONS, monte_carlo
from ._performance import Performance

#: Report modes, in the order the documentation lists them.
MODES = ("full", "basic", "metrics", "charts")
#: Worst drawdowns listed in the drawdowns section.
WORST_DRAWDOWNS = 5
#: Bin width of a simulated figure whose paths all agree: a hundredth of a percent.
MC_HISTOGRAM_FLOOR = 0.0001
#: Rolling statistics charted, with their labels and whether they are percentages.
ROLLING_CHARTS = (
    ("volatility", "Rolling volatility", True),
    ("sharpe", "Rolling Sharpe", False),
    ("sortino", "Rolling Sortino", False),
    ("beta", "Rolling beta", False),
    ("win_rate", "Rolling session win rate", True),
)


def _closed_net_pnl(trades: pd.DataFrame) -> np.ndarray:
    if trades.empty:
        return np.empty(0)
    closed = trades.loc[~pd.isna(trades["exit_time"]), "net_pnl"]
    return closed.to_numpy(dtype=float)


def _legend(*keys: tuple[str, str, bool]) -> str:
    """A legend of (colour token, label, swatch) entries: a swatch for an area, a short
    rule for a line."""
    spans = "".join(
        f"<span{" class='swatch'" if swatch else ''} style='--key: var(--{token})'>{label}</span>"
        for token, label, swatch in keys
    )
    return f"<div class='legend'>{spans}</div>"


def summary(performance: Performance, lists: tuple) -> str:
    metrics = performance.metrics
    body = sections.headline(metrics) + sections.facts(metrics, lists)
    return sections.section("summary", "Summary", body)


def equity(performance: Performance) -> str:
    keys = [("gain", "Strategy", False), ("shade", "Longest drawdowns", True)]
    if performance.benchmark_equity is not None:
        keys.insert(1, ("bench", "Benchmark, scaled to the start equity", False))
    lede = (
        f"Equity in rupees at every mark, with the {charts.SHADED_DRAWDOWNS} longest "
        "drawdown periods shaded. Below it, the fall of the time-weighted wealth index "
        "from its running peak (cash flows excluded)."
    )
    body = _legend(*keys) + charts.equity_chart(performance)
    return sections.section("equity", "Equity and drawdown", body, lede)


def cumulative(performance: Performance, log_scale: bool) -> str:
    keys = [("gain", "Strategy", False)]
    if performance.benchmark_equity is not None:
        keys.append(("bench", "Benchmark", False))
    scale = "on a log scale of growth" if log_scale else "on a linear scale"
    lede = f"Time-weighted cumulative return since the start, {scale}."
    body = _legend(*keys) + series.cumulative_chart(performance, log_scale)
    return sections.section("cumulative", "Cumulative return", body, lede)


def daily(performance: Performance) -> str:
    body = series.session_bars(performance.session_returns)
    return sections.section("daily", "Session returns", body, "The return of every session.")


def monthly(performance: Performance) -> str:
    lede = "Compounded session returns per calendar month."
    return sections.section("monthly", "Monthly returns", tables.monthly_grid(performance), lede)


def yearly(performance: Performance) -> str:
    return sections.section("yearly", "Yearly returns", tables.yearly_bars(performance))


def return_distribution(performance: Performance) -> str:
    groups = tables_data.returns_by_period(performance.session_returns)
    body = simulation.box_plot(groups) + series.return_histogram(performance.session_returns)
    lede = (
        "Compounded returns per session, week, month, quarter and year (lowest, quartiles, "
        "median, highest; a first or last period may be partial), then the histogram of "
        "session returns."
    )
    return sections.section("return-distribution", "Return distribution", body, lede)


def drawdowns(performance: Performance) -> str:
    details = performance.drawdown_details(WORST_DRAWDOWNS)
    lede = (
        f"The {WORST_DRAWDOWNS} deepest falls of the wealth index, in calendar days: from "
        "the peak (the start for a fall from the opening) to the recovery."
    )
    return sections.section("drawdowns", "Worst drawdowns", extra.drawdown_table(details), lede)


def rolling(performance: Performance) -> str:
    window = int(performance.metrics["rolling_window_sessions"])
    frame = performance.rolling
    body = "".join(
        series.rolling_chart(frame[name], label, as_percent, window)
        for name, label, as_percent in ROLLING_CHARTS
        if name in frame.columns
    )
    lede = f"Each statistic over the {window} sessions ending at each session."
    return sections.section("rolling", "Rolling statistics", body, lede)


def horizons(performance: Performance) -> str:
    lede = (
        "Returns over trailing windows to the last session, listed when the run covers "
        "them; drawdowns at session closes."
    )
    body = extra.horizon_table(performance.horizons)
    return sections.section("horizons", "Horizons", body, lede)


def benchmark(performance: Performance) -> str:
    body = extra.benchmark_table(performance.benchmark_comparison("year"))
    lede = "Compounded returns per calendar year, the strategy against the benchmark."
    return sections.section("benchmark", "Against the benchmark", body, lede)


def _simulated(values: np.ndarray, title: str) -> str:
    """One panel: the histogram of a simulated figure (a fraction)."""
    chart = charts.histogram(
        values,
        label=f"Simulated {title.lower()}",
        edge=lambda value: fmt.percent(value, 1),
        unit="path(s)",
        floor=MC_HISTOGRAM_FLOOR,
    )
    return f"<div><p class='panel-title'>{title}</p>{chart}</div>"


def montecarlo(performance: Performance, with_table: bool = True) -> str:
    study = monte_carlo(performance)
    pair = (
        "<div class='pair'>"
        + _simulated(study.runs["final_return"].to_numpy(), "Final return")
        + _simulated(study.runs["max_drawdown"].to_numpy(), "Maximum drawdown")
        + "</div>"
    )
    keys = [("gain-wash-2", "25th to 75th percentile", True), ("gain-wash", "5th to 95th", True)]
    keys += [("gain", "Median", False), ("ink", "This run", False)]
    body = _legend(*keys) + simulation.band_chart(study, "sessions") + pair
    if with_table:
        body += extra.montecarlo_table(study)
    lede = (
        f"{fmt.indian_digits(DEFAULT_SIMULATIONS, 0)} runs drawn with replacement from the "
        f"session returns (seed {DEFAULT_SEED}): the spread of cumulative return, final "
        "return and maximum drawdown (on session closes)."
    )
    return sections.section("montecarlo", "Monte Carlo", body, lede)


def trade_distribution(performance: Performance) -> str:
    pnl = _closed_net_pnl(performance.trades)
    body = charts.pnl_histogram(pnl) if len(pnl) else "<p class='note'>No closed trades.</p>"
    lede = "Net P&amp;L of closed trades, after costs."
    return sections.section("distribution", "Trade P&amp;L distribution", body, lede)


def trade_list(performance: Performance) -> str:
    metrics, trades = performance.metrics, performance.trades
    closed = counted(metrics["trades"], "closed trade")
    counts = f"{closed} and {metrics['open_trades']} open, in entry order."
    listing = tables.trade_list(trades) if len(trades) else "<p class='note'>No trades.</p>"
    return sections.section("trades", "Trades", listing, counts)


def streaks(performance: Performance) -> str:
    lede = "How many losing streaks of each length: sessions in date order, trades in exit order."
    body = extra.streak_table(performance.losing_streaks)
    return sections.section("streaks", "Losing streaks", body, lede)


def costs(performance: Performance) -> str:
    body = tables.cost_table(performance.metrics, performance.cost_breakdown)
    return sections.section("costs", "Costs", body, "Costs of closed trades.")


def strategies_section(strategies: Mapping[str, Performance]) -> str:
    correlation = return_correlation(strategies)
    body = sections.comparison(strategies, correlation)
    return sections.section("strategies", "Strategies", body)


def assumptions(lines: list[str]) -> str:
    return sections.section("assumptions", "Assumptions", sections.assumption_list(lines))


def notes(performance: Performance, lists: tuple) -> str:
    body = sections.method_notes(performance.metrics, performance.exchanges, lists)
    return sections.section("notes", "Notes and definitions", body)


def metrics_page(performance: Performance) -> str:
    body = extra.metrics_list(performance.metrics)
    lede = "Every figure of the run, grouped, with its definition."
    return sections.section("metrics", "Metrics", body, lede)


def body(
    mode: str,
    performance: Performance,
    *,
    assumption_lines: list[str],
    strategies: Mapping[str, Performance] | None,
    log_scale: bool,
) -> list[str]:
    """The sections of `mode` (see the module docstring)."""
    p = performance
    if mode == "charts":
        parts = [equity(p), cumulative(p, log_scale), daily(p), return_distribution(p)]
        return [*parts, rolling(p), montecarlo(p, with_table=False), trade_distribution(p)]
    if mode == "metrics":
        return [metrics_page(p), assumptions(assumption_lines)]
    if mode == "basic":
        lists = sections.BASIC_FACTS
        parts = [summary(p, lists), equity(p), monthly(p), yearly(p), drawdowns(p)]
        return [*parts, trade_list(p), costs(p), assumptions(assumption_lines), notes(p, lists)]
    parts = [summary(p, sections.FACTS), equity(p), cumulative(p, log_scale), daily(p)]
    parts += [monthly(p), yearly(p), return_distribution(p), drawdowns(p), rolling(p)]
    parts.append(horizons(p))
    if p.benchmark_returns is not None:
        parts.append(benchmark(p))
    parts += [montecarlo(p), trade_distribution(p), trade_list(p), streaks(p), costs(p)]
    parts += r_multiples.section(p)
    if strategies is not None:
        parts.append(strategies_section(strategies))
    return [*parts, assumptions(assumption_lines), notes(p, sections.FACTS)]


__all__ = ["MODES", "ROLLING_CHARTS", "WORST_DRAWDOWNS", "body"]
