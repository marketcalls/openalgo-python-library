"""Every key of `Performance.metrics`: its label, its unit and its definition.

This table is the one source of truth for what a metric means. The `metrics` module
appends it to its docstring, and the tearsheet takes labels and units from it.

Units:
    fraction    0.01 is one percent (returns, rates, drawdowns, shares)
    rupees      Indian rupees
    ratio       a pure number
    count       a whole number
    days        calendar days, fractional
    seconds     elapsed seconds
    sessions    sessions per year
    timestamp   a pandas Timestamp in IST, or None
    text        a string
    flag        True or False

Sign of a drawdown: every drawdown figure is a fall, so it is zero or negative, whether
a fraction (max_drawdown, the drawdown series, episode depths, conditional_drawdown,
the horizon and Monte Carlo drawdowns) or rupees (max_drawdown_amount). Durations are
zero or positive, and so is the ulcer index, a root mean square.
"""

from types import MappingProxyType
from typing import NamedTuple

from ._definitions_extra import BENCHMARK_EXTRA, RISK_ADJUSTED, SESSIONS, TAIL, TRADE_RATIOS


class Definition(NamedTuple):
    """How one metric is labelled, measured and defined."""

    label: str
    unit: str
    definition: str


UNITS = (
    "fraction",
    "rupees",
    "ratio",
    "count",
    "days",
    "seconds",
    "sessions",
    "timestamp",
    "text",
    "flag",
)

_RUN = {
    "start": (
        "Start",
        "timestamp",
        "The opening time when one is given (the instant the run opened with its "
        "capital); else the time of the first equity point: the starting equity, or with "
        "an opening wealth the first mark after it (the opening is then counted from "
        "here).",
    ),
    "end": ("End", "timestamp", "Time of the last equity point."),
    "elapsed_days": (
        "Elapsed",
        "days",
        "Calendar days from start to end; CAGR is annualised over them.",
    ),
    "sessions": (
        "Sessions",
        "count",
        "IST session dates with an equity point after the start (every point, with an "
        "opening wealth); one return each, the first one from the start equity.",
    ),
    "missing_sessions": (
        "Missing sessions",
        "count",
        "Calendar sessions of the traded exchanges, between the first and last session "
        "date, with no equity point; each one's move is folded into the next return.",
    ),
    "start_equity": (
        "Start equity",
        "rupees",
        "The opening wealth (the capital before the first point) when one is given, else "
        "the equity at the first point.",
    ),
    "end_equity": ("End equity", "rupees", "Equity at the last point."),
    "net_cash_flows": ("Net cash flows", "rupees", "Deposits minus withdrawals in the run."),
    "net_profit": (
        "Net profit",
        "rupees",
        "End equity minus start equity minus net cash flows.",
    ),
}

_RETURNS = {
    "total_return": (
        "Total return",
        "fraction",
        "Time-weighted: the product of (1 + session return), minus 1, so measured from the "
        "start equity. Cash flows are not returns.",
    ),
    "cagr": (
        "CAGR",
        "fraction",
        "(1 + total return) ^ (365.25 / elapsed days) - 1. NaN below 28 elapsed days, "
        "-1 after ruin.",
    ),
    "volatility": (
        "Volatility",
        "fraction",
        "Sample standard deviation of session returns, times sqrt(sessions per year).",
    ),
    "sharpe": (
        "Sharpe ratio",
        "ratio",
        "Mean excess session return over its sample standard deviation, times "
        "sqrt(sessions per year). NaN when the deviation is zero.",
    ),
    "sortino": (
        "Sortino ratio",
        "ratio",
        "Mean excess session return over the downside deviation "
        "sqrt(sum(min(excess, 0)^2) / n), times sqrt(sessions per year).",
    ),
    "calmar": ("Calmar ratio", "ratio", "CAGR over the magnitude of the maximum drawdown."),
    "exposure": (
        "Exposure",
        "fraction",
        "Share of the equity series' bars during which a position was open. A bar runs "
        "from one equity point to the next; with an opening time the first bar runs from "
        "it to the first point, else time before the first point is not a bar.",
    ),
    "risk_free_rate": ("Risk-free rate", "fraction", "The annual risk-free rate supplied."),
    "risk_free_per_session": (
        "Risk-free per session",
        "fraction",
        "(1 + annual rate) ^ (1 / sessions per year) - 1, subtracted from every session.",
    ),
    "sessions_per_year": (
        "Sessions per year",
        "sessions",
        "Sessions of the traded exchanges (union calendar) in the whole calendar years "
        "the run touches, scaled to 365.25 days.",
    ),
    "annualisation_years": (
        "Annualisation years",
        "text",
        "The calendar years counted for sessions per year.",
    ),
    "annualisation_coverage": (
        "Calendar coverage",
        "text",
        "Weakest holiday-data coverage of those years: sourced, unverified or template.",
    ),
    "annualisation_template_years": (
        "Template-only years",
        "text",
        "Exchange-years with no holiday list; sessions per year is overstated by them.",
    ),
}

_DRAWDOWN = {
    "max_drawdown": (
        "Maximum drawdown",
        "fraction",
        "Deepest fall of the time-weighted wealth index from its running peak, on every "
        "equity point. The peak starts at the start equity, so a first point below an "
        "opening wealth is a fall. Zero or negative.",
    ),
    "max_drawdown_amount": (
        "Maximum drawdown amount",
        "rupees",
        "Largest fall of cumulative P&L (equity less the start equity and cash flows) "
        "from its running peak, which starts at 0. Zero or negative: a fall of 500 "
        "rupees is -500.",
    ),
    "max_drawdown_peak": (
        "Drawdown peak",
        "timestamp",
        "Peak before the maximum drawdown; the start when the peak is the opening wealth "
        "(a drawdown from the opening is timed from the start).",
    ),
    "max_drawdown_trough": ("Drawdown trough", "timestamp", "Lowest point of it."),
    "max_drawdown_recovery": (
        "Drawdown recovery",
        "timestamp",
        "First point back at the peak, or None when not recovered.",
    ),
    "max_drawdown_days": (
        "Maximum drawdown length",
        "days",
        "Calendar days from its peak to its recovery, or to the end when unrecovered.",
    ),
    "longest_drawdown_days": (
        "Longest drawdown",
        "days",
        "Calendar days of the longest time below a peak, recovered or not.",
    ),
    "longest_drawdown_start": (
        "Longest drawdown start",
        "timestamp",
        "Its peak; the start when that is the opening wealth.",
    ),
    "longest_drawdown_end": (
        "Longest drawdown end",
        "timestamp",
        "Its recovery, or the last point when unrecovered.",
    ),
    "longest_drawdown_recovered": ("Longest drawdown recovered", "flag", "Whether it ended."),
    "ruined": ("Ruined", "flag", "Whether equity before cash flows reached zero or below."),
    "ruin_time": ("Ruin", "timestamp", "When it did, or None."),
}

_TRADES = {
    "trades": ("Trades", "count", "Closed round trips."),
    "open_trades": ("Open trades", "count", "Trades with no exit; not in trade statistics."),
    "winning_trades": ("Winning trades", "count", "Closed trades with net P&L above zero."),
    "losing_trades": ("Losing trades", "count", "Closed trades with net P&L below zero."),
    "win_rate": ("Win rate", "fraction", "Winning trades over closed trades."),
    "average_win": ("Average win", "rupees", "Mean net P&L of winning trades."),
    "average_loss": ("Average loss", "rupees", "Mean net P&L of losing trades (negative)."),
    "profit_factor": (
        "Profit factor",
        "ratio",
        "Sum of winning net P&L over the magnitude of the sum of losing net P&L.",
    ),
    "expectancy": ("Expectancy", "rupees", "Mean net P&L per closed trade."),
    "payoff_ratio": ("Payoff ratio", "ratio", "Average win over the magnitude of average loss."),
    "largest_win": ("Largest win", "rupees", "Highest net P&L of a winning trade."),
    "largest_loss": ("Largest loss", "rupees", "Lowest net P&L of a losing trade."),
    "max_consecutive_wins": (
        "Consecutive wins",
        "count",
        "Longest run of wins in exit order; a breakeven trade breaks a run.",
    ),
    "max_consecutive_losses": ("Consecutive losses", "count", "Longest run of losses."),
    "average_holding_seconds": (
        "Average holding time",
        "seconds",
        "Mean of exit time minus entry time over closed trades.",
    ),
    "gross_pnl": ("Gross P&L", "rupees", "Sum of closed trades' P&L before costs."),
    "total_costs": ("Costs", "rupees", "Sum of closed trades' costs."),
    "net_pnl": ("Net P&L", "rupees", "Sum of closed trades' P&L after costs."),
    "cost_share_of_gross_pnl": (
        "Costs / gross P&L",
        "fraction",
        "Costs over gross P&L; NaN when gross P&L is zero or negative.",
    ),
    "turnover": (
        "Turnover",
        "rupees",
        "Traded value: |price| x quantity of every entry and every exit.",
    ),
    "turnover_ratio": ("Turnover ratio", "ratio", "Turnover over mean equity."),
}

_BENCHMARK = {
    "benchmark_total_return": (
        "Benchmark return",
        "fraction",
        "Benchmark value at the end over its first value in the run (as of the start, "
        "which with an opening wealth is its last value before the first point; else as "
        "of the first session end it covers), minus 1.",
    ),
    "benchmark_sessions": (
        "Benchmark sessions",
        "count",
        "Sessions with both a strategy and a benchmark return.",
    ),
    "alpha": (
        "Alpha",
        "fraction",
        "Jensen's alpha per year: (mean(s - rf) - beta * mean(b - rf)) x sessions per year.",
    ),
    "beta": ("Beta", "ratio", "cov(strategy, benchmark) / var(benchmark), per session."),
    "correlation": ("Correlation", "ratio", "Pearson correlation of session returns."),
    **BENCHMARK_EXTRA,
}

_GROUPS = (
    ("Run", _RUN),
    ("Returns and risk", _RETURNS),
    ("Drawdown", _DRAWDOWN),
    ("Trades", _TRADES),
    ("Benchmark", _BENCHMARK),
    ("Risk-adjusted ratios", RISK_ADJUSTED),
    ("Session returns", SESSIONS),
    ("Tail risk", TAIL),
    ("Trade ratios", TRADE_RATIOS),
)

#: Every metric key, in report order, with its label, unit and definition.
DEFINITIONS = MappingProxyType(
    {key: Definition(*row) for _, group in _GROUPS for key, row in group.items()}
)

#: The metric keys by group heading, in report order (every key in exactly one group).
GROUPS = MappingProxyType({heading: tuple(group) for heading, group in _GROUPS})

#: Keys present only when a benchmark is given.
BENCHMARK_KEYS = tuple(_BENCHMARK)


def as_text() -> str:
    """The table as plain text, one metric per paragraph, for a docstring."""
    lines = ["", "Metrics (key, unit: definition)", ""]
    for key, entry in DEFINITIONS.items():
        lines.append(f"    {key} ({entry.unit}): {entry.definition}")
    return "\n".join(lines) + "\n"


__all__ = ["BENCHMARK_KEYS", "DEFINITIONS", "GROUPS", "UNITS", "Definition", "as_text"]
