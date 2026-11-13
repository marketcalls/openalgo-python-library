"""The stat tiles: a hairline grid of label, value and subline, in groups that follow
the reference dashboard's (overview, period returns, risk, benchmark, monthly,
distribution) and one for trades.

Every value is a figure of the SDK: a key of `Performance.metrics`, or a cell of one of
its tables (`data`). Where the reference shows a figure the SDK does not compute, the
tile holds the nearest SDK figure under that figure's own name. The subline is the
benchmark's figure, measured as the strategy's is (`data.benchmark_performance`), when
the figure is one a benchmark has (`data.COMPARABLE`), else a short description.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

from .. import _format as fmt
from .._definitions import DEFINITIONS
from .._performance import Performance
from . import data


class Stat(NamedTuple):
    """One tile: the label (HTML-safe text), the figure (a metrics key, or a reader of
    a table cell with its `definition`), how it is shown ("pct" a fraction as a percent,
    "num" two decimals, "int" a whole number, "rupees" whole rupees, "duration"
    seconds in words), the subline without a benchmark figure, and whether the value is
    coloured by its sign."""

    label: str
    figure: str | Callable[[Performance], float]
    shown: str = "num"
    description: str = ""
    tone: bool = False
    definition: str = ""


def _pct(
    label: str,
    figure: str | Callable[[Performance], float],
    description: str = "",
    tone: bool = True,
    definition: str = "",
) -> Stat:
    return Stat(label, figure, "pct", description, tone, definition)


def _num(label: str, figure: str, description: str = "") -> Stat:
    return Stat(label, figure, "num", description)


_MTD = "The return of the calendar month of the last session."
_YTD = "The return of the calendar year of the last session."
_TRAILING = "The total return over the trailing window to the last session."
_TRAILING_CAGR = "The CAGR over the trailing window to the last session."

OVERVIEW = (
    _pct("Cumulative Return", "total_return", "total"),
    _pct("CAGR", "cagr", "annualised"),
    _num("Sharpe", "sharpe", "risk-adjusted"),
    _num("Sortino", "sortino", "downside-adjusted"),
    _pct("Max Drawdown", "max_drawdown", "peak-to-trough"),
    _pct("Volatility", "volatility", "annualised", tone=False),
    _num("Calmar", "calmar", "CAGR / max DD"),
    _num("Smart Sharpe", "smart_sharpe", "autocorr-adjusted"),
    _num("Omega", "omega", "gain / pain"),
    _num("Recovery Factor", "recovery_factor", "net profit / max DD"),
    _num("Ulcer Index", "ulcer_index", "depth &#215; duration"),
    _pct("Time in Market", "exposure", "exposure", tone=False),
)
PERIODS = (
    _pct("MTD", data.month_to_date, definition=_MTD),
    _pct("YTD", data.year_to_date, definition=_YTD),
    _pct("3 Month", data.horizon("3M", "total_return"), definition=_TRAILING),
    _pct("6 Month", data.horizon("6M", "total_return"), definition=_TRAILING),
    _pct("1 Year", data.horizon("1Y", "total_return"), definition=_TRAILING),
    _pct("3Y (ann.)", data.horizon("3Y", "cagr"), definition=_TRAILING_CAGR),
    _pct("5Y (ann.)", data.horizon("5Y", "cagr"), definition=_TRAILING_CAGR),
    _pct("All-time", "cagr"),
    _pct("Best Year", "best_year"),
    _pct("Worst Year", "worst_year"),
)
RISK = (
    _pct("Max Drawdown", "max_drawdown", "peak-to-trough"),
    _pct("Conditional DD", "conditional_drawdown", "worst 5% mean"),
    Stat("Longest DD", "longest_drawdown_days", "int", "days underwater"),
    _pct("Session VaR", "value_at_risk", "95% 1-session"),
    _pct("Expected Shortfall", "expected_shortfall", "cVaR"),
    _pct("Volatility", "volatility", "annualised", tone=False),
    _pct("Risk of Ruin", "risk_of_ruin", "probability", tone=False),
    _num("Skew", "skew", "asymmetry"),
    _num("Kurtosis", "kurtosis", "excess, tailedness"),
    _num("Tail Ratio", "tail_ratio", "right / left"),
)
BENCHMARK = (
    _num("Beta", "beta", "market sensitivity"),
    _num("Alpha", "alpha", "excess return"),
    _num("Correlation", "correlation", "to benchmark"),
    _num("R&#178;", "r_squared", "explained variance"),
    _num("Information Ratio", "information_ratio", "active / tracking err"),
    _num("Treynor Ratio", "treynor_ratio", "return / beta"),
)
MONTHLY = (
    _pct("Best Month", "best_month"),
    _pct("Worst Month", "worst_month"),
    _num("Gain / Pain", "gain_to_pain", "monthly"),
    _pct("Win Days", "session_win_rate", "positive", tone=False),
    _pct("Avg Win Day", "average_winning_session", "per session"),
    _pct("Avg Loss Day", "average_losing_session", "per session"),
)
DISTRIBUTION = (
    _pct("Best Day", "best_session"),
    _pct("Worst Day", "worst_session"),
    _num("Skew", "skew", "asymmetry"),
    _num("Kurtosis", "kurtosis", "excess, tailedness"),
    _num("Outlier Win", "outlier_win_ratio", "99th pctile"),
    _num("Outlier Loss", "outlier_loss_ratio", "1st pctile"),
)
TRADES = (
    Stat("Trades", "trades", "int", "closed"),
    _pct("Win Rate", "win_rate", "of closed trades", tone=False),
    _num("Profit Factor", "profit_factor", "gross win / loss"),
    _num("Payoff Ratio", "payoff_ratio", "avg win / avg loss"),
    Stat("Expectancy", "expectancy", "rupees", "per trade", tone=True),
    Stat("Avg Holding", "average_holding_seconds", "duration", "entry to exit"),
    Stat("Avg Win", "average_win", "rupees", "net, per trade", tone=True),
    Stat("Avg Loss", "average_loss", "rupees", "net, per trade", tone=True),
    Stat("Largest Win", "largest_win", "rupees", "net", tone=True),
    Stat("Largest Loss", "largest_loss", "rupees", "net", tone=True),
    Stat("Charges", "total_costs", "rupees", "of closed trades"),
    Stat("Net P&amp;L", "net_pnl", "rupees", "after charges", tone=True),
)


def figure(stat: Stat, performance: Performance) -> float:
    """The tile's figure from `performance`; NaN when the metric is absent."""
    if isinstance(stat.figure, str):
        value = performance.metrics.get(stat.figure)
        return float("nan") if value is None else value
    return stat.figure(performance)


def text(value: object, shown: str) -> str:
    """A figure as the tile shows it ("n/a" for an undefined one)."""
    if fmt.is_missing(value):
        return fmt.MISSING
    number = float(value)
    if shown == "pct":
        return fmt.percent(number)
    if shown == "int":
        return fmt.number(number, 0)
    if shown == "rupees":
        return fmt.rupees(number, 0)
    if shown == "duration":
        return fmt.duration(number)
    return fmt.number(number)


def tone(value: object) -> str:
    """The CSS class of a value coloured by its sign: gains, losses, or none."""
    if fmt.is_missing(value) or float(value) == 0:
        return ""
    return " pos" if float(value) > 0 else " neg"


def _comparable(stat: Stat) -> bool:
    return not isinstance(stat.figure, str) or stat.figure in data.COMPARABLE


def _definition(stat: Stat) -> str:
    if isinstance(stat.figure, str):
        return fmt.text(DEFINITIONS[stat.figure].definition)
    return fmt.text(stat.definition)


def tile(stat: Stat, performance: Performance, bench: Performance | None, name: str) -> str:
    """One tile; `bench` is the benchmark measured as the strategy is, `name` its
    name."""
    value = figure(stat, performance)
    shown = text(value, stat.shown)
    colour = tone(value) if stat.tone else ""
    sub = stat.description
    if bench is not None and _comparable(stat):
        sub = f"{fmt.text(name)} {text(figure(stat, bench), stat.shown)}"
    line = f"<div class='stat-sub nums'>{sub}</div>" if sub else ""
    return (
        f"<div class='stat'><div class='eyebrow trunc' title='{_definition(stat)}'>"
        f"{stat.label}</div><div class='stat-value nums{colour}'>{shown}</div>{line}</div>"
    )


def grid(
    stats: Sequence[Stat], performance: Performance, bench: Performance | None, name: str
) -> str:
    """A grid of tiles."""
    tiles = "".join(tile(stat, performance, bench, name) for stat in stats)
    return f"<div class='stats'>{tiles}</div>"


__all__ = [
    "BENCHMARK",
    "DISTRIBUTION",
    "MONTHLY",
    "OVERVIEW",
    "PERIODS",
    "RISK",
    "TRADES",
    "Stat",
    "figure",
    "grid",
    "text",
    "tile",
    "tone",
]
