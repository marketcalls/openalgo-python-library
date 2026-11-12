"""Figures beyond the core table: the distribution of session returns, its tails, the
corrected and drawdown-based ratios, the ratios of trades, and the composites of
figures from several inputs. The kernel computes every statistic; the composites here
divide figures the kernel returned, with its convention for a zero denominator."""

import math

import numpy as np
import pandas as pd

from ._inputs import TradeTable
from ._kernel import call

# Metrics keys of the session statistics, by the kernel's name where it differs.
_SESSION_KEYS = {
    "expected_return": "expected_return",
    "average_session_return": "average_return",
    "best_session": "best",
    "worst_session": "worst",
    "winning_sessions": "winning",
    "losing_sessions": "losing",
    "session_win_rate": "win_rate",
    "average_winning_session": "average_winning",
    "average_losing_session": "average_losing",
    "max_consecutive_winning_sessions": "max_consecutive_wins",
    "max_consecutive_losing_sessions": "max_consecutive_losses",
}
# Metrics keys the kernel returns under the same name.
_SAME_NAME = (
    "smart_sharpe",
    "smart_sortino",
    "adjusted_sortino",
    "probabilistic_sharpe",
    "probabilistic_sortino",
    "probabilistic_adjusted_sortino",
    "autocorrelation",
    "autocorrelation_penalty",
    "omega",
    "risk_return_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "serenity_index",
    "skew",
    "kurtosis",
    "value_at_risk",
    "expected_shortfall",
    "conditional_drawdown",
    "tail_ratio",
    "outlier_win_ratio",
    "outlier_loss_ratio",
)
_TRADE_RATIO_KEYS = ("win_loss_ratio", "kelly_criterion", "risk_of_ruin", "cpc_index")


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator for a denominator of zero or more, as the kernel
    divides: infinite over zero when the numerator is positive, NaN when it is not, and
    NaN for a NaN input."""
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else math.nan


def _extreme(values: np.ndarray, pick) -> float:
    known = values[np.isfinite(values)]
    return float(pick(known)) if len(known) else math.nan


def session_figures(
    returns: pd.Series,
    sessions_per_year: float,
    rate: float,
    cagr: float,
    monthly: pd.DataFrame,
    yearly: pd.Series,
) -> tuple[dict, list[int]]:
    """(figures, losing streak lengths) of the session returns in date order, with the
    best and worst month and year and the gain to pain of the calendar months."""
    stats = call("sessions", returns.to_numpy(dtype=float), sessions_per_year, rate, cagr)
    figures = {key: stats[name] for key, name in _SESSION_KEYS.items()}
    figures |= {key: stats[key] for key in _SAME_NAME}
    # Row by row: the months in calendar order.
    months = monthly.to_numpy(dtype=float).ravel()
    years = yearly.to_numpy(dtype=float)
    figures["gain_to_pain"] = call("gain_to_pain", months[np.isfinite(months)])
    figures["best_month"] = _extreme(months, np.max)
    figures["worst_month"] = _extreme(months, np.min)
    figures["best_year"] = _extreme(years, np.max)
    figures["worst_year"] = _extreme(years, np.min)
    return figures, list(stats["loss_streaks"])


def trade_ratio_figures(table: TradeTable, capital: float) -> tuple[dict, list[int]]:
    """(figures, losing streak lengths in exit order) of the closed trades, with the
    risk of ruin measured from `capital` rupees."""
    closed = table.closed
    stats = call(
        "trade_ratios",
        closed(table.entry_ts),
        closed(table.exit_ts),
        closed(table.gross_pnl),
        closed(table.net_pnl),
        closed(table.costs),
        capital,
    )
    return {key: stats[key] for key in _TRADE_RATIO_KEYS}, list(stats["loss_streaks"])


def composite_figures(figures: dict) -> dict:
    """Figures that combine the others: recovery factor, return over exposure and the
    common sense ratio (definitions in `DEFINITIONS`). Return over exposure is NaN for
    a run never in the market: its CAGR owes nothing to time in a position."""
    exposure = figures["exposure"]
    return {
        "recovery_factor": ratio(figures["net_profit"], abs(figures["max_drawdown_amount"])),
        "risk_adjusted_return": figures["cagr"] / exposure if exposure > 0 else math.nan,
        "common_sense_ratio": figures["profit_factor"] * figures["tail_ratio"],
    }


__all__ = ["composite_figures", "ratio", "session_figures", "trade_ratio_figures"]
