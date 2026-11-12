"""The metrics added to the core table of `_definitions`: corrected and drawdown-based
ratios, the distribution of session returns, its tails, ratios of trades, and the
active-return figures of a benchmark.

Each entry is (label, unit, definition), as in `_definitions`; the source of every
definition is cited in the native kernel's module docs (`oa_bt::metrics`).
"""

RISK_ADJUSTED = {
    "smart_sharpe": (
        "Smart Sharpe",
        "ratio",
        "Sharpe ratio divided by the autocorrelation penalty (Lo 2002): annualising by "
        "sqrt(sessions per year) assumes independent sessions.",
    ),
    "smart_sortino": ("Smart Sortino", "ratio", "Sortino ratio divided by the same penalty."),
    "adjusted_sortino": (
        "Adjusted Sortino",
        "ratio",
        "Sortino ratio / sqrt(2) (Schwager), to read on the Sharpe ratio's scale.",
    ),
    "probabilistic_sharpe": (
        "Probabilistic Sharpe",
        "fraction",
        "Probability that the true Sharpe ratio is above 0 (Bailey and Lopez de Prado "
        "2012): Phi(S sqrt(n - 1) / sqrt(1 - g1 S + (g2 + 2) / 4 S^2)) with S the "
        "per-session ratio, n sessions, g1 and g2 the skewness and excess kurtosis "
        "moment ratios.",
    ),
    "probabilistic_sortino": (
        "Probabilistic Sortino",
        "fraction",
        "The same probability for the per-session Sortino ratio.",
    ),
    "probabilistic_adjusted_sortino": (
        "Probabilistic adjusted Sortino",
        "fraction",
        "The same probability for the per-session adjusted Sortino ratio.",
    ),
    "autocorrelation": (
        "Autocorrelation",
        "ratio",
        "Lag-1 sample autocorrelation of session returns. NaN below three sessions.",
    ),
    "autocorrelation_penalty": (
        "Autocorrelation penalty",
        "ratio",
        "sqrt(1 + 2 sum((1 - k/q) rho^k, k = 1..q-1)), rho the autocorrelation and q the "
        "sessions per year rounded; 1 for independent sessions.",
    ),
    "omega": (
        "Omega ratio",
        "ratio",
        "Sum of session returns above the per-session risk-free rate over the sum of the "
        "shortfalls below it (Keating and Shadwick 2002).",
    ),
    "gain_to_pain": (
        "Gain to pain",
        "ratio",
        "Sum of monthly returns over the magnitude of the sum of the losing months (Schwager).",
    ),
    "risk_return_ratio": (
        "Risk-return ratio",
        "ratio",
        "Mean session return over its sample standard deviation, per session, with no "
        "risk-free rate.",
    ),
    "risk_adjusted_return": (
        "Return over exposure",
        "fraction",
        "CAGR divided by exposure: the yearly growth per unit of time in the market.",
    ),
    "ulcer_index": (
        "Ulcer index",
        "fraction",
        "Root mean square of the drawdowns at session closes (Martin and McCann 1989); "
        "zero or positive.",
    ),
    "ulcer_performance_index": (
        "Ulcer performance index",
        "ratio",
        "(CAGR - risk-free rate) / ulcer index: the Martin ratio.",
    ),
    "serenity_index": (
        "Serenity index",
        "ratio",
        "(CAGR - risk-free rate) / (ulcer index x pitfall), with the pitfall indicator "
        "-conditional drawdown / annualised volatility (the serenity ratio of KeyQuant).",
    ),
    "recovery_factor": (
        "Recovery factor",
        "ratio",
        "Net profit over the magnitude of the maximum drawdown amount, both in rupees.",
    ),
}

SESSIONS = {
    "expected_return": (
        "Expected return",
        "fraction",
        "Geometric mean session return: (product of (1 + r)) ^ (1 / sessions) - 1, the "
        "per-session rate that compounds to the total return.",
    ),
    "average_session_return": (
        "Average session return",
        "fraction",
        "Arithmetic mean of the session returns, flat sessions included.",
    ),
    "best_session": ("Best session", "fraction", "Highest session return."),
    "worst_session": ("Worst session", "fraction", "Lowest session return."),
    "best_month": (
        "Best month",
        "fraction",
        "Highest compounded calendar-month return (a first or last month may be partial).",
    ),
    "worst_month": ("Worst month", "fraction", "Lowest compounded calendar-month return."),
    "best_year": ("Best year", "fraction", "Highest compounded calendar-year return."),
    "worst_year": ("Worst year", "fraction", "Lowest compounded calendar-year return."),
    "winning_sessions": ("Winning sessions", "count", "Sessions with a return above zero."),
    "losing_sessions": ("Losing sessions", "count", "Sessions with a return below zero."),
    "session_win_rate": (
        "Session win rate",
        "fraction",
        "Winning sessions over the sessions that won or lost; flat sessions are left out.",
    ),
    "average_winning_session": (
        "Average winning session",
        "fraction",
        "Mean return of the winning sessions.",
    ),
    "average_losing_session": (
        "Average losing session",
        "fraction",
        "Mean return of the losing sessions (negative).",
    ),
    "max_consecutive_winning_sessions": (
        "Consecutive winning sessions",
        "count",
        "Longest run of winning sessions; a flat session breaks a run.",
    ),
    "max_consecutive_losing_sessions": (
        "Consecutive losing sessions",
        "count",
        "Longest run of losing sessions.",
    ),
    "rolling_window_sessions": (
        "Rolling window",
        "count",
        "Sessions in each window of the rolling statistics.",
    ),
}

TAIL = {
    "skew": (
        "Skew",
        "ratio",
        "Adjusted Fisher-Pearson skewness G1 of session returns (Joanes and Gill 1998). "
        "NaN below three sessions.",
    ),
    "kurtosis": (
        "Excess kurtosis",
        "ratio",
        "Sample excess kurtosis G2 of session returns (Joanes and Gill 1998); about 0 for "
        "normal returns. NaN below four sessions.",
    ),
    "value_at_risk": (
        "Value at risk (95%)",
        "fraction",
        "Historical: the 5th percentile of session returns (linear interpolation between "
        "order statistics). A loss is negative.",
    ),
    "expected_shortfall": (
        "Expected shortfall (CVaR 95%)",
        "fraction",
        "Mean of the session returns at or below the value at risk (Rockafellar and Uryasev 2002).",
    ),
    "conditional_drawdown": (
        "Conditional drawdown (95%)",
        "fraction",
        "Mean of the drawdowns at session closes at or below their 5th percentile "
        "(Chekhlov, Uryasev and Zabarankin 2005). Zero or negative.",
    ),
    "tail_ratio": (
        "Tail ratio",
        "ratio",
        "95th percentile of session returns over the magnitude of the 5th.",
    ),
    "outlier_win_ratio": (
        "Outlier win ratio",
        "ratio",
        "99th percentile of session returns over the mean winning session.",
    ),
    "outlier_loss_ratio": (
        "Outlier loss ratio",
        "ratio",
        "1st percentile of session returns over the mean losing session.",
    ),
}

TRADE_RATIOS = {
    "win_loss_ratio": (
        "Win/loss ratio",
        "ratio",
        "Winning trades over losing trades (counts).",
    ),
    "kelly_criterion": (
        "Kelly criterion",
        "fraction",
        "p - (1 - p) / payoff ratio, p the share of wins among trades that won or lost "
        "(Kelly 1956): the stake per trade, as a share of capital, that maximises growth.",
    ),
    "risk_of_ruin": (
        "Risk of ruin",
        "fraction",
        "Lundberg bound exp(-g x start equity) on losing the capital, trades drawn "
        "independently from the run's net P&L at fixed size; g > 0 solves "
        "mean(exp(-g x net P&L)) = 1. 1 when the mean trade is not positive.",
    ),
    "cpc_index": (
        "CPC index",
        "ratio",
        "Profit factor x win rate x payoff ratio.",
    ),
    "common_sense_ratio": (
        "Common sense ratio",
        "ratio",
        "Profit factor (trades) x tail ratio (session returns).",
    ),
}

BENCHMARK_EXTRA = {
    "r_squared": (
        "R squared",
        "ratio",
        "Square of the correlation: the share of the strategy's session variance the "
        "benchmark explains.",
    ),
    "tracking_error": (
        "Tracking error",
        "fraction",
        "Sample standard deviation of (strategy - benchmark) session returns, times "
        "sqrt(sessions per year).",
    ),
    "information_ratio": (
        "Information ratio",
        "ratio",
        "Mean active session return over its standard deviation, times sqrt(sessions per "
        "year) (Grinold and Kahn).",
    ),
    "treynor_ratio": (
        "Treynor ratio",
        "ratio",
        "Mean excess session return x sessions per year, over beta (Treynor 1965).",
    ),
}

__all__ = ["BENCHMARK_EXTRA", "RISK_ADJUSTED", "SESSIONS", "TAIL", "TRADE_RATIOS"]
