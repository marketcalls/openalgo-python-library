"""The distribution of session returns: averages, best and worst periods, winning and
losing sessions, skew, kurtosis, value at risk, expected shortfall, tail and outlier
ratios, Omega and gain to pain.

Each equity series opens at 100 rupees (the opening wealth) and closes once per NSE
session from Monday 2 June 2025, compounding the session returns the test gives, so
every expected value is the published formula applied to those returns.
"""

import math
import statistics

import pandas as pd
import pytest

from openalgo.backtest import metrics

IST = "Asia/Kolkata"
RETURNS = [0.02, -0.01, 0.03, -0.02, 0.01]


def closes(returns: list[float], start: str = "2025-06-02") -> pd.Series:
    """Equity from 100 compounding `returns`, one 15:30 close per weekday."""
    values, level = [], 100.0
    for r in returns:
        level *= 1 + r
        values.append(level)
    days = pd.bdate_range(start, periods=len(returns), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


def analyse(returns: list[float], **kwargs) -> dict:
    return metrics.analyse(closes(returns), exchanges="NSE", opening_wealth=100.0, **kwargs).metrics


def quantile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation between order statistics at (n - 1) p."""
    position = p * (len(sorted_values) - 1)
    below = math.floor(position)
    above = min(below + 1, len(sorted_values) - 1)
    return sorted_values[below] + (position - below) * (sorted_values[above] - sorted_values[below])


def test_expected_return_is_the_geometric_mean_and_average_return_the_arithmetic():
    m = analyse(RETURNS)
    growth = math.prod(1 + r for r in RETURNS)
    assert m["expected_return"] == pytest.approx(growth ** (1 / 5) - 1)
    assert m["average_session_return"] == pytest.approx(statistics.mean(RETURNS))


def test_best_and_worst_session_month_and_year():
    m = analyse(RETURNS)
    assert (m["best_session"], m["worst_session"]) == (pytest.approx(0.03), pytest.approx(-0.02))
    # One June 2025: the month and the year are the whole run.
    june = math.prod(1 + r for r in RETURNS) - 1
    for key in ("best_month", "worst_month", "best_year", "worst_year"):
        assert m[key] == pytest.approx(june), key


def across_june_and_july(returns: list[float]) -> dict:
    """Two sessions in June (Friday 27 and Monday 30) and two in July (1 and 2)."""
    equity = closes(returns, start="2025-06-27")
    return metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0).metrics


def test_months_are_ranked_by_their_compounded_return():
    m = across_june_and_july([0.05, -0.02, -0.03, 0.01])
    june, july = 1.05 * 0.98 - 1, 0.97 * 1.01 - 1
    assert m["best_month"] == pytest.approx(june)
    assert m["worst_month"] == pytest.approx(july)


def test_winning_and_losing_sessions():
    m = analyse(RETURNS)
    assert (m["winning_sessions"], m["losing_sessions"]) == (3, 2)
    assert m["session_win_rate"] == pytest.approx(0.6)
    assert m["average_winning_session"] == pytest.approx(0.02)
    assert m["average_losing_session"] == pytest.approx(-0.015)
    assert (m["max_consecutive_winning_sessions"], m["max_consecutive_losing_sessions"]) == (1, 1)


def test_a_flat_session_breaks_a_run_and_is_not_counted_in_the_win_rate():
    m = analyse([0.01, 0.01, 0.0, -0.01, -0.01, -0.01])
    assert m["session_win_rate"] == pytest.approx(2 / 5)
    assert (m["max_consecutive_winning_sessions"], m["max_consecutive_losing_sessions"]) == (2, 3)


def test_skew_and_kurtosis_are_the_adjusted_sample_estimators():
    n, mean = len(RETURNS), statistics.mean(RETURNS)
    m2 = sum((r - mean) ** 2 for r in RETURNS) / n
    m3 = sum((r - mean) ** 3 for r in RETURNS) / n
    m4 = sum((r - mean) ** 4 for r in RETURNS) / n
    g1, g2 = m3 / m2**1.5, m4 / m2**2 - 3
    m = analyse(RETURNS)
    assert m["skew"] == pytest.approx(g1 * math.sqrt(n * (n - 1)) / (n - 2))
    assert m["kurtosis"] == pytest.approx(((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)))


def test_value_at_risk_and_expected_shortfall_read_the_left_tail():
    ordered = sorted(RETURNS)
    m = analyse(RETURNS)
    # The 5th percentile sits at position 0.2, between -0.02 and -0.01.
    assert m["value_at_risk"] == pytest.approx(-0.018)
    assert m["value_at_risk"] == pytest.approx(quantile(ordered, 0.05))
    # Only -0.02 is at or below it.
    assert m["expected_shortfall"] == pytest.approx(-0.02)


def test_tail_and_outlier_ratios():
    ordered = sorted(RETURNS)
    m = analyse(RETURNS)
    assert m["tail_ratio"] == pytest.approx(quantile(ordered, 0.95) / 0.018)
    assert m["outlier_win_ratio"] == pytest.approx(quantile(ordered, 0.99) / 0.02)
    assert m["outlier_loss_ratio"] == pytest.approx(quantile(ordered, 0.01) / -0.015)


def test_omega_weighs_gains_over_the_risk_free_rate_against_shortfalls():
    assert analyse(RETURNS)["omega"] == pytest.approx(0.06 / 0.03)
    rate = 0.065
    m = analyse(RETURNS, risk_free_rate=rate)
    rf = m["risk_free_per_session"]
    gains = sum(max(r - rf, 0.0) for r in RETURNS)
    shortfalls = sum(max(rf - r, 0.0) for r in RETURNS)
    assert m["omega"] == pytest.approx(gains / shortfalls)


def test_gain_to_pain_sums_the_months_over_the_losing_ones():
    # June gains 1.05 x 0.98 - 1, July loses 0.97 x 1.01 - 1.
    m = across_june_and_july([0.05, -0.02, -0.03, 0.01])
    june, july = 1.05 * 0.98 - 1, 0.97 * 1.01 - 1
    assert m["gain_to_pain"] == pytest.approx((june + july) / abs(july))
    # A single winning month has no pain.
    assert analyse(RETURNS)["gain_to_pain"] == math.inf


# ----- edge cases ------------------------------------------------------------------


def test_a_single_session_is_its_own_average_and_tail():
    m = analyse([0.01])
    assert m["expected_return"] == pytest.approx(0.01)
    assert m["value_at_risk"] == pytest.approx(0.01)
    assert m["session_win_rate"] == 1.0
    assert math.isnan(m["skew"]) and math.isnan(m["kurtosis"])


def test_a_flat_curve_has_no_win_rate_tail_or_omega():
    m = analyse([0.0, 0.0, 0.0])
    assert (m["value_at_risk"], m["expected_shortfall"], m["best_session"]) == (0.0, 0.0, 0.0)
    for key in ("session_win_rate", "tail_ratio", "omega", "skew", "outlier_win_ratio"):
        assert math.isnan(m[key]), key


def test_all_winning_sessions():
    m = analyse([0.01, 0.02, 0.03])
    assert m["session_win_rate"] == 1.0 and m["losing_sessions"] == 0
    assert m["omega"] == math.inf
    assert math.isnan(m["average_losing_session"]) and math.isnan(m["outlier_loss_ratio"])


def test_all_losing_sessions():
    m = analyse([-0.01, -0.02, -0.03])
    assert m["session_win_rate"] == 0.0 and m["max_consecutive_losing_sessions"] == 3
    assert m["expected_shortfall"] == pytest.approx(-0.03)
    # No gain over the threshold: Omega is zero.
    assert m["omega"] == 0.0
    assert math.isnan(m["outlier_win_ratio"])
