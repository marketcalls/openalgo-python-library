"""Corrected and drawdown-based ratios: smart, adjusted and probabilistic Sharpe and
Sortino, the autocorrelation penalty, the risk-return ratio, the ulcer, ulcer
performance and serenity indexes, the recovery factor, return over exposure, and the
benchmark's R squared, tracking error, information ratio and Treynor ratio.

Expected values are the published formulas (cited in `openalgo.backtest.metrics`)
applied to the literal returns with the standard library. Sharpe, Sortino, CAGR and
sessions per year are inputs here: their own definitions are tested elsewhere.
"""

import math
import statistics
from statistics import NormalDist

import pandas as pd
import pytest

from openalgo.backtest import metrics

IST = "Asia/Kolkata"
RETURNS = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.0]


def closes(returns: list[float], start: str = "2025-06-02") -> pd.Series:
    values, level = [], 100.0
    for r in returns:
        level *= 1 + r
        values.append(level)
    days = pd.bdate_range(start, periods=len(returns), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


def analyse(returns: list[float], **kwargs) -> dict:
    return metrics.analyse(closes(returns), exchanges="NSE", opening_wealth=100.0, **kwargs).metrics


def lag_one_autocorrelation(values: list[float]) -> float:
    mean = statistics.mean(values)
    spread = sum((v - mean) ** 2 for v in values)
    return (
        sum((a - mean) * (b - mean) for a, b in zip(values[:-1], values[1:], strict=True)) / spread
    )


def moment_ratios(values: list[float]) -> tuple[float, float]:
    n, mean = len(values), statistics.mean(values)
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m3 / m2**1.5, m4 / m2**2 - 3


def test_smart_ratios_divide_by_the_autocorrelation_penalty():
    m = analyse(RETURNS)
    rho = lag_one_autocorrelation(RETURNS)
    q = round(m["sessions_per_year"])
    penalty = math.sqrt(1 + 2 * sum((1 - k / q) * rho**k for k in range(1, q)))
    assert m["autocorrelation"] == pytest.approx(rho)
    assert m["autocorrelation_penalty"] == pytest.approx(penalty)
    assert m["smart_sharpe"] == pytest.approx(m["sharpe"] / penalty)
    assert m["smart_sortino"] == pytest.approx(m["sortino"] / penalty)


def test_the_adjusted_sortino_is_the_sortino_over_root_two():
    m = analyse(RETURNS)
    assert m["adjusted_sortino"] == pytest.approx(m["sortino"] / math.sqrt(2))


def test_the_probabilistic_sharpe_follows_bailey_and_lopez_de_prado():
    m = analyse(RETURNS)
    n = len(RETURNS)
    g1, g2 = moment_ratios(RETURNS)

    def probability(annual: float) -> float:
        s = annual / math.sqrt(m["sessions_per_year"])
        z = s * math.sqrt(n - 1) / math.sqrt(1 - g1 * s + (g2 + 2) / 4 * s * s)
        return NormalDist().cdf(z)

    assert m["probabilistic_sharpe"] == pytest.approx(probability(m["sharpe"]))
    assert m["probabilistic_sortino"] == pytest.approx(probability(m["sortino"]))
    assert m["probabilistic_adjusted_sortino"] == pytest.approx(probability(m["adjusted_sortino"]))


def test_the_risk_return_ratio_is_the_unannualised_mean_over_deviation():
    m = analyse(RETURNS, risk_free_rate=0.065)
    expected = statistics.mean(RETURNS) / statistics.stdev(RETURNS)
    assert m["risk_return_ratio"] == pytest.approx(expected)


def close_drawdowns(returns: list[float]) -> list[float]:
    wealth, peak, out = 1.0, 1.0, []
    for r in returns:
        wealth *= 1 + r
        peak = max(peak, wealth)
        out.append(wealth / peak - 1)
    return out


# Six and a half weeks of sessions: long enough for CAGR (28 days or more).
LONG = RETURNS * 4


def test_the_ulcer_index_and_its_performance_index():
    m = analyse(LONG, risk_free_rate=0.05)
    drawdowns = close_drawdowns(LONG)
    ulcer = math.sqrt(sum(d * d for d in drawdowns) / len(drawdowns))
    assert m["ulcer_index"] == pytest.approx(ulcer)
    assert math.isfinite(m["cagr"])
    assert m["ulcer_performance_index"] == pytest.approx((m["cagr"] - 0.05) / ulcer)


def test_the_serenity_index_divides_by_ulcer_index_times_pitfall():
    m = analyse(LONG, risk_free_rate=0.05)
    ordered = sorted(close_drawdowns(LONG))
    # 32 drawdowns: the 5th percentile sits at position 0.05 x 31 = 1.55.
    cutoff = ordered[1] + 0.55 * (ordered[2] - ordered[1])
    cdar = statistics.mean(d for d in ordered if d <= cutoff)
    assert m["conditional_drawdown"] == pytest.approx(cdar)
    pitfall = -cdar / m["volatility"]
    expected = (m["cagr"] - 0.05) / (m["ulcer_index"] * pitfall)
    assert m["serenity_index"] == pytest.approx(expected)


def test_the_serenity_definition_cites_the_published_ratio_by_its_name_alone():
    definition = metrics.DEFINITIONS["serenity_index"].definition
    assert definition.endswith(
        "-conditional drawdown / annualised volatility (the published serenity ratio)."
    )


def test_the_recovery_factor_is_net_profit_over_the_drawdown_amount():
    m = analyse(RETURNS)
    assert m["recovery_factor"] == pytest.approx(m["net_profit"] / -m["max_drawdown_amount"])
    assert analyse([0.01, 0.02])["recovery_factor"] == math.inf


def test_return_over_exposure_is_undefined_for_a_run_never_in_the_market():
    assert math.isnan(analyse(RETURNS)["risk_adjusted_return"])


def test_a_flat_curve_has_no_corrected_ratio():
    m = analyse([0.0] * 5)
    for key in ("smart_sharpe", "probabilistic_sharpe", "autocorrelation", "risk_return_ratio"):
        assert math.isnan(m[key]), key
    assert m["ulcer_index"] == 0.0
    assert math.isnan(m["ulcer_performance_index"])


def test_a_single_session_has_no_corrected_ratio():
    m = analyse([0.01])
    for key in ("smart_sharpe", "adjusted_sortino", "probabilistic_sharpe", "autocorrelation"):
        assert math.isnan(m[key]), key


# ----- benchmark --------------------------------------------------------------------


def test_active_return_figures_against_a_benchmark():
    strategy = [0.02, -0.04, 0.06, 0.01]
    benchmark = [0.01, -0.02, 0.03, 0.02]
    equity = closes(strategy)
    levels, level = [1000.0], 1000.0
    for b in benchmark:
        level *= 1 + b
        levels.append(level)
    index = [equity.index[0] - pd.Timedelta(days=3), *equity.index]
    m = metrics.analyse(
        equity,
        exchanges="NSE",
        opening_wealth=100.0,
        benchmark=pd.Series(levels, index=index),
        risk_free_rate=0.05,
    ).metrics
    n_year = m["sessions_per_year"]
    active = [s - b for s, b in zip(strategy, benchmark, strict=True)]
    assert m["r_squared"] == pytest.approx(statistics.correlation(strategy, benchmark) ** 2)
    assert m["tracking_error"] == pytest.approx(statistics.stdev(active) * math.sqrt(n_year))
    expected_ir = statistics.mean(active) / statistics.stdev(active) * math.sqrt(n_year)
    assert m["information_ratio"] == pytest.approx(expected_ir)
    rf = m["risk_free_per_session"]
    beta = statistics.covariance(strategy, benchmark) / statistics.variance(benchmark)
    assert m["treynor_ratio"] == pytest.approx((statistics.mean(strategy) - rf) * n_year / beta)
