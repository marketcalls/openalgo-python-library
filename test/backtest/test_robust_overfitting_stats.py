"""The optimizer's overfitting statistics (design 17, section 5), on hand-built returns:
the deflated Sharpe ratio of the best for the number of combinations tried, and the
probability of backtest overfitting by combinatorially symmetric cross-validation."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from openalgo.backtest._robust._deflated_sharpe import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_best,
    moment_ratios,
    sharpe,
)
from openalgo.backtest._robust._overfitting_probability import blocks_for, overfitting

NORMAL = NormalDist()


def test_a_sharpe_ratio_is_the_mean_over_the_sample_deviation_per_session():
    # Mean 0.02; deviations -0.01 and +0.01 over n - 1 = 1: 0.01414; 0.02 / 0.01414.
    assert sharpe(np.array([0.01, 0.03])) == pytest.approx(math.sqrt(2))


def test_returns_that_never_vary_have_no_sharpe_ratio():
    assert math.isnan(sharpe(np.zeros(5)))


def test_symmetric_returns_have_no_skew():
    # -1, 0, 1: m2 = 2/3, m3 = 0, m4 = 2/3; excess kurtosis (2/3) / (4/9) - 3 = -1.5.
    g1, g2 = moment_ratios(np.array([-1.0, 0.0, 1.0]))
    assert g1 == pytest.approx(0.0) and g2 == pytest.approx(-1.5)


def test_the_best_of_two_tries_shows_a_sharpe_ratio_by_luck_alone():
    # sqrt(1) ((1 - g) Z(1/2) + g Z(1 - 1/(2e))): Z(1/2) is 0.
    expected = EULER_GAMMA * NORMAL.inv_cdf(1 - 1 / (2 * math.e))
    assert expected_best(1.0, 2) == pytest.approx(expected)
    assert expected_best(1.0, 2) == pytest.approx(0.5772 * 0.9007, abs=1e-3)


def test_more_tries_raise_the_sharpe_ratio_luck_alone_shows():
    # 1,000 tries: 0.4228 x Z(0.999) + 0.5772 x Z(1 - 1/(1000 e)) = 0.4228 x 3.0902 +
    # 0.5772 x Z(0.99963) = 0.4228 x 3.0902 + 0.5772 x 3.3747, about 3.255 standard
    # deviations; a quarter of the variance halves it.
    assert expected_best(1.0, 1_000) == pytest.approx(3.255, abs=0.002)
    assert expected_best(0.25, 1_000) == pytest.approx(3.255 / 2, abs=0.001)


def test_one_try_has_nothing_to_deflate():
    assert math.isnan(expected_best(1.0, 1))
    single = deflated_sharpe(np.array([0.01, -0.005, 0.02]), [np.array([0.01, -0.005, 0.02])])
    assert math.isnan(single.deflated)


def _returns(mean: float, spread: float, sessions: int) -> np.ndarray:
    """Returns alternating mean + spread and mean - spread: symmetric, so g1 = 0 and g2 =
    -2 (m2 = spread^2, m4 = spread^4)."""
    return mean + spread * np.where(np.arange(sessions) % 2 == 0, 1.0, -1.0)


def test_the_deflated_sharpe_is_the_probabilistic_sharpe_against_the_luck_of_the_tries():
    best = _returns(0.002, 0.01, 100)
    other = _returns(0.0, 0.01, 100)
    found = deflated_sharpe(best, [best, other])
    # Sharpe 0.002 / (0.01 sqrt(100/99)); the tries' Sharpe ratios 0 and that, variance
    # half its square; with g1 = 0, g2 = -2 the variance term is 1 + 0 x S^2 = 1.
    per_session = 0.002 / (0.01 * math.sqrt(100 / 99))
    variance = per_session**2 / 2
    expected = expected_best(variance, 2)
    assert found.best_sharpe == pytest.approx(per_session)
    assert found.variance == pytest.approx(variance)
    assert found.expected == pytest.approx(expected)
    assert found.deflated == pytest.approx(NORMAL.cdf((per_session - expected) * math.sqrt(99)))


def test_the_same_best_among_more_tries_is_deflated_more():
    best = _returns(0.002, 0.01, 250)
    tries = [_returns(m, 0.01, 250) for m in (-0.001, 0.0, 0.001)]
    few = deflated_sharpe(best, [best, *tries])
    many = deflated_sharpe(best, [best, *tries * 30])
    assert many.trials == 91 and many.expected > few.expected
    assert many.deflated < few.deflated


# ----- the probability of backtest overfitting --------------------------------------------


def test_the_sessions_are_cut_into_an_even_number_of_blocks_of_two_or_more():
    assert blocks_for(1_000) == 16
    assert blocks_for(20) == 10
    assert blocks_for(9) == 4
    assert blocks_for(7) == 0


def _blocks(means: list[float]) -> np.ndarray:
    """Two sessions a block, mean m: m + 0.5 and m - 0.5, in thousandths."""
    return np.array([[m + 0.5, m - 0.5] for m in means]).ravel() / 1_000


def test_a_combination_better_in_every_block_never_overfits():
    # A is B plus a thousandth every session: best in sample and best out of sample in
    # each of the C(4, 2) = 6 splits, rank 2 of 2: w = 2/3, logit ln 2.
    base = _blocks([1.0, 2.0, 4.0, -7.0])
    found = overfitting(np.column_stack([base + 0.001, base]))
    assert found.blocks == 4 and found.splits == 6
    assert found.probability == 0.0
    assert found.logits == pytest.approx([math.log(2)] * 6)


def test_a_best_in_sample_that_is_worst_out_of_sample_always_overfits():
    # A's block means 1, 2, 4 and -7 sum to zero and B is minus A, so in every split the
    # one ahead in sample is behind out of sample (no pair of blocks sums to zero): rank
    # 1 of 2, w = 1/3, logit ln(1/2).
    a = _blocks([1.0, 2.0, 4.0, -7.0])
    found = overfitting(np.column_stack([a, -a]))
    assert found.probability == 1.0
    assert found.logits == pytest.approx([math.log(0.5)] * 6)


def test_too_few_sessions_or_one_combination_leave_the_probability_undefined():
    assert math.isnan(overfitting(np.ones((7, 3))).probability)
    assert math.isnan(overfitting(np.ones((40, 1))).probability)
