"""Variance (design 17, section 2.13): the trades resampled at the run's win rate and at
a win rate some points lower, from the same draws, with odds worked by hand.

24 trades of +100 and 16 of -100: a win rate of 60%. A path of 40 draws with W winners
ends at 100 x (2W - 40), profitable when W >= 21, W binomial with n = 40 and p the rate:

    at the run's 60%          P(W >= 21) = 87.02%  (above the 80% the rule asks)
    5 points lower, at 55%    P(W >= 21) = 68.44%  (below it: the check fails)

computed below from the binomial terms C(40, k) p^k (1 - p)^(40 - k).
"""

import math

import numpy as np
import pytest
from robust_runs import check_of, trade_performance

from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED

TRADES = [100.0] * 24 + [-100.0] * 16
TOLERANCE = 0.03


def profitable_odds(rate: float) -> float:
    return sum(math.comb(40, k) * rate**k * (1 - rate) ** (40 - k) for k in range(21, 41))


@pytest.fixture(scope="module")
def result():
    return check_of(trade_performance(TRADES), "variance", runs=4_000)


def test_the_run_win_rate_and_the_lower_one_are_stated(result):
    assert result.summary["win_rate"] == pytest.approx(0.60)
    assert result.summary["lower_win_rate"] == pytest.approx(0.55)


def test_profitable_paths_at_each_rate_match_the_binomial_odds(result):
    assert result.summary["own_profitable_share"] == pytest.approx(
        profitable_odds(0.60), abs=TOLERANCE
    )
    assert result.summary["lower_profitable_share"] == pytest.approx(
        profitable_odds(0.55), abs=TOLERANCE
    )


def test_a_win_rate_that_survives_only_at_its_own_level_fails(result):
    assert result.verdict == FAILED


def test_the_lower_rate_only_turns_winners_into_others_on_the_same_draws(result):
    # Same draws: a path never ends higher at the lower rate than at the run's.
    table = result.table
    own = table[table["win_rate"] == result.summary["win_rate"]]["final_return"].to_numpy()
    lower = table[table["win_rate"] == result.summary["lower_win_rate"]]["final_return"]
    assert (lower.to_numpy() <= own + 1e-12).all()
    assert (lower.to_numpy() < own).any()


def test_a_planted_edge_passes_at_the_lower_rate():
    edge = [100.0] * 36 + [-100.0] * 4
    assert check_of(trade_performance(edge), "variance").verdict == PASSED


def test_more_points_set_in_the_checks_dict_lower_the_rate_further():
    result = check_of(trade_performance(TRADES), "variance", runs=200, points=20)
    assert result.summary["lower_win_rate"] == pytest.approx(0.40)
    assert "20 points lower" in result.rule


def test_the_lower_rate_never_goes_below_zero():
    # One winner in 30: a win rate of 3.3%, five points lower is 0.
    trades = [400.0] + [-10.0] * 29
    result = check_of(trade_performance(trades), "variance", runs=100)
    assert result.summary["lower_win_rate"] == 0.0
    assert result.summary["lower_profitable_share"] == 0.0


def test_a_run_that_did_not_make_money_is_not_judged():
    losing = [100.0] * 16 + [-100.0] * 24
    result = check_of(trade_performance(losing), "variance")
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("No edge to judge")


def test_a_run_whose_every_trade_won_is_not_judged():
    result = check_of(trade_performance([50.0] * 40), "variance")
    assert result.verdict == NOT_JUDGED
    assert "Every closed trade won" in result.note


def test_both_rates_have_a_row_per_path():
    result = check_of(trade_performance(TRADES), "variance", runs=50)
    assert len(result.table) == 100
    assert np.unique(result.table["win_rate"]) == pytest.approx([0.55, 0.60])
