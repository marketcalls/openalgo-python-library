"""Trade resample (design 17, section 2.11): the closed trades drawn with replacement, the
Monte Carlo of `metrics.monte_carlo(resample="trades")`, with odds worked by hand.

- A planted edge: 40 trades, 36 winning +100 and 4 losing -50. A path of 40 draws with L
  losers ends at 100 x (40 - L) - 50 x L, a loss only for L >= 27, whose odds at one loser
  in ten are below 1e-15: every path is profitable, and none comes near a 50% drawdown on
  1,00,000.
- A small edge: 20 trades of +100 and 20 of -90, making 200. A path's P&L is
  100 W - 90 (40 - W) = 190 W - 3,600 with W the winners drawn, binomial with n = 40 and
  p = 1/2, so it is profitable when W >= 19: 68.2% of paths, computed below from the
  binomial terms C(40, k) / 2^40.
- No edge: 20 trades of +100 and 20 of -100 make nothing, so there is no edge to judge.
- Ruin: 29 trades of +40 and one of -1,100 on 1,000, making 60. A path rises only until
  its first loss, so its peak is at most 1,000 + 29 x 40 = 2,160, and that loss is a fall
  of at least 1,100 / 2,160 = 50.9% from it: a path reaches a 50% drawdown exactly when it
  draws the loss at least once, 1 - (29/30)^30 = 63.84% of paths.
"""

import math

import numpy as np
import pytest
from robust_runs import blocks_of, check_of, trade_performance

from openalgo.backtest import metrics
from openalgo.backtest._robust._paths import SAMPLED_PATHS, walk
from openalgo.backtest._robust._trade_common import path_figures
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED

EDGE = [100.0] * 36 + [-50.0] * 4
SMALL_EDGE = [100.0, -90.0] * 20
SMALL_EDGE_PROFITABLE = sum(math.comb(40, k) for k in range(19, 41)) / 2**40
NO_EDGE = [100.0, -100.0] * 20
RUIN = [40.0] * 29 + [-1_100.0]
RUIN_ODDS = 1 - (29 / 30) ** 30
# Many runs keep the drawn shares within this of the worked odds (about 4 standard
# errors at 4,000 runs).
TOLERANCE = 0.03


def test_the_paths_are_the_monte_carlo_of_the_trades_with_the_checks_seed():
    performance = trade_performance(SMALL_EDGE)
    result = check_of(performance, "resample", runs=300, seed=5)
    # The resample check is number 11, so it draws from the stream of seed 5 + 11.
    study = metrics.monte_carlo(performance, simulations=300, seed=16, resample="trades")
    assert result.seed == 16
    assert result.table["final_return"].equals(study.runs["final_return"].rename_axis("run"))


def test_the_ruin_odds_are_the_monte_carlo_bust_odds_at_the_ruin_drawdown():
    performance = trade_performance(RUIN, capital=1_000.0)
    result = check_of(performance, "resample", runs=500, seed=1)
    study = metrics.monte_carlo(performance, simulations=500, seed=12, resample="trades", bust=-0.5)
    assert result.summary["ruin_odds"] == study.bust_probability


def test_a_planted_edge_passes():
    result = check_of(trade_performance(EDGE), "resample")
    assert result.verdict == PASSED
    assert result.summary["profitable_share"] == 1.0
    assert result.summary["ruin_odds"] == 0.0


def test_a_small_edge_fails_with_the_share_of_profitable_paths_worked_by_hand():
    result = check_of(trade_performance(SMALL_EDGE), "resample", runs=4_000)
    assert result.verdict == FAILED
    assert result.summary["profitable_share"] == pytest.approx(SMALL_EDGE_PROFITABLE, abs=TOLERANCE)


def test_a_run_with_no_edge_is_not_judged():
    result = check_of(trade_performance(NO_EDGE), "resample")
    assert result.verdict == NOT_JUDGED
    assert result.note.startswith("No edge to judge")


def test_a_path_that_breaks_even_is_not_profitable_whatever_the_rounding():
    # 0.1 rupees has no exact binary fraction, so a path of 20 wins and 20 losses ends a
    # rounding residue away from its capital; it made nothing and is not profitable. With
    # 21 wins of 0.1 in 40 trades a path is profitable when it draws 21 wins or more.
    small = [0.1] * 21 + [-0.1] * 19
    rate = 21 / 40
    odds = sum(math.comb(40, k) * rate**k * (1 - rate) ** (40 - k) for k in range(21, 41))
    result = check_of(trade_performance(small), "resample", runs=4_000)
    assert result.summary["profitable_share"] == pytest.approx(odds, abs=TOLERANCE)


def test_the_ruin_odds_are_the_share_of_paths_reaching_the_ruin_drawdown():
    result = check_of(trade_performance(RUIN, capital=1_000.0), "resample", runs=4_000)
    assert result.summary["ruin_odds"] == pytest.approx(RUIN_ODDS, abs=TOLERANCE)
    assert result.verdict == FAILED


def test_the_average_profit_is_the_mean_final_return_on_the_capital():
    result = check_of(trade_performance(SMALL_EDGE), "resample", runs=500)
    table = result.table
    assert result.summary["average_profit_rupees"] == pytest.approx(table["profit_rupees"].mean())
    assert table["profit_rupees"].to_numpy() == pytest.approx(
        table["final_return"].to_numpy() * 100_000
    )


def test_the_95_percent_interval_holds_95_percent_of_the_paths():
    result = check_of(trade_performance(SMALL_EDGE), "resample", runs=4_000)
    summary, table = result.summary, result.table
    inside = table["profit_rupees"].between(
        summary["profit_low_rupees"], summary["profit_high_rupees"]
    )
    assert inside.mean() == pytest.approx(0.95, abs=0.02)


def test_a_looser_rule_set_in_the_checks_dict_judges_the_same_paths():
    result = check_of(trade_performance(SMALL_EDGE), "resample", runs=4_000, pass_share=0.4)
    assert result.verdict == PASSED
    assert result.rule.startswith("at least 40% of paths profitable")


def test_the_fan_draws_the_first_runs_and_they_end_where_the_table_says():
    performance = trade_performance(SMALL_EDGE)
    result = check_of(performance, "resample", runs=300)
    fan = blocks_of(performance, "resample", runs=300)[0][0].fan
    assert len(fan.samples) == SAMPLED_PATHS and fan.runs == 300
    ends = result.table["final_return"].to_numpy()[:SAMPLED_PATHS]
    assert fan.samples[:, -1] == pytest.approx(ends, abs=1e-12)


def test_too_few_trades_are_not_judged():
    result = check_of(trade_performance(EDGE[:20]), "resample")
    assert result.verdict == NOT_JUDGED
    assert result.note == "Too few trades to judge: 20 closed trades, 30 needed."


def test_the_rule_says_ruin_is_a_fall_from_a_peak():
    result = check_of(trade_performance(EDGE), "resample", runs=50)
    assert result.rule.endswith("at most 5% of paths falling 50% from a peak")


def test_ruin_is_a_fall_from_a_peak_not_a_share_of_the_capital():
    # On 1,00,000 a path that rises to 3,00,000 and falls 1,00,000 loses half the
    # capital but a third of its peak: it is not ruined at a 50% fall, as the
    # Monte Carlo's bust odds count it.
    path = walk(np.array([[200_000.0, -100_000.0]]), 100_000.0)
    assert path_figures(path.final_return, path.max_drawdown, 100_000.0, 0.5)["ruin_odds"] == 0.0
