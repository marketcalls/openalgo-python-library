"""Trade resample (design 17, section 2.11): the closed trades drawn with replacement, the
Monte Carlo of `metrics.monte_carlo(resample="trades")`, with odds worked by hand.

- A planted edge: 40 trades, 36 winning +100 and 4 losing -50. A path of 40 draws with L
  losers ends at 100 x (40 - L) - 50 x L, a loss only for L >= 27, whose odds at one loser
  in ten are below 1e-15: every path is profitable, and none comes near a 50% drawdown on
  1,00,000.
- Noise: 20 trades of +100 and 20 of -100. A path's P&L is 100 x (2W - 40) with W the
  winners drawn, binomial with n = 40 and p = 1/2, so it is profitable when W >= 21:
  (1 - C(40, 20) / 2^40) / 2 = 43.73% of paths.
- Ruin: 29 trades of +1 and one of -700 on 1,000. One -700 is a fall of at least
  700 / 1,029 = 68% from any peak, so a path reaches a 50% drawdown exactly when it draws
  the loss at least once: 1 - (29/30)^30 = 63.84% of paths.
"""

import math

import pytest
from robust_runs import check_of, trade_performance

from openalgo.backtest import metrics
from openalgo.backtest._robust._paths import SAMPLED_PATHS
from openalgo.backtest._robust._verdict import FAILED, NOT_JUDGED, PASSED

EDGE = [100.0] * 36 + [-50.0] * 4
NOISE = [100.0, -100.0] * 20
NOISE_PROFITABLE = (1 - math.comb(40, 20) / 2**40) / 2
RUIN = [1.0] * 29 + [-700.0]
RUIN_ODDS = 1 - (29 / 30) ** 30
# Many runs keep the drawn shares within this of the worked odds (about 4 standard
# errors at 4,000 runs).
TOLERANCE = 0.03


def test_the_paths_are_the_monte_carlo_of_the_trades_with_the_checks_seed():
    performance = trade_performance(NOISE)
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


def test_noise_fails_with_the_share_of_profitable_paths_worked_by_hand():
    result = check_of(trade_performance(NOISE), "resample", runs=4_000)
    assert result.verdict == FAILED
    assert result.summary["profitable_share"] == pytest.approx(NOISE_PROFITABLE, abs=TOLERANCE)


def test_a_path_that_breaks_even_is_not_profitable_whatever_the_rounding():
    # 0.1 rupees has no exact binary fraction, so a path of 20 wins and 20 losses ends a
    # rounding residue away from its capital; it made nothing and is not profitable.
    small = [0.1, -0.1] * 20
    result = check_of(trade_performance(small), "resample", runs=4_000)
    assert result.summary["profitable_share"] == pytest.approx(NOISE_PROFITABLE, abs=TOLERANCE)


def test_the_ruin_odds_are_the_share_of_paths_reaching_the_ruin_drawdown():
    result = check_of(trade_performance(RUIN, capital=1_000.0), "resample", runs=4_000)
    assert result.summary["ruin_odds"] == pytest.approx(RUIN_ODDS, abs=TOLERANCE)
    assert result.verdict == FAILED


def test_the_average_profit_is_the_mean_final_return_on_the_capital():
    result = check_of(trade_performance(NOISE), "resample", runs=500)
    table = result.table
    assert result.summary["average_profit_rupees"] == pytest.approx(table["profit_rupees"].mean())
    assert table["profit_rupees"].to_numpy() == pytest.approx(
        table["final_return"].to_numpy() * 100_000
    )


def test_the_95_percent_interval_holds_95_percent_of_the_paths():
    result = check_of(trade_performance(NOISE), "resample", runs=4_000)
    summary, table = result.summary, result.table
    inside = table["profit_rupees"].between(
        summary["profit_low_rupees"], summary["profit_high_rupees"]
    )
    assert inside.mean() == pytest.approx(0.95, abs=0.02)


def test_a_looser_rule_set_in_the_checks_dict_judges_the_same_paths():
    result = check_of(trade_performance(NOISE), "resample", runs=4_000, pass_share=0.4)
    assert result.verdict == PASSED
    assert result.rule.startswith("at least 40% of paths profitable")


def test_the_fan_draws_the_first_runs_and_they_end_where_the_table_says():
    result = check_of(trade_performance(NOISE), "resample", runs=300)
    fan = result._blocks[0][0].fan
    assert len(fan.samples) == SAMPLED_PATHS and fan.runs == 300
    ends = result.table["final_return"].to_numpy()[:SAMPLED_PATHS]
    assert fan.samples[:, -1] == pytest.approx(ends, abs=1e-12)


def test_too_few_trades_are_not_judged():
    result = check_of(trade_performance(EDGE[:20]), "resample")
    assert result.verdict == NOT_JUDGED
    assert result.note == "Too few trades to judge: 20 closed trades, 30 needed."
