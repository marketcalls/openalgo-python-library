"""Monte Carlo by bootstrap of session returns and of trades: deterministic for a seed,
and its percentiles on cases whose distribution is known exactly.

A run of two sessions at +10 and -10 percent draws two sessions per path, each +10 or
-10 percent with probability 1/2: final returns 1.21 - 1 (1/4), 0.99 - 1 (1/2) and
0.81 - 1 (1/4), and maximum drawdowns 0 (1/4), -10 percent (1/2) and -19 percent (1/4).
"""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def closes(*values: float) -> pd.Series:
    days = pd.bdate_range("2025-06-02", periods=len(values), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


@pytest.fixture(scope="module")
def coin() -> metrics.Performance:
    """Sessions of +10 and -10 percent from 100."""
    return metrics.analyse(closes(110.0, 99.0), exchanges="NSE", opening_wealth=100.0)


def share(values: pd.Series, target: float) -> float:
    return float(((values - target).abs() < 1e-12).mean())


def test_the_same_seed_gives_the_same_simulations(coin):
    first = metrics.monte_carlo(coin, simulations=300, seed=11)
    again = metrics.monte_carlo(coin, simulations=300, seed=11)
    other = metrics.monte_carlo(coin, simulations=300, seed=12)
    pd.testing.assert_frame_equal(first.runs, again.runs)
    pd.testing.assert_frame_equal(first.band, again.band)
    assert not first.runs["final_return"].equals(other.runs["final_return"])


def test_final_returns_follow_the_binomial_distribution(coin):
    study = metrics.monte_carlo(coin, simulations=4_000, seed=3)
    finals = study.runs["final_return"]
    for outcome, probability in ((0.21, 0.25), (-0.01, 0.5), (-0.19, 0.25)):
        assert share(finals, outcome) == pytest.approx(probability, abs=0.03)
    table = study.percentiles
    assert table.loc[0.05, "final_return"] == pytest.approx(-0.19)
    assert table.loc[0.50, "final_return"] == pytest.approx(-0.01)
    assert table.loc[0.95, "final_return"] == pytest.approx(0.21)


def test_maximum_drawdowns_follow_the_same_draws(coin):
    table = metrics.monte_carlo(coin, simulations=4_000, seed=3).percentiles
    assert table.loc[0.05, "max_drawdown"] == pytest.approx(-0.19)
    assert table.loc[0.50, "max_drawdown"] == pytest.approx(-0.10)
    assert table.loc[0.95, "max_drawdown"] == pytest.approx(0.0)


def test_bust_and_goal_are_the_shares_of_paths_that_reach_them(coin):
    study = metrics.monte_carlo(coin, simulations=4_000, seed=3, bust=-0.15, goal=0.2)
    deepest, finals = study.runs["max_drawdown"], study.runs["final_return"]
    assert study.bust_probability == pytest.approx(float((deepest <= -0.15).mean()))
    assert study.goal_probability == pytest.approx(float((finals >= 0.2).mean()))
    assert study.bust_probability == pytest.approx(0.25, abs=0.03)
    unasked = metrics.monte_carlo(coin, simulations=10, seed=3)
    assert math.isnan(unasked.bust_probability) and math.isnan(unasked.goal_probability)


def test_the_band_runs_from_zero_to_the_final_percentiles(coin):
    study = metrics.monte_carlo(coin, simulations=4_000, seed=3)
    assert list(study.band.index) == [0, 1, 2]
    assert (study.band.loc[0] == 0.0).all()
    assert list(study.band.columns) == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert study.band.loc[2, 0.5] == pytest.approx(study.percentiles.loc[0.5, "final_return"])


def test_the_runs_own_path_and_figures_are_reported(coin):
    study = metrics.monte_carlo(coin, simulations=10, seed=3)
    assert study.actual_path.to_numpy() == pytest.approx([0.0, 0.1, -0.01])
    assert study.actual["final_return"] == pytest.approx(-0.01)
    assert study.actual["max_drawdown"] == pytest.approx(-0.1)


def trading_run() -> metrics.Performance:
    """Two closed trades of +100 and -100 rupees on 1,000."""
    equity = closes(1_100.0, 1_000.0)
    trades = pd.DataFrame(
        {
            "entry_time": equity.index - pd.Timedelta(hours=5),
            "exit_time": equity.index - pd.Timedelta(hours=1),
            "entry_price": 100.0,
            "exit_price": [200.0, 0.0],
            "quantity": 1.0,
            "side": "BUY",
            "gross_pnl": [100.0, -100.0],
            "net_pnl": [100.0, -100.0],
            "costs": 0.0,
        }
    )
    return metrics.analyse(equity, exchanges="NSE", trades=trades, opening_wealth=1_000.0)


def test_trades_are_booked_on_the_capital_at_fixed_size():
    study = metrics.monte_carlo(trading_run(), simulations=4_000, seed=5, resample="trades")
    finals = study.runs["final_return"]
    # 1,200, 1,000 or 800 rupees after two trades.
    for outcome, probability in ((0.2, 0.25), (0.0, 0.5), (-0.2, 0.25)):
        assert share(finals, outcome) == pytest.approx(probability, abs=0.03)
    assert study.actual_path.to_numpy() == pytest.approx([0.0, 0.1, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"simulations": 0},
        {"simulations": 2.5},
        {"seed": -1},
        {"resample": "bars"},
        {"bust": 0.2},
        {"goal": -2.0},
    ],
)
def test_unusable_settings_are_refused(coin, kwargs):
    with pytest.raises(ConfigError):
        metrics.monte_carlo(coin, **kwargs)


def test_a_run_without_closed_trades_has_nothing_to_resample(coin):
    with pytest.raises(ConfigError, match="nothing to resample"):
        metrics.monte_carlo(coin, resample="trades")
