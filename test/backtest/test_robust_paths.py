"""Equity paths of booked trades: wealth, drawdown and ruin worked by hand; the fan's
points and bands; and the resampled paths drawn the way the kernel's Monte Carlo draws
them.

Hand-worked paths on a capital of 1,000:

- +100, -300, +50: equity 1,100, 800, 850; wealth 1.10, 0.80, 0.85; the deepest fall is
  from 1.10 to 0.80, 0.80 / 1.10 - 1 = -27.27%; final return -15%.
- -600, -500, +2,000: equity 400, then -100, ruined at the second trade; wealth 0.40,
  then 0 to the end; final return -100%, drawdown -100%.
- +50, +50: never below a peak; drawdown 0.
"""

import numpy as np
import pytest
from robust_runs import trade_performance

from openalgo.backtest import metrics
from openalgo.backtest._robust import _paths
from openalgo.backtest._robust._trades import trade_run


def test_a_path_books_trades_on_the_capital_and_falls_from_its_peak():
    path = _paths.walk(np.array([[100.0, -300.0, 50.0]]), 1_000.0)
    assert path.wealth[0] == pytest.approx([1.0, 1.1, 0.8, 0.85])
    assert path.final_return[0] == pytest.approx(-0.15)
    assert path.max_drawdown[0] == pytest.approx(0.8 / 1.1 - 1)
    assert not path.ruined[0]


def test_a_path_whose_equity_reaches_zero_is_ruined_there_and_stays_at_zero():
    path = _paths.walk(np.array([[-600.0, -500.0, 2_000.0]]), 1_000.0)
    assert path.wealth[0] == pytest.approx([1.0, 0.4, 0.0, 0.0])
    assert path.final_return[0] == -1.0
    assert path.max_drawdown[0] == -1.0
    assert path.ruined[0]


def test_a_path_that_never_falls_has_no_drawdown():
    assert _paths.walk(np.array([[50.0, 50.0]]), 1_000.0).max_drawdown[0] == 0.0


def test_a_fan_of_few_trades_has_a_point_at_every_trade():
    assert list(_paths.fan_steps(5)) == [0, 1, 2, 3, 4, 5]


def test_a_fan_of_many_trades_has_201_points_evenly_spaced():
    steps = _paths.fan_steps(1_000)
    assert len(steps) == 201 and steps[0] == 0 and steps[1] == 5 and steps[-1] == 1_000


def test_the_bands_are_percentiles_interpolated_between_order_statistics():
    # 21 runs worth 0 to 20: the p-quantile sits at position 20 p.
    cumulative = np.arange(21, dtype=float)[:, None]
    assert _paths.bands(cumulative)[0] == pytest.approx([1.0, 5.0, 10.0, 15.0, 19.0])


@pytest.mark.parametrize(("capital", "some_ruined"), [(100_000.0, False), (2_000.0, True)])
def test_resampled_paths_are_the_kernel_monte_carlo_runs(capital, some_ruined):
    # On 2,000 rupees many paths are ruined, and a ruined run takes fewer draws.
    pnl = [300.0, -450.0, 120.0, -80.0, 500.0, -900.0] * 5
    performance = trade_performance(pnl, capital=capital)
    run = trade_run(performance)
    study = metrics.monte_carlo(performance, simulations=150, seed=7, resample="trades")
    paths = _paths.walk(_paths.resampled_trades(run.pnl, run.capital, 7, 150), run.capital)
    assert paths.ruined.any() == some_ruined
    assert paths.final_return == pytest.approx(study.runs["final_return"].to_numpy(), abs=1e-12)
    assert paths.max_drawdown == pytest.approx(study.runs["max_drawdown"].to_numpy(), abs=1e-12)
