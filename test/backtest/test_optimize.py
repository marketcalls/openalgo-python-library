"""`bt.optimize` (14-strategy-api 5.1 and 5.2; 03-architecture 5).

The strategy is `strategy_samples.EmaCross` on `wave_bars`, a 3 by 3 grid declared with
`bt.param` (fast 3, 5, 8; slow 10, 15, 20). Worker processes use spawn, the only start
method Windows has, so these tests exercise the Windows path on every platform.
"""

import datetime as dt
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from strategy_samples import (
    CAPITAL,
    EmaCross,
    EmaCrossIntraday,
    final_equity,
    process_id,
    wave_bars,
)

import openalgo.backtest as bt
from openalgo.backtest import _causality_checks
from openalgo.backtest._optimize import LARGE_EVENT_GRID, _plan
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ConfigError

GRID = [(fast, slow) for fast in (3, 5, 8) for slow in (10, 15, 20)]
SETTINGS = {"capital": CAPITAL, "costs": None, "slippage": 0}


def sweep(**settings):
    return bt.optimize(EmaCross, data=wave_bars(), **{**SETTINGS, **settings})


def single(fast, slow):
    return bt.backtest(EmaCross, data=wave_bars(), params={"fast": fast, "slow": slow}, **SETTINGS)


@pytest.fixture(scope="module")
def pooled():
    lines = []
    result = sweep(workers=2, progress=lines.append)
    return result, lines


def test_a_3_by_3_grid_in_worker_processes_equals_the_nine_single_runs(pooled):
    table = pooled[0].table
    assert list(zip(table["fast"], table["slow"], strict=True)) == GRID
    for row, (fast, slow) in zip(table.itertuples(), GRID, strict=True):
        result = single(fast, slow)
        net_profit = result.equity["equity_rupees"].iloc[-1] - CAPITAL
        assert row.objective == net_profit
        assert row.net_profit == net_profit
        assert row.trades == len(result.trades)


def test_the_grid_has_trades_to_compare(pooled):
    assert pooled[0].table["trades"].sum() > 0


def test_the_table_does_not_depend_on_the_number_of_workers(pooled):
    pd.testing.assert_frame_equal(sweep(workers=1).table, pooled[0].table, check_exact=True)


def test_the_plan_is_reported_before_the_sweep_with_its_estimate(pooled):
    result, lines = pooled
    assert lines == [result.plan]
    assert result.plan.startswith(
        "bt.optimize: 9 combinations of EmaCross on the event path (on_bar runs in Python "
        "for every strategy bar), in 2 worker processes. The first took "
    )
    assert "the rest should take about" in result.plan
    assert result.path == "event"


def test_best_is_the_highest_objective_and_select_reproduces_it(pooled):
    result = pooled[0]
    table = result.table
    top = table.loc[table["objective"].idxmax()]
    assert result.best == {"fast": int(top["fast"]), "slow": int(top["slow"])}
    selected = result.select(**result.best)
    alone = single(**result.best)
    pd.testing.assert_frame_equal(selected.fills, alone.fills, check_exact=True)
    assert np.array_equal(selected.equity.to_numpy(), alone.equity.to_numpy())


def test_select_refuses_values_outside_the_grid(pooled):
    with pytest.raises(ConfigError, match="not a combination"):
        pooled[0].select(fast=4, slow=10)


def test_the_heatmap_is_the_objective_over_two_parameters(pooled):
    table = pooled[0].table
    heatmap = pooled[0].heatmap("fast", "slow")
    assert list(heatmap.columns) == [3, 5, 8]
    assert list(heatmap.index) == [10, 15, 20]
    for fast, slow in GRID:
        expected = table[(table["fast"] == fast) & (table["slow"] == slow)]["objective"].item()
        assert heatmap.loc[slow, fast] == expected


def test_maximize_false_picks_the_lowest_objective():
    result = sweep(workers=1, maximize=False)
    table = result.table
    low = table.loc[table["objective"].idxmin()]
    assert result.best == {"fast": int(low["fast"]), "slow": int(low["slow"])}


def test_a_custom_objective_runs_in_worker_processes():
    table = sweep(workers=2, objective=final_equity, fast=(3,), slow=(10, 20)).table
    for row in table.itertuples():
        result = single(int(row.fast), int(row.slow))
        assert row.objective == result.equity["equity_rupees"].iloc[-1]


def test_the_rest_of_the_grid_runs_in_spawned_worker_processes():
    table = sweep(workers=2, objective=process_id).table
    first, rest = table["objective"].iloc[0], set(table["objective"].iloc[1:])
    assert first == os.getpid()
    assert os.getpid() not in rest and 1 <= len(rest) <= 2


def test_ranges_in_the_call_replace_the_declared_ones():
    table = sweep(workers=1, fast=(3, 5)).table
    assert list(zip(table["fast"], table["slow"], strict=True)) == [
        (3, 10),
        (3, 15),
        (3, 20),
        (5, 10),
        (5, 15),
        (5, 20),
    ]


def test_params_fixes_a_parameter_out_of_the_grid():
    table = sweep(workers=1, params={"fast": 5}).table
    assert list(table.columns[:1]) == ["slow"]
    assert list(table["slow"]) == [10, 15, 20]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"speed": (1, 2)}, "neither run arguments of bt.backtest nor parameters"),
        ({"fast": (3,), "params": {"fast": 3}}, "both fixed by params= and swept"),
        ({"fast": ()}, "the range of fast is empty"),
        ({"objective": "luck"}, "objective takes a metric name"),
        ({"workers": 0}, "workers takes at least 1"),
    ],
)
def test_unusable_settings_are_refused(settings, message):
    with pytest.raises(ConfigError, match=message):
        sweep(**settings)


def test_a_strategy_without_ranges_has_nothing_to_sweep():
    class Fixed(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"
        fast = 3

    with pytest.raises(ConfigError, match="nothing to sweep"):
        bt.optimize(Fixed, data=wave_bars(), **SETTINGS)


@pytest.mark.skipif(
    importlib.util.find_spec("cloudpickle") is not None, reason="cloudpickle is installed"
)
def test_a_class_workers_cannot_import_is_refused_with_advice():
    class Local(EmaCross):
        pass

    with pytest.raises(ConfigError, match="install cloudpickle, or pass workers=1"):
        bt.optimize(Local, data=wave_bars(), workers=2, slow=(10,), **SETTINGS)


def test_a_class_workers_cannot_import_runs_with_one_worker():
    class Local(EmaCross):
        pass

    table = bt.optimize(Local, data=wave_bars(), workers=1, fast=(3, 5), slow=(10,), **SETTINGS)
    assert len(table.table) == 2


def test_a_run_with_its_own_calendar_sweeps_in_this_process():
    calendar = default_calendar().copy()
    result = sweep(workers=2, calendar=calendar, slow=(10,))
    assert "in this process (the run's own calendar or cost model cannot be sent" in result.plan


def test_a_sweep_with_the_default_charges_and_slippage_matches_the_single_runs():
    # Workers resolve the cost model and the slippage table again from costs="zerodha"
    # and slippage="default".
    data = wave_bars()
    swept = bt.optimize(EmaCrossIntraday, data=data, capital=CAPITAL, workers=2, slow=(10,))
    for row in swept.table.itertuples():
        params = {"fast": int(row.fast)}
        alone = bt.backtest(EmaCrossIntraday, data=data, capital=CAPITAL, params=params)
        assert row.objective == alone.metrics["net_profit"]
    assert swept.table["trades"].sum() > 0


def test_a_cost_model_that_cannot_be_sent_to_workers_sweeps_in_this_process():
    result = bt.optimize(
        EmaCrossIntraday,
        data=wave_bars(),
        capital=CAPITAL,
        costs=bt.costs.flat(20),
        workers=2,
        slow=(10,),
    )
    assert "own calendar or cost model cannot be sent" in result.plan


def test_the_causality_checks_run_once_for_the_sweep(monkeypatch):
    calls = []
    original = _causality_checks.IndicatorChecks._truncation

    def counted(self, *args):
        calls.append(args[0])
        return original(self, *args)

    monkeypatch.setattr(_causality_checks.IndicatorChecks, "_truncation", counted)
    sweep(workers=1)
    # Both lines are ta.ema over the same closes: one check for nine combinations.
    assert calls == ["ema(SBIN.close, 3)"]


def test_a_large_event_grid_suggests_the_condition_form():
    text = _plan(EmaCross, LARGE_EVENT_GRID + 1, 4, 0.5, "")
    assert "suits the condition form" in text
    assert "about 125.0 s" in text  # 1000 more combinations x 0.5 s / 4 workers


def test_optimize_is_reachable_from_the_package_without_a_start_date():
    result = bt.optimize(
        EmaCross,
        data=wave_bars(),
        end=dt.date(2025, 7, 1),
        workers=1,
        fast=(3,),
        slow=(10,),
        **SETTINGS,
    )
    assert isinstance(result, bt.OptimizeResult)
