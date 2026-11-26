"""`bt.live(positional_stops=...)` (`_live/_plan_args.py`, `_live/plan.py`; design
20-positional-live section 6, decisions P2, P5 and 11.4; the lead's ruling N5b-2): one value
for every strategy or a dict by strategy class name, checked by the function `bt.backtest`
checks it with, so both refuse the same values with the same words, and resolved into each
strategy's settings as a backtest of the day resolves it.

`specify` runs on a clock at 10:15 IST of Monday 2026-11-16 and contacts nothing.
"""

from __future__ import annotations

import pytest
from plan_samples import EmaTrend, backtest_arguments, ist, specify

from openalgo.backtest._settings import resolve
from openalgo.backtest.errors import ConfigError


class CncTrend(EmaTrend):
    """EmaTrend in CNC: its position is held overnight."""

    product = "CNC"


class CncOther(EmaTrend):
    """A second CNC strategy, on another symbol."""

    symbol = "INFY"
    product = "CNC"


def stops_of(spec) -> dict[str, str]:
    return {each.name: each.plan.risk["positional_stops"] for each in spec.strategies}


def test_the_default_holds_every_positional_stop_as_a_gtt():
    assert stops_of(specify(CncTrend)) == {"CncTrend": "gtt"}


def test_one_value_is_every_strategys():
    spec = specify([CncTrend, CncOther], positional_stops="rearm_daily", capital=500_000)

    assert stops_of(spec) == {"CncTrend": "rearm_daily", "CncOther": "rearm_daily"}


def test_a_dict_gives_each_strategy_its_own():
    choice = {"CncTrend": "rearm_daily", "CncOther": "gtt"}

    spec = specify([CncTrend, CncOther], positional_stops=choice, capital=500_000)

    assert stops_of(spec) == choice


def test_a_value_bt_backtest_refuses_is_refused_with_the_same_words():
    # bt.backtest resolves its settings with `_settings.resolve`, on its own arguments.
    arguments = backtest_arguments(CncTrend, None, ist(10, 15), positional_stops="overnight")
    with pytest.raises(ConfigError) as backtest_refusal:
        resolve(CncTrend, arguments)
    with pytest.raises(ConfigError) as live_refusal:
        specify(CncTrend, positional_stops="overnight")

    assert str(live_refusal.value) == str(backtest_refusal.value)
    assert str(live_refusal.value) == (
        "positional_stops takes one of ('gtt', 'rearm_daily'), got 'overnight'"
    )


def test_a_dict_naming_a_strategy_the_session_does_not_run_is_refused():
    with pytest.raises(ConfigError, match=r"positional_stops names \['Other'\], which this"):
        specify(CncTrend, positional_stops={"Other": "rearm_daily"})


def test_a_dict_that_leaves_a_strategy_out_is_refused():
    with pytest.raises(ConfigError, match="positional_stops as a dict must name every strategy"):
        specify([CncTrend, CncOther], positional_stops={"CncTrend": "gtt"}, capital=500_000)


def test_the_choice_is_in_each_strategys_assumptions_as_in_a_backtest():
    spec = specify(CncTrend, positional_stops="rearm_daily")

    (each,) = spec.strategies
    assert each.plan.risk["positional_stops"] == "rearm_daily"
    assert each.arguments["positional_stops"] == "rearm_daily"
