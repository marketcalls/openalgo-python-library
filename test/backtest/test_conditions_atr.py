"""ATR distances and the ATR-risk sizer in condition strategies (14-strategy-api 4.1,
SPEC-25): `bt.atr`, `bt.trail_atr` and `bt.size.atr_risk` in `entry_when` and
`short_when` take the `openalgo.ta` ATR of the deciding bar, as an `on_bar` order does,
so the two forms give identical results.

Bars are two NSE sessions of 5-minute bars that walk by whole ticks (`condition_runs`),
some missing; the product is MIS; no charges, no slippage.
"""

import numpy as np
import pytest
from condition_runs import BARS_PER_SESSION, SESSIONS, assert_same_result, run, walk_bars
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_conditions_equivalence import conditions, literal_on_bar

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError
from openalgo.indicators import causality

BARS = len(SESSIONS) * BARS_PER_SESSION
PERIOD = 5
ORDER_ARGUMENTS = (
    {"qty": 10, "stop": bt.atr(2, PERIOD)},
    {"qty": 10, "stop": bt.atr(1.5, PERIOD), "target": bt.atr(3, PERIOD)},
    {"qty": 10, "targets": [(bt.atr(1, PERIOD), 0.5), (bt.atr(2, 8), 0.5)]},
    {"qty": 10, "trail": bt.trail_atr(1.5, PERIOD)},
    {"qty": 10, "stop": bt.atr(1, PERIOD), "breakeven": bt.breakeven(bt.atr(1, 8))},
    {"size": bt.size.atr_risk(0.5, 2, period=PERIOD), "stop": bt.atr(2, PERIOD)},
)
# No entry before the slowest ATR above (period 8) has a value: an ATR distance without
# one stops the run in both forms (tested below).
WARM_UP = 8


@st.composite
def runs(draw):
    moves = draw(st.lists(st.integers(-30, 30), min_size=5, max_size=40))
    missing = draw(st.sets(st.integers(WARM_UP, BARS - 1), max_size=10))
    count = BARS - len(missing)
    arrays = {
        name: np.array(draw(st.lists(st.booleans(), min_size=count, max_size=count)))
        for name in ("entry", "exit", "short", "cover")
    }
    for name in ("entry", "short"):
        arrays[name][:WARM_UP] = False
    pyramiding = draw(st.integers(0, 2))
    order_args = draw(st.sampled_from(ORDER_ARGUMENTS))
    return walk_bars(moves, missing), arrays, pyramiding, order_args


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=runs())
def test_conditions_with_atr_rules_equal_the_literal_on_bar(case):
    data, arrays, pyramiding, order_args = case
    by_conditions = run(conditions(arrays, pyramiding, order_args), data)
    by_on_bar = run(literal_on_bar(arrays, pyramiding, order_args), data)
    assert_same_result(by_conditions, by_on_bar)


def entering_at(bar: int, order_args: dict) -> dict:
    arrays = {name: np.zeros(BARS, dtype=bool) for name in ("entry", "exit", "short", "cover")}
    arrays["entry"][bar] = True
    return arrays


def test_an_atr_entry_before_the_atr_has_a_value_stops_the_run():
    # The entry is decided at the close of the first bar; an ATR of 5 bars has no value.
    order_args = {"qty": 10, "stop": bt.atr(2, PERIOD)}
    strategy = conditions(entering_at(0, order_args), 0, order_args)
    with pytest.raises(ConfigError, match=r"bt\.atr\(period=5\) has no positive value"):
        run(strategy, walk_bars([3, -2, 5, -1]))


def test_an_atr_the_causality_registry_does_not_clear_is_refused(monkeypatch):
    monkeypatch.setattr(causality, "is_causal", lambda name, output=None: False)
    order_args = {"qty": 10, "stop": bt.atr(2, PERIOD)}
    strategy = conditions(entering_at(10, order_args), 0, order_args)
    with pytest.raises(ConfigError, match="looks ahead"):
        run(strategy, walk_bars([3, -2, 5, -1]))
