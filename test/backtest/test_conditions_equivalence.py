"""The equivalence of 14-strategy-api 4.1: for any condition arrays and any bars, a
condition strategy (run by the Rust signal driver) and the literal `on_bar` of section 4.1
(run by the Python event driver) give identical fills, orders, trades, positions, equity,
events and costs, bit for bit, with the default broker's charges and the default
slippage on (the Phase 1 exit criterion: the fast path is bit-identical to the general
path).

Bars are two NSE sessions of 5-minute bars that walk by whole ticks, with some bars
missing (SPEC-07); the product is MIS, so the entry window and the square-off apply in
both forms. A property test draws the cases, and a fixed seeded set repeats the same
cases on every run.
"""

import numpy as np
import pytest
from condition_runs import (
    BARS_PER_SESSION,
    CONDITION_NAMES,
    SESSIONS,
    assert_same_result,
    conditions,
    literal_on_bar,
    run,
    walk_bars,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import openalgo.backtest as bt

pytestmark = pytest.mark.filterwarnings("ignore::openalgo.backtest.costs.UnverifiedScheduleWarning")

BARS = len(SESSIONS) * BARS_PER_SESSION
ORDER_ARGUMENTS = (
    {"qty": 10},
    {"lots": 2},
    {"size": bt.size.qty(7)},
    {"qty": 10, "stop": bt.points(1)},
    {"qty": 10, "stop": bt.points(1), "target": bt.points(2)},
    {"qty": 10, "targets": [(bt.points(1), 0.5), (bt.points(2), 0.5)]},
    {"qty": 10, "trail": bt.trail_points(1)},
    {"qty": 10, "stop": bt.pct(0.3), "breakeven": bt.breakeven(0.5)},
)
# The run settings of `bt.backtest` when a run names no charges and no slippage.
CHARGES_AND_SLIPPAGE = {"costs": "zerodha", "slippage": "default"}
SEEDED_CASES = 30
# Tick moves of a bar, at most this many either way; bars missing, at most this many.
MAX_MOVE_TICKS = 30
MAX_MISSING_BARS = 12


def assert_both_forms_agree(data, arrays: dict, pyramiding: int, order_args: dict) -> None:
    by_conditions = run(conditions(arrays, pyramiding, order_args), data, **CHARGES_AND_SLIPPAGE)
    by_on_bar = run(literal_on_bar(arrays, pyramiding, order_args), data, **CHARGES_AND_SLIPPAGE)
    assert_same_result(by_conditions, by_on_bar)


@st.composite
def runs(draw):
    moves = draw(st.lists(st.integers(-MAX_MOVE_TICKS, MAX_MOVE_TICKS), min_size=5, max_size=40))
    missing = draw(st.sets(st.integers(0, BARS - 1), max_size=MAX_MISSING_BARS))
    count = BARS - len(missing)
    arrays = {
        name: np.array(draw(st.lists(st.booleans(), min_size=count, max_size=count)))
        for name in CONDITION_NAMES
    }
    pyramiding = draw(st.integers(0, 2))
    order_args = draw(st.sampled_from(ORDER_ARGUMENTS))
    return walk_bars(moves, missing), arrays, pyramiding, order_args


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=runs())
def test_conditions_and_the_literal_on_bar_give_identical_results(case):
    assert_both_forms_agree(*case)


def seeded_case(seed: int) -> tuple:
    """Bars, arrays, pyramiding and order arguments drawn from a fixed seed."""
    rng = np.random.default_rng(seed)
    moves = rng.integers(-MAX_MOVE_TICKS, MAX_MOVE_TICKS + 1, size=int(rng.integers(5, 41)))
    missing = set(rng.choice(BARS, size=int(rng.integers(0, MAX_MISSING_BARS + 1)), replace=False))
    count = BARS - len(missing)
    arrays = {name: rng.random(count) < rng.uniform(0.1, 0.5) for name in CONDITION_NAMES}
    pyramiding = int(rng.integers(0, 3))
    order_args = ORDER_ARGUMENTS[seed % len(ORDER_ARGUMENTS)]
    return walk_bars(moves.tolist(), missing), arrays, pyramiding, order_args


@pytest.mark.parametrize("seed", range(SEEDED_CASES))
def test_conditions_and_the_literal_on_bar_agree_on_a_fixed_seeded_set(seed):
    assert_both_forms_agree(*seeded_case(seed))
