"""The equivalence of 14-strategy-api 4.1, as a property: for any condition arrays and
any bars, a condition strategy (run by the Rust signal driver) and the literal `on_bar`
of section 4.1 (run by the Python event driver) give identical fills, orders, trades,
positions, equity, events and costs.

Bars are two NSE sessions of 5-minute bars that walk by whole ticks, with some bars
missing (SPEC-07); the product is MIS, so the entry window and the square-off apply in
both forms.
"""

import numpy as np
from condition_runs import BARS_PER_SESSION, SESSIONS, assert_same_result, run, walk_bars
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import openalgo.backtest as bt

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


def literal_on_bar(arrays: dict, pyramiding: int, order_args: dict) -> type:
    """The `on_bar` of 14-strategy-api 4.1 written out. A strategy bar with no base bar
    is not delivered, so the bar's position in the arrays is the number of bars seen."""
    entry, exit_, short, cover = (arrays[name] for name in ("entry", "exit", "short", "cover"))

    class Literal(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def on_bar(self, bar):
            i = len(self.close) - 1
            position = self.position
            flat = position.is_flat and not self._ctx.has_pending_entry(self.symbol)
            if flat:
                if entry[i]:
                    self.buy(**order_args)
                elif short[i]:
                    self.sell(**order_args)
            elif position.is_long:
                if exit_[i]:
                    self.exit()
                elif entry[i] and position.adds < pyramiding:
                    self.buy(**order_args)
            elif position.is_short:
                if cover[i]:
                    self.exit()
                elif short[i] and position.adds < pyramiding:
                    self.sell(**order_args)

    Literal.pyramiding = pyramiding
    return Literal


def conditions(arrays: dict, pyramiding: int, order_args: dict) -> type:
    """The same strategy declared with entry_when, exit_when, short_when, cover_when."""

    class Conditions(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(arrays["entry"], **order_args)
            self.exit_when(arrays["exit"])
            self.short_when(arrays["short"], **order_args)
            self.cover_when(arrays["cover"])

    Conditions.pyramiding = pyramiding
    return Conditions


@st.composite
def runs(draw):
    moves = draw(st.lists(st.integers(-30, 30), min_size=5, max_size=40))
    missing = draw(st.sets(st.integers(0, BARS - 1), max_size=12))
    count = BARS - len(missing)
    arrays = {
        name: np.array(draw(st.lists(st.booleans(), min_size=count, max_size=count)))
        for name in ("entry", "exit", "short", "cover")
    }
    pyramiding = draw(st.integers(0, 2))
    order_args = draw(st.sampled_from(ORDER_ARGUMENTS))
    return walk_bars(moves, missing), arrays, pyramiding, order_args


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=runs())
def test_conditions_and_the_literal_on_bar_give_identical_results(case):
    data, arrays, pyramiding, order_args = case
    by_conditions = run(conditions(arrays, pyramiding, order_args), data)
    by_on_bar = run(literal_on_bar(arrays, pyramiding, order_args), data)
    assert_same_result(by_conditions, by_on_bar)
