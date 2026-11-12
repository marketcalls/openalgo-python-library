"""A strategy's own sizing function, `size=` a callable `size(ctx, order) -> units`
(12-position-sizing 5, FR-SIZE-06).

Bars are the rising 1-minute bars of `strategy_samples` on 2025-07-01: the close of bar 0
is 500.05, so a decision there is sized at 500.05. No charges, no slippage, capital
10,00,000, whole shares unless a test says otherwise.
"""

import math

import numpy as np
import pytest
from strategy_samples import run

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError


def buying(size, **order) -> type:
    """A strategy that buys with `size=` and `order` after its first bar."""

    class Buys(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(size=size, **order)

    return Buys


def test_a_sizing_function_sees_the_account_the_price_the_stop_and_the_order():
    seen = []

    def size(ctx, order):
        seen.append((ctx, order))
        return 10

    run(buying(size, stop=bt.points(2)), quantity_mode="venue")
    ctx, order = seen[0]
    assert (ctx.equity, ctx.capital, ctx.cash) == (1_000_000.0, 1_000_000.0, 1_000_000.0)
    assert ctx.price == 500.05
    assert ctx.stop_distance == 2.0
    assert (ctx.lot_size, ctx.multiplier, ctx.fractional) == (1, 1.0, False)
    assert ctx.position == 0.0
    assert (order.symbol, order.side, order.product) == ("SBIN", "BUY", "CNC")


def test_the_quantity_a_sizing_function_returns_rounds_down_to_whole_shares():
    result = run(buying(lambda ctx, order: 10.7), quantity_mode="venue")
    assert result.fills.loc[0, "quantity_units"] == 10.0


def test_a_sizing_function_may_return_a_numpy_number():
    result = run(buying(lambda ctx, order: np.float64(12.0)), quantity_mode="venue")
    assert result.fills.loc[0, "quantity_units"] == 12.0


def test_a_sizing_function_that_returns_zero_skips_the_order():
    result = run(buying(lambda ctx, order: 0))
    assert result.fills.empty
    assert list(result.skipped["reason"]) == ["rounded_to_zero"]


@pytest.mark.parametrize("returned", [-1, math.nan, "5", True, None])
def test_a_sizing_function_must_return_a_quantity_of_zero_or_more(returned):
    with pytest.raises(ConfigError, match="sizing function"):
        run(buying(lambda ctx, order: returned))


def test_a_sizing_function_sees_the_position_it_adds_to():
    seen = []

    def size(ctx, order):
        seen.append(ctx.position)
        return 5

    class Adds(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

        def on_bar(self, bar):
            if len(self.close) in (1, 3):
                self.buy(size=size)

    run(Adds)
    assert seen == [0.0, 5.0]


def fifteen_shares(ctx, order):
    """A sizing function used as a strategy's default size."""
    return 15


def test_a_strategy_default_size_can_be_a_sizing_function():
    class Default(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"
        size = fifteen_shares

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy()

    result = run(Default)
    assert result.fills.loc[0, "quantity_units"] == 15.0


def test_a_condition_strategy_refuses_a_sizing_function():
    class Conditions(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.entry_when(self.close > 0, size=fifteen_shares)

    with pytest.raises(ConfigError, match="on_bar"):
        run(Conditions)
