"""Whole shares by default (decision D26): `quantity_mode` is "venue" unless a run asks
for "fractional", and every way of sizing an order rounds down to whole shares.

The runs are those of `accounting_runs` (capital 10,00,000; a decision at a close of
1,000.00 that fills at an open of 1,000.00), `strategy_samples` (a decision at 500.05)
and one session of daily bars at 1,200.00 for the rebalance, with no charges and no
slippage. Each expected quantity is worked out in its comment.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from accounting_runs import round_trip
from strategy_samples import run

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NO_COSTS = {"costs": None, "slippage": 0}


def entry_units(result: bt.Result) -> float:
    return result.fills.loc[0, "quantity_units"]


def test_the_default_quantity_mode_is_venue():
    assert round_trip(**NO_COSTS).assumptions["quantity_mode"] == "venue"


def test_a_sizer_gives_whole_shares_by_default():
    # 1,00,550 rupees at the decision price 1,000.00 is 100.55 shares: 100.
    result = round_trip({"size": bt.size.value(100_550)}, **NO_COSTS)
    assert entry_units(result) == 100.0


def test_part_shares_are_available_on_request():
    result = round_trip({"size": bt.size.value(100_550)}, quantity_mode="fractional", **NO_COSTS)
    assert entry_units(result) == 100.55


def test_a_percent_sizer_gives_whole_shares_by_default():
    # 33.333 percent of 10,00,000 is 3,33,330 at 1,000.00: 333.33 shares, so 333.
    result = round_trip({"size": bt.size.percent(33.333)}, **NO_COSTS)
    assert entry_units(result) == 333.0


def test_an_order_without_a_quantity_buys_whole_shares_by_default():
    # The engine default keeps a 2 percent buffer at the decision price: 10,00,000 x
    # 0.98 / 1,000.00 = 980 shares.
    assert entry_units(round_trip(**NO_COSTS)) == 980.0


def test_the_assumptions_say_whole_shares_keep_a_two_percent_cash_buffer():
    assert round_trip(**NO_COSTS).assumptions["sizing"] == (
        "whole shares and whole lots; an order with no quantity is sized at the decision "
        "close and keeps 2 percent of its budget in cash for the move to the next open "
        "(the engine's cash buffer, the same in live trading), then takes off its "
        "estimated charges; bt.size sizers and target() keep no buffer, and rebalance "
        "keeps its own cash_buffer"
    )


def test_the_assumptions_say_part_shares_spend_the_whole_budget_at_the_fill():
    result = round_trip(quantity_mode="fractional", **NO_COSTS)
    assert result.assumptions["sizing"] == (
        "part shares of cash equity, whole lots of derivatives; an order with no quantity "
        "is sized at the fill price and spends its whole budget less its estimated "
        "charges, with no cash buffer"
    )


def test_a_target_value_trades_whole_shares_by_default():
    # 10,000 rupees at the decision price 500.05 is 19.998 shares: 19.
    class Target(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.target(value=10_000)

    assert entry_units(run(Target)) == 19.0


def test_a_rebalance_trades_whole_shares_by_default():
    # 10,00,000 less the 2 percent cash buffer is 9,80,000 at 1,200.00: 816.67 shares,
    # so 816.
    class Rebalance(bt.Strategy):
        symbols, exchange, interval, product = ("AAA",), "NSE", "D", "CNC"

        def on_bar(self, bars):
            if len(self.data["AAA"].close) == 1:
                self.rebalance({"AAA": 100})

    days = list(bt.calendar.default_calendar().sessions("NSE", "2026-11-02", "2026-11-04")["date"])
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    prices = np.full(len(days), 1_200.0)
    bars = pd.DataFrame(
        {"open": prices, "high": prices + 10, "low": prices - 10, "close": prices},
        index=index,
    )
    bars["volume"] = 1_000.0
    result = bt.backtest(Rebalance, data={"AAA": bars}, capital=1_000_000, **NO_COSTS)
    assert entry_units(result) == 816.0
