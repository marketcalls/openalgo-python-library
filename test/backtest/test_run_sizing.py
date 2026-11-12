"""Sizers in a backtest (12-position-sizing 5, 03-architecture 7): the engine default,
the percent and risk sizers, lot rounding and the freeze quantity, end to end.

Capital is 10,00,000; the entry is decided at a close of 1,000.00 and fills at an open
of 1,000.00 (see accounting_runs). Each sizer's arithmetic is tested by hand in the
kernel (`sizing`); these tests check the run around it.
"""

import math

import pytest
from accounting_runs import CAPITAL, DAY, future_round_trip, round_trip, session_bars

import openalgo.backtest as bt
from openalgo.backtest import costs
from openalgo.backtest.errors import ConfigError

PRICE = 1_000.0


def entry_units(result):
    return result.fills.loc[0, "quantity_units"]


def test_the_engine_default_spends_the_capital_at_the_fill_in_fractional_mode():
    result = round_trip(costs=None, slippage=0)
    assert entry_units(result) == CAPITAL / PRICE


def test_the_engine_default_is_net_of_the_estimated_costs():
    # The estimate prices the 1,000 shares the capital buys before costs.
    estimate = costs.trade_cost("BUY", PRICE, CAPITAL / PRICE, DAY, segment="NSE_EQ_INTRADAY")
    result = round_trip(slippage=0)
    assert entry_units(result) == pytest.approx((CAPITAL - estimate.total) / PRICE)


def test_the_engine_default_keeps_a_two_percent_buffer_in_venue_mode():
    # 10,00,000 x 0.98 / 1,000.00 = 980 shares.
    result = round_trip(costs=None, slippage=0, quantity_mode="venue")
    assert entry_units(result) == 980.0


def test_the_engine_default_in_venue_mode_rounds_down_after_the_costs():
    estimate = costs.trade_cost("BUY", PRICE, 980.0, DAY, segment="NSE_EQ_INTRADAY")
    result = round_trip(slippage=0, quantity_mode="venue")
    assert entry_units(result) == math.floor((0.98 * CAPITAL - estimate.total) / PRICE)


def test_the_engine_default_of_a_future_is_one_lot():
    result = future_round_trip(costs=None, slippage=0)
    assert entry_units(result) == 75.0


def test_a_percent_sizer_takes_its_percent_of_equity_at_the_decision():
    # 50 percent of 10,00,000 at 1,000.00 is 500 shares.
    result = round_trip({"size": bt.size.percent(50)}, costs=None, slippage=0)
    assert entry_units(result) == 500.0


def test_a_sized_quantity_is_capped_at_the_freeze_quantity():
    # 30 lots of 75 = 2,250 units; the freeze quantity 1,800 is 24 whole lots.
    result = future_round_trip({"size": bt.size.lots(30)}, costs=None, slippage=0)
    assert entry_units(result) == 1_800.0


def test_explicit_lots_above_the_freeze_quantity_are_denied():
    result = future_round_trip({"lots": 30}, costs=None, slippage=0)
    assert list(result.orders["reject_code"]) == ["freeze_qty"]


def test_a_size_that_rounds_to_zero_is_skipped_and_logged():
    # 100 rupees buys 0.1 of a share: nothing in venue mode.
    result = round_trip({"size": bt.size.value(100)}, costs=None, slippage=0, quantity_mode="venue")
    assert result.fills.empty
    assert list(result.skipped["reason"]) == ["rounded_to_zero"]


def test_fixed_risk_without_a_stop_is_refused():
    with pytest.raises(ConfigError, match="needs a stop"):
        round_trip({"size": bt.size.fixed_risk(1)}, costs=None, slippage=0)


class AtrRisk(bt.Strategy):
    """Buys at the 14th one-minute close with the ATR-risk sizer."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if len(self.close) == 14:
            self.buy(size=bt.size.atr_risk(0.1, 2, period=14))


def test_atr_risk_puts_the_stop_a_multiple_of_the_atr_away():
    # Every bar's range is 2.00 and it opens at the previous close, so the ATR is 2.00;
    # the stop is 2 x 2.00 = 4.00 away and 0.1 percent of 10,00,000 is 1,000 of risk:
    # 250 shares.
    result = bt.backtest(
        AtrRisk,
        data=session_bars(),
        capital=CAPITAL,
        costs=None,
        slippage=0,
        quantity_mode="venue",
        product="CNC",
    )
    assert entry_units(result) == 250.0
