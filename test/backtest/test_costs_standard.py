"""The standard brokerage preset and the default preset (maintainer, 2026-11-23; item C-1
of 2.4.1).

Most brokers charge for equity delivery, so the default preset is "standard": equity
delivery 2.5 percent of the order value or Rs 20 per executed order, whichever is lower;
equity intraday 0.1 percent or Rs 20; futures (equity, currency and commodity) 0.03 percent
or Rs 20; options (equity, currency and commodity) Rs 20 per executed order; the DP fee is
Zerodha's Rs 13 standing in. "zerodha" stays as it is. Every figure is worked by hand in
the test's comment.
"""

import inspect

import pytest
from accounting_runs import CAPITAL, session_bars

import openalgo.backtest as bt
from openalgo.backtest import costs
from openalgo.backtest.errors import CostError

DAY = "2026-11-12"  # the day the built-in rates are verified for

# (segment, price, quantity, brokerage of one order): 2.5 percent of Rs 1,14,000 is
# Rs 2,850, so Rs 20; of Rs 500 it is Rs 12.50. 0.1 percent of Rs 1,14,000 is Rs 114, so
# Rs 20; of Rs 5,000 it is Rs 5.00.
EQUITY_ORDERS = [
    ("NSE_EQ_DELIVERY", 1_140.0, 100, 20.0),
    ("NSE_EQ_DELIVERY", 500.0, 1, 12.5),
    ("NSE_EQ_INTRADAY", 1_140.0, 100, 20.0),
    ("NSE_EQ_INTRADAY", 500.0, 10, 5.0),
]


def test_the_standard_preset_has_the_maintainers_rules():
    assert costs.standard().rules == {
        "delivery": ("rate", 0.025, 20.0),
        "intraday": ("rate", 0.001, 20.0),
        "futures": ("rate", 0.0003, 20.0),
        "options": ("flat", 20.0),
    }


@pytest.mark.parametrize(("segment", "price", "quantity", "brokerage"), EQUITY_ORDERS)
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_default_equity_brokerage_is_rs_20_or_the_lower_percentage(
    segment, price, quantity, brokerage, side
):
    cost = costs.trade_cost(side, price, quantity, DAY, segment=segment)
    assert cost.brokerage == pytest.approx(brokerage)


def test_zerodha_still_charges_no_delivery_brokerage():
    cost = costs.trade_cost(
        "BUY", 1_140.0, 100, DAY, segment="NSE_EQ_DELIVERY", brokerage="zerodha"
    )
    assert cost.brokerage == 0.0


@pytest.mark.parametrize(
    ("segment", "buy", "sell", "quantity"),
    [
        # A small futures order (0.03 percent of Rs 50,000 is Rs 15) and a large one.
        ("NFO_FUT", 500.0, 510.0, 100),
        ("NFO_FUT", 24_000.0, 24_100.0, 75),
        ("NFO_OPT", 100.0, 110.0, 400),
        ("MCX_FUT", 250.0, 252.0, 1_000),
    ],
)
def test_futures_and_options_cost_what_they_cost_under_zerodha(segment, buy, sell, quantity):
    default = costs.round_trip_cost(buy, sell, quantity, DAY, segment=segment)
    zerodha = costs.round_trip_cost(buy, sell, quantity, DAY, segment=segment, brokerage="zerodha")
    assert default.as_dict() == zerodha.as_dict()


def test_the_default_delivery_round_trip_of_the_worked_example_pays_rs_40_more():
    # The reference note's delivery example (buy 1000, sell 1100, 400 shares): Zerodha
    # 931.42 with no brokerage; standard adds Rs 20 a leg and GST on it, 0.18 x 40 = 7.20:
    # 931.42 + 40 + 7.20 = 978.62.
    cost = costs.round_trip_cost(
        1000, 1100, 400, DAY, segment="NSE_EQ_DELIVERY", rounding="calculator"
    )
    assert (cost.brokerage, cost.gst, cost.total) == (40.0, 11.99, 978.62)


def test_the_dp_charge_under_the_default_says_the_fee_stands_in():
    # Rs 13 before GST, GST 18 percent = 2.34: Rs 15.34, Zerodha's fee standing in.
    charge = costs.dp_charge(DAY)
    assert (charge.fee, charge.gst, charge.total) == (13.0, 2.34, 15.34)
    assert charge.fee_source == "zerodha_stand_in"


def test_standard_is_a_built_in_name_and_the_cost_models_default():
    assert costs.preset("standard") == costs.standard()
    assert costs.CostModel().brokerage.name == "standard"
    assert costs.CostModel().brokerage == costs.standard()


DEFAULTED = {
    "costs.trade_cost": (costs.trade_cost, "brokerage"),
    "costs.round_trip_cost": (costs.round_trip_cost, "brokerage"),
    "costs.trade_costs": (costs.trade_costs, "brokerage"),
    "costs.dp_charge": (costs.dp_charge, "brokerage"),
    "costs.CostModel": (costs.CostModel, "brokerage"),
    "bt.backtest": (bt.backtest, "costs"),
    "bt.live": (bt.live, "costs"),
}


@pytest.mark.parametrize(("function", "name"), DEFAULTED.values(), ids=DEFAULTED)
def test_every_defaulted_preset_is_standard(function, name):
    assert inspect.signature(function).parameters[name].default == "standard"


@pytest.mark.parametrize("verb", [bt.optimize, bt.robustness, bt.walk_forward])
def test_the_sweeps_take_the_backtests_default_charges(verb):
    # They pass their settings to bt.backtest and default nothing themselves.
    assert "costs" not in inspect.signature(verb).parameters


def test_an_unknown_preset_name_lists_the_built_in_presets():
    with pytest.raises(CostError, match=r"standard \(the default\), zerodha and zero_brokerage"):
        costs.preset("my_broker")


class PartSale(bt.Strategy):
    """Buys 400 SBIN at its first 1-minute bar and sells 200 of them at its third."""

    symbol = "SBIN"
    exchange = "NSE"
    interval = "1m"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=400)
        elif len(self.close) == 3:
            self.sell(qty=200)


def test_a_cnc_buy_partly_sold_the_same_day_pays_the_brokerage_of_one_order():
    # Both fills at 1,000.00 on 2026-11-04 (accounting_runs). The buy is then 200 units
    # intraday and 200 delivery, one executed order: Rs 20 under the default preset, not
    # Rs 20 a part. GST is 18 percent of brokerage 20 + exchange 12.28 (0.00297 percent
    # plus IPFT 0.0001 percent of 4,00,000) + SEBI 0.40 = 5.8824. The sale is all
    # intraday: Rs 20.
    run = bt.backtest(
        PartSale, data=session_bars(), capital=CAPITAL, base="1m", product="CNC", slippage=0
    )
    assert list(run.costs["brokerage"]) == [20.0, 20.0]
    assert run.costs["gst"].iloc[0] == pytest.approx(5.8824, abs=1e-9)
