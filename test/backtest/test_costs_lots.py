"""Costs counted in lots (item C-5 of 2.4.1): `lots=` with `symbol=` and `exchange=` on
`trade_cost`, `round_trip_cost` and `trade_costs`.

A commodity or currency trader counts in lots, as the broker's calculator does (its
quantity 1 is one lot). The helpers resolve the lot size and the multiplier of the
contract from the reference tables for the trade date and charge lots x lot_size x
multiplier price units. The acceptance figures are the maintainer's calculator
screenshots of 2026-11-23, in calculator rounding under brokerage="zerodha".
"""

import pytest

from openalgo.backtest import costs
from openalgo.backtest.errors import CostError, ReferenceDataError

DAY = "2026-11-23"
ZERODHA_CALCULATOR = {"brokerage": "zerodha", "rounding": "calculator"}


def test_one_lot_of_alumini_is_a_thousand_price_units():
    # ALUMINI: lot 1 (1 MT) quoted per kg, multiplier 1,000. Buy 250, sell 252: turnover
    # 5,02,000; brokerage 40.00; exchange 0.0021 percent 10.54; GST 18 percent of 40 +
    # 10.54 + 0.50 = 9.19; CTT 0.01 percent of 2,52,000 = 25.20; SEBI 0.50; stamp 0.002
    # percent of 2,50,000 = 5; total 90.43.
    by_lots = costs.round_trip_cost(
        250, 252, date=DAY, lots=1, symbol="ALUMINI31DEC26FUT", exchange="MCX",
        segment="MCX_FUT", **ZERODHA_CALCULATOR,
    )  # fmt: skip
    by_units = costs.round_trip_cost(250, 252, 1_000, DAY, segment="MCX_FUT", **ZERODHA_CALCULATOR)
    assert by_lots.total == by_units.total == 90.43
    assert by_lots.as_dict() == by_units.as_dict()


@pytest.mark.parametrize(
    ("symbol", "exchange", "buy", "sell", "total"),
    [
        # COPPER option, one lot of 2,500 units: 452.28 (test_costs_currency_and_mcx_options).
        ("COPPER24DEC26900CE", "MCX", 100.0, 110.0, 452.28),
        # USDINR, one lot of USD 1,000: future 48.14, option 116.42.
        ("USDINR29DEC26FUT", "CDS", 88.33, 88.3525, 48.14),
        ("USDINR29DEC2688.5CE", "CDS", 88.33, 88.3525, 116.42),
    ],
)
def test_one_lot_reproduces_the_calculator(symbol, exchange, buy, sell, total):
    cost = costs.round_trip_cost(
        buy, sell, date=DAY, lots=1, symbol=symbol, exchange=exchange, product="NRML",
        **ZERODHA_CALCULATOR,
    )  # fmt: skip
    assert cost.total == total


def test_the_segment_follows_the_symbol_when_only_the_product_is_given():
    # An MCX COPPER option is MCX_OPT; a USDINR future is CDS_FUT.
    option = costs.trade_cost(
        "SELL", 110, date=DAY, lots=1, symbol="COPPER24DEC26900CE", exchange="MCX", product="NRML"
    )
    future = costs.trade_cost(
        "BUY", 88.33, date=DAY, lots=1, symbol="USDINR29DEC26FUT", exchange="CDS", product="NRML"
    )
    assert (option.segment, future.segment) == ("MCX_OPT", "CDS_FUT")


def test_two_lots_cost_what_twice_the_price_units_cost():
    # Two USDINR lots are 2,000 units of the quoted price.
    by_lots = costs.trade_cost(
        "BUY", 88.33, date=DAY, lots=2, symbol="USDINR29DEC26FUT", exchange="CDS", product="NRML"
    )
    by_units = costs.trade_cost("BUY", 88.33, 2_000, DAY, segment="CDS_FUT")
    assert by_lots.as_dict() == by_units.as_dict()


def test_trade_costs_take_lots_per_row():
    # One ALUMINI lot is 1,000 units, so lots 1 and 3 are 1,000 and 3,000 units.
    by_lots = costs.trade_costs(
        ["BUY", "SELL"], [250.0, 252.0], date=DAY, segment="MCX_FUT", lots=[1, 3],
        symbol="ALUMINI31DEC26FUT", exchange="MCX",
    )  # fmt: skip
    by_units = costs.trade_costs(["BUY", "SELL"], [250.0, 252.0], [1_000, 3_000], DAY, "MCX_FUT")
    assert by_lots.equals(by_units)


CONTRACT = {"symbol": "USDINR29DEC26FUT", "exchange": "CDS", "product": "NRML"}
QUANTITY_ERRORS = {
    "both": ({"quantity": 1_000, "lots": 1, **CONTRACT}, "either quantity .* or lots"),
    "neither": (CONTRACT, "either quantity .* or lots"),
    "lots without a symbol": ({"lots": 1, "exchange": "CDS", "product": "NRML"}, "symbol="),
    "a part lot": ({"lots": 1.5, **CONTRACT}, "lots must be a whole number"),
    "negative lots": ({"lots": -1, **CONTRACT}, "lots must be a whole number"),
}


@pytest.mark.parametrize(("arguments", "words"), QUANTITY_ERRORS.values(), ids=QUANTITY_ERRORS)
def test_a_quantity_given_both_ways_neither_way_or_as_part_lots_is_refused(arguments, words):
    with pytest.raises(CostError, match=words):
        costs.trade_cost("BUY", 88.33, date=DAY, **arguments)


def test_a_symbol_without_its_exchange_is_refused():
    with pytest.raises(CostError, match="symbol= needs exchange="):
        costs.trade_cost(
            "BUY", 88.33, date=DAY, lots=1, symbol="USDINR29DEC26FUT", segment="CDS_FUT"
        )


def test_lots_on_cash_equity_are_refused():
    with pytest.raises(CostError, match="lots= counts futures and options"):
        costs.trade_cost("BUY", 500, date=DAY, lots=1, symbol="SBIN", exchange="NSE", product="CNC")


def test_a_contract_the_reference_tables_do_not_cover_raises_their_error():
    # BSE currency futures have no built-in table: the backtest's ReferenceDataError.
    with pytest.raises(ReferenceDataError):
        costs.trade_cost(
            "BUY", 88.33, date=DAY, lots=1, symbol="USDINR29DEC26FUT", exchange="BCD",
            segment="CDS_FUT",
        )  # fmt: skip
