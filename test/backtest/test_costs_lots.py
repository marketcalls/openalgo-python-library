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
from openalgo.backtest.reference import default_tables

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


@pytest.mark.parametrize(
    ("symbol", "price"),
    [
        # USD 1,000 a lot quoted per dollar: lot 1,000 x multiplier 1.
        ("USDINR29DEC26FUT", 88.33),
        # JPY 1,00,000 a lot quoted per 100 yen: lot 1,00,000 x multiplier 0.01.
        ("JPYINR29DEC26FUT", 57.5),
    ],
)
def test_one_currency_lot_is_a_thousand_price_units(symbol, price):
    by_lots = costs.trade_cost(
        "BUY", price, date=DAY, lots=1, symbol=symbol, exchange="CDS", product="NRML"
    )
    by_units = costs.trade_cost("BUY", price, 1_000, DAY, segment="CDS_FUT")
    assert by_lots.as_dict() == by_units.as_dict()


def test_two_lots_cost_what_twice_the_price_units_cost():
    # Two USDINR lots are USD 2,000: 2,000 units of the quoted price.
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
            product="NRML",
        )  # fmt: skip


@pytest.mark.parametrize("quantity", [{"quantity": 100}, {"lots": 1}], ids=["quantity", "lots"])
def test_an_index_is_refused_as_not_traded(quantity):
    with pytest.raises(CostError, match="NIFTY on NSE_INDEX is an index, which is not traded"):
        costs.trade_cost(
            "BUY", 100, date=DAY, symbol="NIFTY", exchange="NSE_INDEX", product="MIS", **quantity
        )


FUTURE = {"symbol": "USDINR29DEC26FUT", "exchange": "CDS"}
MISMATCHES = {
    "an equity segment": {"segment": "NSE_EQ_DELIVERY", **FUTURE},
    "an options segment": {"segment": "CDS_OPT", **FUTURE},
    "another exchange's futures": {"segment": "MCX_FUT", **FUTURE},
    "an instrument against the symbol": {"product": "NRML", "instrument": "OPT", **FUTURE},
}


@pytest.mark.parametrize("arguments", MISMATCHES.values(), ids=MISMATCHES)
@pytest.mark.parametrize("quantity", [{"quantity": 1_000}, {"lots": 1}], ids=["quantity", "lots"])
def test_a_segment_that_is_not_the_contracts_is_refused(arguments, quantity):
    with pytest.raises(CostError, match="not USDINR29DEC26FUT, which is CDS futures"):
        costs.trade_cost("BUY", 88.33, date=DAY, **quantity, **arguments)


def test_trade_costs_refuse_a_segment_that_is_not_the_contracts():
    with pytest.raises(CostError, match="not USDINR29DEC26FUT, which is CDS futures"):
        costs.trade_costs(["BUY"], [88.33], date=DAY, segment="NSE_EQ_DELIVERY", lots=[1], **FUTURE)


def test_trade_costs_refuse_an_exchange_without_a_symbol():
    with pytest.raises(CostError, match="exchange= is read only with symbol="):
        costs.trade_costs(["BUY"], [88.33], [1_000], DAY, "CDS_FUT", exchange="MCX")


@pytest.mark.parametrize(
    ("option", "exchange", "segment", "units"),
    [
        ("COPPER24DEC26900CE", "MCX", "MCX_OPT", 2_500),  # the COPPER futures lot
        ("USDINR29DEC2688.5CE", "CDS", "CDS_OPT", 1_000),  # the USDINR futures lot, USD 1,000
    ],
)
def test_a_lot_of_an_mcx_or_currency_option_is_its_underlyings_futures_lot(
    option, exchange, segment, units
):
    # The helpers read the contract-terms row of the underlying (mcx_contracts,
    # currency_contracts), whose contract size the option shares today. A backtest of the
    # option itself still needs a 'symbols' row for it: the built-in tables do not cover it.
    by_lots = costs.trade_cost("BUY", 100.0, date=DAY, lots=1, symbol=option, exchange=exchange,
                               product="NRML")  # fmt: skip
    by_units = costs.trade_cost("BUY", 100.0, units, DAY, segment=segment)
    assert by_lots.as_dict() == by_units.as_dict()
    with pytest.raises(ReferenceDataError, match=f"No lot_size is known for {option}"):
        default_tables().instrument(option, exchange, DAY)
