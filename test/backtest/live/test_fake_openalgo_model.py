"""The parts of the fake OpenAlgo's broker model on their own: positions netted from
fills, the sandbox's and a broker's IDs, 1-minute candles and the day's statistics, the
manual clock, which instruments are options, the REST layer's schema words, and the log.
"""

from __future__ import annotations

import pytest
from fake_openalgo import ManualClock, ist_ns
from fake_openalgo.ids import BrokerIds, SandboxIds
from fake_openalgo.log import FakeLog
from fake_openalgo.market import Candle, Instrument, MarketState
from fake_openalgo.orders import Fill
from fake_openalgo.positions import Position
from fake_openalgo.schema import PLACE, load

NS = 1_000_000_000
OPEN_NS = ist_ns(2026, 11, 16, 9, 15)


def fill(action: str, quantity: int, price: float) -> Fill:
    return Fill("t", "o", "s", "SBIN", "NSE", "MIS", action, quantity, price, OPEN_NS)


# -- positions -------------------------------------------------------------------


def test_adding_to_a_position_averages_its_price():
    position = Position("SBIN", "NSE", "MIS")

    position.take(fill("BUY", 10, 100.0))
    position.take(fill("BUY", 10, 110.0))

    assert (position.quantity, position.average_price, position.realized) == (20, 105.0, 0.0)


def test_closing_a_position_to_flat_realizes_it_and_clears_the_average():
    position = Position("SBIN", "NSE", "MIS")
    position.take(fill("SELL", 10, 100.0))

    position.take(fill("BUY", 10, 96.0))

    assert (position.quantity, position.average_price, position.realized) == (0, 0.0, 40.0)


def test_a_fill_through_flat_opens_the_rest_at_its_price():
    position = Position("SBIN", "NSE", "MIS")
    position.take(fill("BUY", 10, 100.0))

    position.take(fill("SELL", 15, 110.0))

    assert (position.quantity, position.average_price, position.realized) == (-5, 110.0, 100.0)


# -- IDs -----------------------------------------------------------------------------


def test_sandbox_order_ids_rise_by_one_within_a_millisecond():
    ids = SandboxIds()

    assert [ids.order_id(OPEN_NS), ids.order_id(OPEN_NS)] == ["26111633300000", "26111633300001"]


def test_sandbox_trade_ids_carry_the_ist_time():
    assert SandboxIds().trade_id(OPEN_NS) == "TRADE-20261116-091500-00000001"


def test_broker_order_ids_are_fifteen_digits_in_sequence():
    ids = BrokerIds()

    assert [ids.order_id(OPEN_NS), ids.order_id(OPEN_NS)] == ["261116000000001", "261116000000002"]


# -- candles and the day's statistics --------------------------------------------------


def test_a_candle_takes_the_trades_of_its_minute():
    market = MarketState(Instrument("SBIN", "NSE"))
    for offset, price in ((0, 100.0), (30, 103.0), (59, 99.0), (60, 101.0)):
        market.trade(OPEN_NS + offset * NS, price, 1)

    candles = market.candles("2026-11-16", "2026-11-16")

    assert [(c.start_ns - OPEN_NS) // NS for c in candles] == [0, 60]
    assert [(c.open, c.high, c.low, c.close, c.volume) for c in candles] == [
        (100.0, 103.0, 99.0, 99.0, 3),
        (101.0, 101.0, 101.0, 101.0, 1),
    ]


def test_a_new_ist_day_resets_the_days_range_and_keeps_the_last_close():
    market = MarketState(Instrument("SBIN", "NSE"))
    market.trade(ist_ns(2026, 11, 13, 15, 29), 100.0, 5)

    market.trade(OPEN_NS, 104.0, 2)

    assert (market.prev_close, market.day_open, market.day_low, market.day_volume) == (
        100.0,
        104.0,
        104.0,
        2,
    )


def test_a_seeded_candle_starts_on_the_minute():
    market = MarketState(Instrument("SBIN", "NSE"))

    with pytest.raises(ValueError, match="on the minute"):
        market.seed(Candle(OPEN_NS + NS, 100.0, 101.0, 99.0, 100.5, 10))


def test_a_trade_needs_a_price_above_zero():
    market = MarketState(Instrument("SBIN", "NSE"))

    with pytest.raises(ValueError, match="price above 0"):
        market.trade(OPEN_NS, 0.0, 1)


# -- the clock -----------------------------------------------------------------------


def test_the_manual_clock_moves_only_when_told_and_keeps_what_was_slept():
    clock = ManualClock(OPEN_NS)

    clock.sleep(1.5)
    clock.advance(2)

    assert (clock.now() - OPEN_NS, clock.slept) == (3_500_000_000, [1.5])


def test_the_manual_clock_never_goes_back():
    with pytest.raises(ValueError, match="forward"):
        ManualClock(OPEN_NS).advance(-1)


# -- instruments -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exchange", "instrument_type", "is_option"),
    [
        ("NFO", "CE", True),
        ("BFO", "PE", True),
        ("NFO", "FUT", False),
        ("MCX", "CE", False),
        ("NSE", "EQ", False),
    ],
)
def test_an_option_is_a_ce_or_pe_on_nfo_or_bfo(exchange, instrument_type, is_option):
    instrument = Instrument("X", exchange, instrument_type=instrument_type)

    assert instrument.is_option is is_option


def test_an_instrument_needs_a_lot_of_at_least_one_unit():
    with pytest.raises(ValueError, match="lot_size"):
        Instrument("SBIN", "NSE", lot_size=0)


# -- the REST layer's schema -------------------------------------------------------------


def test_the_schema_names_each_missing_field_in_the_schemas_order():
    _, message = load(PLACE, {"apikey": "k", "quantity": "1"})

    assert message == (
        "{'strategy': ['Missing data for required field.'], "
        "'exchange': ['Missing data for required field.'], "
        "'symbol': ['Missing data for required field.'], "
        "'action': ['Missing data for required field.']}"
    )


def test_the_schema_refuses_a_field_it_does_not_declare():
    payload = {"apikey": "k", "strategy": "s", "exchange": "NSE", "symbol": "SBIN"}
    _, message = load(PLACE, payload | {"action": "BUY", "quantity": "1", "tag": "x"})

    assert message == "{'tag': ['Unknown field.']}"


def test_the_schema_loads_numbers_sent_as_text_and_fills_the_defaults():
    payload = {"apikey": "k", "strategy": "s", "exchange": "NSE", "symbol": "SBIN"}
    loaded, message = load(PLACE, payload | {"action": "buy", "quantity": "10"})

    assert message == ""
    assert (loaded["quantity"], loaded["pricetype"], loaded["product"], loaded["price"]) == (
        10,
        "MARKET",
        "MIS",
        0.0,
    )


# -- the log -------------------------------------------------------------------------------


def test_the_log_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown log kind"):
        FakeLog().record(OPEN_NS, "rumour", "x")
