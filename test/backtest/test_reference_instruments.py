"""The Instrument record and the generic cash-equity default (FR-INST-02)."""

import dataclasses
import datetime as dt

import pytest

from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference.instruments import GENERIC_LOT_SIZE, GENERIC_TICK_SIZE, Instrument

EXPIRY = dt.date(2024, 11, 28)


def _nifty_future(lot_size=25):
    return Instrument(
        symbol="NIFTY28NOV24FUT",
        exchange="NFO",
        kind="future",
        lot_size=lot_size,
        tick_size=0.05,
        expiry=EXPIRY,
        underlying="NIFTY",
        settlement="cash",
    )


def _mcx_future(symbol, underlying, lot_size, multiplier):
    return Instrument(
        symbol=symbol,
        exchange="MCX",
        kind="future",
        lot_size=lot_size,
        tick_size=1.0,
        multiplier=multiplier,
        expiry=dt.date(2024, 12, 5),
        underlying=underlying,
        settlement="physical",
    )


def test_instrument_fields_are_the_agreed_twelve_in_order():
    names = [field.name for field in dataclasses.fields(Instrument)]
    assert names[:12] == [
        "symbol",
        "exchange",
        "kind",
        "lot_size",
        "tick_size",
        "freeze_qty",
        "multiplier",
        "expiry",
        "strike",
        "option_type",
        "underlying",
        "settlement",
    ]


def test_instrument_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Instrument.cash_equity("INFY").lot_size = 5


def test_instruments_with_the_same_terms_are_equal_whatever_their_provenance():
    plain = Instrument(symbol="INFY", exchange="NSE", kind="equity", tick_size=0.05)
    sourced = dataclasses.replace(plain, provenance=(("tick_size", "a circular", "verified"),))
    assert plain == sourced and hash(plain) == hash(sourced)


def test_generic_cash_equity_default_is_lot_one_and_tick_five_paise():
    assert (GENERIC_LOT_SIZE, GENERIC_TICK_SIZE) == (1, 0.05)
    inst = Instrument.cash_equity("ANYTHING")
    assert (inst.exchange, inst.kind, inst.lot_size, inst.tick_size) == ("NSE", "equity", 1, 0.05)
    assert (inst.freeze_qty, inst.multiplier, inst.settlement) == (None, 1.0, "delivery")


def test_generic_cash_equity_default_is_flagged_as_an_assumption():
    inst = Instrument.cash_equity("ANYTHING", "BSE")
    assert inst.generic is True
    assert inst.provenance == (
        ("lot_size", "generic cash-equity default", "assumed"),
        ("tick_size", "generic cash-equity default", "assumed"),
    )


def test_an_instrument_built_from_terms_is_not_flagged_generic():
    assert _nifty_future().generic is False


def test_pnl_is_price_move_times_quantity_in_units():
    # Two lots of 25 units, a ten point move.
    assert _nifty_future(lot_size=25).pnl(24000.0, 24010.0, 50) == 500.0


def test_notional_is_price_times_quantity_in_units():
    assert _nifty_future(lot_size=25).notional(24000.0, 50) == 1_200_000.0


@pytest.mark.parametrize(
    "symbol,underlying,lot_size,multiplier,contract_multiplier",
    [
        ("GOLD05DEC24FUT", "GOLD", 1, 100.0, 100),  # 1 kg quoted per 10 g
        ("GOLDM05DEC24FUT", "GOLDM", 100, 0.1, 10),  # 100 g quoted per 10 g
        ("CRUDEOIL19NOV24FUT", "CRUDEOIL", 100, 1.0, 100),  # 100 barrels quoted per barrel
        ("NATURALGAS26NOV24FUT", "NATURALGAS", 1250, 1.0, 1250),
        ("ZINC29NOV24FUT", "ZINC", 5, 1000.0, 5000),  # 5 tonnes quoted per kg
    ],
)
def test_contract_multiplier_is_lot_size_times_per_unit_multiplier(
    symbol, underlying, lot_size, multiplier, contract_multiplier
):
    inst = _mcx_future(symbol, underlying, lot_size, multiplier)
    assert inst.contract_multiplier == contract_multiplier


def test_pnl_of_one_lot_uses_the_per_unit_multiplier():
    gold = _mcx_future("GOLD05DEC24FUT", "GOLD", lot_size=1, multiplier=100.0)
    # One lot is quantity 1 (a kilogram); a ten rupee move per 10 g is 1000 rupees.
    assert gold.pnl(75000.0, 75010.0, 1) == 1000.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "bond"},
        {"lot_size": 0},
        {"lot_size": 1.5},
        {"lot_size": True},
        {"tick_size": 0},
        {"tick_size": -0.05},
        {"freeze_qty": 0},
        {"freeze_qty": 1800.5},
        {"multiplier": 0},
        {"settlement": "barter"},
        {"symbol": ""},
        {"exchange": ""},
        {"expiry": EXPIRY},
        {"strike": 100.0},
        {"option_type": "CE"},
    ],
)
def test_equity_with_an_invalid_value_is_rejected(overrides):
    values = {"symbol": "INFY", "exchange": "NSE", "kind": "equity", "tick_size": 0.05}
    values.update(overrides)
    with pytest.raises(ConfigError):
        Instrument(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "future"},  # no expiry
        {"kind": "future", "expiry": dt.datetime(2024, 11, 28, 15, 30)},  # a datetime
        {"kind": "option", "expiry": EXPIRY, "strike": 100.0},  # no option type
        {"kind": "option", "expiry": EXPIRY, "strike": 100.0, "option_type": "XX"},
        {"kind": "option", "expiry": EXPIRY, "option_type": "CE"},  # no strike
        {"kind": "option", "expiry": EXPIRY, "strike": 0.0, "option_type": "CE"},
        {"kind": "option", "strike": 100.0, "option_type": "CE"},  # no expiry
    ],
)
def test_derivative_with_missing_or_invalid_contract_terms_is_rejected(values):
    with pytest.raises(ConfigError):
        Instrument(symbol="X", exchange="NFO", **values)


def test_option_with_all_its_terms_is_a_tradable_derivative():
    option = Instrument(
        symbol="NIFTY28NOV2424000CE",
        exchange="NFO",
        kind="option",
        lot_size=25,
        expiry=EXPIRY,
        strike=24000.0,
        option_type="CE",
        underlying="NIFTY",
        settlement="cash",
    )
    assert option.is_derivative and option.is_tradable


def test_index_is_neither_tradable_nor_a_derivative():
    index = Instrument(symbol="NIFTY", exchange="NSE_INDEX", kind="index", settlement="none")
    assert not index.is_tradable and not index.is_derivative


def test_to_dict_gives_plain_values_with_an_iso_expiry():
    record = _nifty_future().to_dict()
    assert record["expiry"] == "2024-11-28"
    assert (record["symbol"], record["lot_size"], record["settlement"]) == (
        "NIFTY28NOV24FUT",
        25,
        "cash",
    )


# ---- last trading day and contract identity -------------------------------------------------


def test_contract_dates_follow_the_twelve_agreed_fields():
    names = [field.name for field in dataclasses.fields(Instrument)]
    assert names[12:14] == ["last_trading_day", "contract_date"]


def test_last_trading_day_and_contract_date_appear_in_to_dict():
    future = dataclasses.replace(
        _nifty_future(), last_trading_day=dt.date(2024, 11, 27), contract_date=EXPIRY
    )
    record = future.to_dict()
    assert (record["last_trading_day"], record["contract_date"]) == ("2024-11-27", "2024-11-28")


def test_contract_key_ignores_the_listed_expiry_and_names_the_contract():
    listed_early = dataclasses.replace(
        _nifty_future(),
        symbol="NIFTY27NOV24FUT",
        expiry=dt.date(2024, 11, 27),
        contract_date=EXPIRY,
    )
    listed_late = dataclasses.replace(_nifty_future(), contract_date=EXPIRY)
    assert listed_early.contract_key == listed_late.contract_key
    assert listed_early.contract_key == ("NFO", "NIFTY", "future", EXPIRY, None, None)


@pytest.mark.parametrize("field", ["last_trading_day", "contract_date"])
def test_contract_dates_need_an_expiry_and_a_real_date(field):
    with pytest.raises(ConfigError, match=field):
        Instrument(symbol="INFY", exchange="NSE", kind="equity", **{field: EXPIRY})
    with pytest.raises(ConfigError, match=field):
        dataclasses.replace(_nifty_future(), **{field: "2024-11-28"})
