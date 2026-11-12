"""The instruments of a run (14-strategy-api 6): as given, the generic cash-equity
default, or the reference tables, each stated in the Assumptions block.

Reference values used: the NIFTY futures lot was 75 from 2024-11-20 (NSE/FAOP/64625),
the NIFTY freeze quantity 1,800 (NSE/FAOP/54248), the NIFTY futures tick five paise for
every price until 2025-04-14 and by price slab from 2025-04-15 (NSE/FAOP/67135); the
February 2025 monthly contract expired on Thursday 27 February 2025.
"""

import datetime as dt
from types import SimpleNamespace

import pytest
from strategy_samples import run

import openalgo.backtest as bt
from openalgo.backtest._instruments import resolve_instruments
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import Instrument


class Quarterly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"
    product = "CNC"


def test_cash_equity_without_an_instrument_is_the_generic_one_and_says_so():
    result = run(Quarterly)
    assert result.assumptions["instruments"] == (
        "SBIN: generic cash-equity instrument assumed (lot 1, tick 0.05, no freeze quantity, "
        "NSE equity costs); pass instrument= for its own terms"
    )
    assert result.manifest["instruments"][0]["generic"] is True


def test_a_given_instrument_is_used_as_given():
    given = Instrument("SBIN", "NSE", "equity", lot_size=1, tick_size=0.01, freeze_qty=50_000)
    result = run(Quarterly, instrument=given)
    assert result.assumptions["instruments"] == "SBIN: as given, lot 1, tick 0.01, freeze 50000"


def test_an_instrument_of_another_exchange_is_refused():
    given = Instrument("SBIN", "BSE", "equity", lot_size=1, tick_size=0.05)
    with pytest.raises(ConfigError, match="is on BSE, the strategy trades on NSE"):
        run(Quarterly, instrument=given)


def test_instrument_takes_instruments_only():
    with pytest.raises(ConfigError, match="instrument= takes an Instrument"):
        run(Quarterly, instrument={"SBIN": "lot 1"})


INFY = Instrument("INFY", "NSE", "equity", lot_size=1, tick_size=0.05)


def test_an_instrument_under_the_key_of_another_symbol_is_refused():
    with pytest.raises(ConfigError, match="the key 'SBIN' holds the Instrument of 'INFY'"):
        run(Quarterly, instrument={"SBIN": INFY})


@pytest.mark.parametrize("given", [INFY, {"INFY": INFY}], ids=["an Instrument", "a dict"])
def test_an_instrument_of_a_symbol_the_strategy_does_not_trade_is_refused(given):
    with pytest.raises(ConfigError, match=r"instrument= gives INFY, which this strategy does not"):
        run(Quarterly, instrument=given)


def plan(symbol, day):
    return SimpleNamespace(symbols=(symbol,), exchange="NFO", start=day)


def test_a_future_takes_its_terms_from_the_reference_tables():
    instruments, lines, version = resolve_instruments(
        plan("NIFTY27FEB25FUT", dt.date(2025, 2, 3)), None, default_calendar()
    )
    future = instruments[0]
    assert (future.lot_size, future.tick_size, future.freeze_qty) == (75, 0.05, 1800)
    assert future.last_trading_day == dt.date(2025, 2, 27)
    assert lines["NIFTY27FEB25FUT"].startswith(
        "NIFTY27FEB25FUT: lot 75, tick 0.05, freeze 1800 from the reference tables (builtin:"
    )
    assert version.startswith("builtin:")


def test_a_tick_by_price_slab_is_left_for_the_runs_data():
    # From 2025-04-15 the NIFTY futures tick follows a price slab; the tables cannot
    # choose it, so it is resolved later from the data (`_slab_ticks`).
    instruments, _, _ = resolve_instruments(
        plan("NIFTY31JUL25FUT", dt.date(2025, 7, 1)), None, default_calendar()
    )
    assert instruments[0].tick_size is None
