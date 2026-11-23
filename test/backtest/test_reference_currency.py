"""The built-in terms of NSE currency futures (`currency_contracts`, item C-4 of 2.4.1).

One lot is one contract (OpenAlgo's master contract gives lotsize 1); a contract is USD,
EUR or GBP 1,000 quoted in rupees per unit, or JPY 1,00,000 quoted per 100 yen, so a
one-rupee move is Rs 1,000 a contract for all four; the tick is Rs 0.0025; the contracts
settle in cash. The quantity freeze is that of NSE/CD/52655: orders of up to 10,000 lots
from trade date 2022-06-20, up to 5,000 before.
"""

import datetime as dt
import warnings

import pytest

from openalgo.backtest.errors import ReferenceDataError
from openalgo.backtest.reference import ReferenceDataWarning, default_tables

DAY = "2026-11-23"


def instrument(symbol: str, on_date: str = DAY):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)  # the rows are unverified
        return default_tables().instrument(symbol, "CDS", on_date)


@pytest.mark.parametrize("pair", ["USDINR", "EURINR", "GBPINR", "JPYINR"])
def test_a_currency_future_is_one_contract_of_a_thousand_price_units(pair):
    future = instrument(f"{pair}29DEC26FUT")
    assert (future.lot_size, future.multiplier, future.tick_size) == (1, 1000.0, 0.0025)
    assert (future.settlement, future.kind, future.underlying) == ("cash", "future", pair)


def test_the_last_trading_day_is_the_date_in_the_symbol():
    assert instrument("USDINR04DEC26FUT").last_trading_day == dt.date(2026, 12, 4)


@pytest.mark.parametrize(("on_date", "freeze"), [("2022-06-17", 5_000), ("2022-06-20", 10_000)])
def test_the_freeze_quantity_follows_the_nse_circular(on_date, freeze):
    assert instrument("USDINR28JUN22FUT", on_date).freeze_qty == freeze


def test_a_date_before_the_known_freeze_has_no_row():
    with pytest.raises(ReferenceDataError, match="currency_contracts"):
        instrument("USDINR27AUG21FUT", "2021-07-30")
