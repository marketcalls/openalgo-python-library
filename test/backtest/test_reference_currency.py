"""The built-in terms of NSE currency futures (`currency_contracts`, item C-4 of 2.4.1).

Quantity counts units of the base currency, as OpenAlgo counts it (its Charges API prices
BUY 1000 USDINR29DEC26FUT at 88.5 as a trade value of 88,500). One lot is USD, EUR or GBP
1,000 quoted in rupees per unit, so lot 1,000 and multiplier 1; or JPY 1,00,000 quoted in
rupees per 100 yen, so lot 1,00,000 and multiplier 0.01. A one-rupee move is Rs 1,000 a
lot for all four; the tick is Rs 0.0025; the contracts settle in cash. The quantity
freeze is that of NSE/CD/52655, in units: orders of up to 10,000 lots from trade date
2022-06-20, up to 5,000 lots before.
"""

import datetime as dt
import warnings

import pandas as pd
import pytest

from openalgo.backtest.errors import ReferenceDataError
from openalgo.backtest.reference import (
    ReferenceDataWarning,
    ReferenceTables,
    default_tables,
    write_symbol_master_snapshot,
)

DAY = "2026-11-23"


def instrument(symbol: str, on_date: str = DAY, tables: ReferenceTables | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)  # the rows are unverified
        return (tables or default_tables()).instrument(symbol, "CDS", on_date)


@pytest.mark.parametrize(
    ("pair", "lot_size", "multiplier"),
    [
        ("USDINR", 1_000, 1.0),  # USD 1,000 quoted per dollar
        ("EURINR", 1_000, 1.0),  # EUR 1,000 quoted per euro
        ("GBPINR", 1_000, 1.0),  # GBP 1,000 quoted per pound
        ("JPYINR", 100_000, 0.01),  # JPY 1,00,000 quoted per 100 yen
    ],
)
def test_a_currency_future_counts_units_of_its_base_currency(pair, lot_size, multiplier):
    future = instrument(f"{pair}29DEC26FUT")
    assert (future.lot_size, future.multiplier, future.tick_size) == (lot_size, multiplier, 0.0025)
    assert (future.settlement, future.kind, future.underlying) == ("cash", "future", pair)


@pytest.mark.parametrize("pair", ["USDINR", "EURINR", "GBPINR", "JPYINR"])
def test_a_one_rupee_move_is_rs_1000_a_lot(pair):
    future = instrument(f"{pair}29DEC26FUT")
    assert future.lot_size * future.multiplier == pytest.approx(1_000.0, abs=1e-9)


def test_the_last_trading_day_is_the_date_in_the_symbol():
    assert instrument("USDINR04DEC26FUT").last_trading_day == dt.date(2026, 12, 4)


@pytest.mark.parametrize(
    ("symbol", "on_date", "freeze"),
    [
        ("USDINR28JUN22FUT", "2022-06-17", 5_000_000),  # 5,000 lots of USD 1,000
        ("USDINR28JUN22FUT", "2022-06-20", 10_000_000),  # 10,000 lots of USD 1,000
        ("JPYINR28JUN22FUT", "2022-06-20", 1_000_000_000),  # 10,000 lots of JPY 1,00,000
    ],
)
def test_the_freeze_quantity_follows_the_nse_circular_in_units(symbol, on_date, freeze):
    assert instrument(symbol, on_date).freeze_qty == freeze


def test_a_date_before_the_known_freeze_has_no_row():
    with pytest.raises(ReferenceDataError, match="currency_contracts"):
        instrument("USDINR27AUG21FUT", "2021-07-30")


class CurrencyMaster:
    """A broker's CDS symbol master as `client.instruments(exchange=...)` returns it: the
    master contract of a broker that counts currency in contracts gives lotsize 1."""

    def instruments(self, *, exchange=None):
        columns = ["symbol", "brsymbol", "name", "exchange", "token", "expiry", "strike"]
        columns += ["lotsize", "instrumenttype", "tick_size"]
        row = ["USDINR29DEC26FUT", "x", "USDINR", "CDS", "1", "29-DEC-26", -1.0, 1, "FUT", 0.0025]
        return pd.DataFrame([row], columns=columns)


def test_a_snapshot_that_gives_a_currency_lot_of_one_does_not_change_the_lot(tmp_path):
    write_symbol_master_snapshot(
        CurrencyMaster(), tmp_path, exchanges=("CDS",), as_of=DAY, broker="zerodha"
    )
    future = instrument("USDINR29DEC26FUT", tables=ReferenceTables(snapshots=tmp_path))
    assert (future.lot_size, future.multiplier) == (1_000, 1.0)
