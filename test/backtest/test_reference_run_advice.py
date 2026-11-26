"""The advice a reference error gives, by who meets it (the lead's ruling N5c-4 (f)): a caller of
the public reference API is told to supply the data with `load_tables` (a user file or a
symbol-master snapshot); a `bt.backtest` run, which reads the built-in tables only, is told
to pass the instrument with `instrument=` instead. One rule makes both wordings
(`reference._store.run_advice`).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ReferenceDataError
from openalgo.backtest.reference import default_tables

#: The advice a run gives, written out.
RUN_ADVICE = (
    "Pass the instrument to the run with instrument=Instrument(...) (Instrument from "
    "openalgo.backtest.reference) with the figure you have confirmed."
)


class StockOption(bt.Strategy):
    """A strategy on an SBIN option on NFO: the reference tables do not say what SBIN is."""

    symbol, exchange, interval, product = "SBIN24NOV26600CE", "NFO", "D", "NRML"

    def on_bar(self, bar):
        pass


def test_the_public_api_keeps_the_load_tables_advice():
    with pytest.raises(ReferenceDataError) as refused:
        default_tables().underlying_type("SBIN", "NFO", dt.date(2026, 11, 16))

    assert "load_tables(user_files=[path])" in str(refused.value)


def test_a_backtest_run_gives_only_advice_a_run_can_follow():
    days = pd.date_range("2026-10-01", periods=40, freq="B", tz="Asia/Kolkata")
    bars = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000}, index=days
    )

    with pytest.raises(ReferenceDataError) as refused:
        bt.backtest(StockOption, data=bars, capital=100000, start="2026-10-15", end="2026-11-20")

    words = str(refused.value)
    assert words.endswith(RUN_ADVICE)
    assert "load_tables" not in words
    assert "write_symbol_master_snapshot" not in words
