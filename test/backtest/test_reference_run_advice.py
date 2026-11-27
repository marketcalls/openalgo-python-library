"""The advice a reference error or warning gives, by who meets it (the lead's ruling N5c-4 (f);
review round 1 of unit N5c, finding 3): a caller of the public reference API is told to
supply the data with `load_tables` (a user file or a symbol-master snapshot); a
`bt.backtest` run, which reads the built-in tables only, is told to pass the instrument with
`instrument=` instead. One rule makes both wordings (`reference._store.run_advice`, inside a
run's `run_wording`).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest import _futures, reference
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ReferenceDataError
from openalgo.backtest.reference import ReferenceDataWarning, default_tables

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


class NiftyFutures(bt.Strategy):
    """The current-month NIFTY future, rolled `roll` sessions before each expiry."""

    underlying, exchange, interval = "NIFTY", "NFO", "D"

    def on_bar(self, bar):
        pass


#: The warning of November's expiry, a date after the expiry rules were last checked, up
#: to its advice.
NOVEMBER_PAST_CHECK = (
    "reference data: expiry_rules[NIFTY NFO monthly] 2025-09-01 to open (builtin, verified) "
    "was last checked on 2026-11-12 and is being used for 2026-11-24. The exchange may have "
    "changed it since: "
)


def reference_warnings(call):
    """The ReferenceDataWarnings `call()` issues."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [str(w.message) for w in caught if w.category is ReferenceDataWarning]


def test_a_backtest_runs_reference_warning_gives_only_advice_a_run_can_follow(monkeypatch):
    # From 2026-10-01 to 2026-11-12 with roll=10, the run moves to December ten sessions
    # before November's 2026-11-24 expiry, which the tables were checked up to 2026-11-12.
    fresh = reference.ReferenceTables()
    monkeypatch.setattr(reference, "default_tables", lambda: fresh)
    days = pd.date_range("2026-09-01", "2026-11-12", freq="B", tz="Asia/Kolkata")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}, index=days
    )
    contracts = ("NIFTY27OCT26FUT", "NIFTY23NOV26FUT", "NIFTY29DEC26FUT")

    found = reference_warnings(
        lambda: bt.backtest(
            NiftyFutures,
            data=dict.fromkeys(contracts, bars),
            capital=1_000_000,
            start="2026-10-01",
            end="2026-11-12",
            costs=None,
            slippage=0,
            roll=10,
        )
    )

    assert found == [f"{NOVEMBER_PAST_CHECK}update openalgo. {RUN_ADVICE}"]


def test_the_public_api_keeps_the_load_tables_advice_in_its_warnings(monkeypatch):
    fresh = reference.ReferenceTables()
    monkeypatch.setattr(reference, "default_tables", lambda: fresh)
    plan = type(
        "Plan", (), {"exchange": "NFO", "start": dt.date(2026, 10, 1), "end": dt.date(2026, 11, 12)}
    )
    arguments = {"instrument": None, "roll": 10}

    found = reference_warnings(
        lambda: _futures.resolve_futures(NiftyFutures, plan, arguments, default_calendar())
    )

    assert found == [
        f"{NOVEMBER_PAST_CHECK}update openalgo, or supply a user file or a symbol-master snapshot."
    ]
