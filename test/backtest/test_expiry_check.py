"""A backtest's check of each listed futures and options contract against the expiry in the
client's master contract (`_expiry_check.py`; 18-multi-strategy 12.2, unit M11).

The contract is the NIFTY future of December 2026 on NFO, whose reference tables' listed
expiry and last trading day are both Tuesday 2026-12-29 (OpenAlgo writes "29-DEC-26").
The run is one NFO session, Monday 2026-11-16, on the user's own 1-minute bars (which also
hold Friday 2026-10-30, whose close fixes the contract's tick for November), with a
client of the data layer's fakes that also answers `symbol` as OpenAlgo's master contract
does. A run made after 2026-12-29 finds the contract expired: the server no longer lists
it, so it is not asked about.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import FakeClient

import openalgo.backtest as bt
from openalgo.backtest._expiry_check import check_listed_expiries
from openalgo.backtest.reference import ReferenceDataWarning, default_tables

FUTURE = "NIFTY29DEC26FUT"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
with warnings.catch_warnings():
    # The tables' rows were last checked on 2026-11-12.
    warnings.simplefilter("ignore", ReferenceDataWarning)
    CONTRACT = default_tables().instrument(FUTURE, "NFO", "2026-11-16")
SBIN = default_tables().instrument("SBIN", "NSE", "2026-11-16")


def present(day: dt.date) -> int:
    """UTC nanoseconds of 10:00 IST on `day`."""
    moment = dt.datetime(day.year, day.month, day.day, 10, tzinfo=IST)
    return int(moment.timestamp()) * 1_000_000_000


class MasterContract(FakeClient):
    """The data layer's fake client, which also answers `symbol` with `expiry` (or with
    `answer` as given)."""

    def __init__(self, expiry: str = "29-DEC-26", answer: dict | None = None) -> None:
        super().__init__()
        self.expiry, self.answer = expiry, answer
        self.looked: list[tuple[str, str]] = []

    def symbol(self, *, symbol: str, exchange: str) -> dict:
        self.looked.append((symbol, exchange))
        if self.answer is not None:
            return self.answer
        data = {"symbol": symbol, "exchange": exchange, "expiry": self.expiry}
        return {"status": "success", "data": data}


def master_warnings(instruments, client, day=dt.date(2026, 11, 16)) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_listed_expiries(instruments, client, present(day))
    return [str(w.message) for w in caught if "master contract" in str(w.message)]


def test_a_listed_contract_that_agrees_is_asked_about_once_and_not_warned_of():
    client = MasterContract()
    assert master_warnings([CONTRACT], client) == []
    assert client.looked == [(FUTURE, "NFO")]


def test_a_listed_contract_that_disagrees_is_a_reference_data_warning():
    with pytest.warns(ReferenceDataWarning, match="gives the expiry 2026-12-22"):
        check_listed_expiries(
            [CONTRACT], MasterContract("22-DEC-26"), present(dt.date(2026, 11, 16))
        )


def test_the_warning_names_both_dates_and_says_the_run_uses_the_tables():
    [note] = master_warnings([CONTRACT], MasterContract("22-DEC-26"))
    assert note == (
        f"NFO:{FUTURE}: OpenAlgo's master contract gives the expiry 2026-12-22, the "
        "reference tables listed expiry 2026-12-29: the run uses the reference tables, which "
        "hold the dated history; upgrade openalgo (pip install -U openalgo), and if it still "
        "disagrees, report the table row with the exchange's circular"
    )


def test_an_expired_contract_is_not_asked_about():
    client = MasterContract("22-DEC-26")
    assert master_warnings([CONTRACT], client, day=dt.date(2026, 12, 30)) == []
    assert client.looked == []


def test_a_contract_on_its_last_trading_day_is_still_listed():
    client = MasterContract()
    master_warnings([CONTRACT], client, day=dt.date(2026, 12, 29))
    assert client.looked == [(FUTURE, "NFO")]


def test_a_client_without_symbol_skips_the_check():
    assert master_warnings([CONTRACT], FakeClient()) == []


def test_cash_equity_makes_no_call():
    client = MasterContract()
    assert master_warnings([SBIN], client) == []
    assert client.looked == []


def test_a_lookup_that_fails_is_warned_of_as_not_checked():
    failed = {"status": "error", "message": "Symbol not found"}
    [note] = master_warnings([CONTRACT], MasterContract(answer=failed))
    assert note.startswith(f"NFO:{FUTURE}: its expiry could not be checked")


# ----- through bt.backtest --------------------------------------------------------------------


class Future(bt.Strategy):
    """Holds nothing on the December NIFTY future."""

    symbol, exchange, interval = FUTURE, "NFO", "1m"

    def on_bar(self, bar):
        pass


#: Friday 2026-10-30, the last session before November: its close fixes the contract's
#: tick for the month (`_slab_ticks`), and the run's own data must hold it.
TICK_REFERENCE_DAY = dt.date(2026, 10, 30)
RUN_DAY = dt.date(2026, 11, 16)
MINUTES_PER_SESSION = 375


def session_bars() -> pd.DataFrame:
    """The future's 1-minute bars of the tick's reference session and of the run's day."""
    days = (TICK_REFERENCE_DAY, RUN_DAY)
    index = pd.DatetimeIndex(
        [
            dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST) + dt.timedelta(minutes=k)
            for day in days
            for k in range(MINUTES_PER_SESSION)
        ]
    )
    opens = 24_000.0 + np.tile(np.arange(MINUTES_PER_SESSION) * 0.1, len(days))
    columns = {"open": opens, "high": opens + 1, "low": opens - 1, "close": opens + 0.5}
    return pd.DataFrame(columns | {"volume": np.full(len(index), 1_000.0)}, index=index)


def run(client) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt.backtest(
            Future,
            client,
            data=session_bars(),
            capital=1_000_000,
            start="2026-11-16",
            end="2026-11-16",
            now="2026-11-17 10:00",
        )
    return [str(w.message) for w in caught if "master contract" in str(w.message)]


def test_a_backtest_warns_of_a_listed_contracts_disagreeing_expiry_and_runs_on():
    client = MasterContract("22-DEC-26")
    [note] = run(client)
    assert "gives the expiry 2026-12-22" in note
    assert client.looked == [(FUTURE, "NFO")]


def test_a_backtest_whose_master_contract_agrees_has_no_such_warning():
    assert run(MasterContract()) == []
