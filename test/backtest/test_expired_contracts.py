"""A run that needs the history of a futures contract the server no longer lists.

The OpenAlgo server checks a history request's symbol against the master contract it
downloaded from the broker, and answers HTTP 400 "Symbol '...' not found for exchange
'...'" for one it does not hold (`validate_symbol_exchange` in the server's history
service). A contract leaves the master contract once it has expired, and brokers stop
serving its history, so a futures run that starts before the last expiry stops there.
The data layer raises SymbolNotFoundError for that answer, and the run says that the
contract expired, on which date, and what to do.

The fake server below holds no symbol at all. NIFTY's October 2026 contract expires on
Tuesday 2026-10-27 (the last Tuesday of the month), so a current-month NIFTY run from
2026-10-01 asks for NIFTY27OCT26FUT first.
"""

import datetime as dt
import warnings

import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    FakeClient,
    FakeResponse,
    SdkOnlyClient,
    at_ist,
    server_load,
)

import openalgo.backtest as bt
from openalgo.backtest.errors import DataError, SymbolNotFoundError
from openalgo.backtest.reference import ReferenceDataWarning

OCTOBER = "NIFTY27OCT26FUT"
# A week of the data layer's own tests: weekday sessions, read on the Monday after.
FIRST, LAST = dt.date(2024, 1, 1), dt.date(2024, 1, 5)
AFTER = at_ist(dt.date(2024, 1, 8), 18)


def load_week(client):
    """A week of daily bars from the broker (named, so the first request is history)."""
    return server_load(
        client, "D", FIRST, LAST, calendar=FakeCalendar(), now_ns=AFTER, source="api",
        cache=False, broker="zerodha",
    )  # fmt: skip


def not_found(symbol: str, exchange: str) -> FakeResponse:
    """The server's answer for a symbol its master contract does not hold."""
    return FakeResponse(
        400,
        {
            "status": "error",
            "message": f"Symbol '{symbol}' not found for exchange '{exchange}'. Please verify "
            "the symbol name and ensure master contracts are downloaded.",
        },
    )


class EmptyMasterContract(FakeClient):
    """A server whose master contract lists no symbol: every history request is refused
    with the server's HTTP 400 for an unknown symbol."""

    def _history(self, payload):
        return not_found(payload["symbol"], payload["exchange"])


class NiftyMonthly(bt.Strategy):
    """Holds nothing: the run only has to load its contracts."""

    underlying, exchange, interval = "NIFTY", "NFO", "D"

    def on_bar(self, bar):
        pass


def nifty_run(end: dt.date, now: str):
    with warnings.catch_warnings():
        # The reference tables' contract dates were last checked on 2026-11-12.
        warnings.simplefilter("ignore", ReferenceDataWarning)
        return bt.backtest(
            NiftyMonthly,
            EmptyMasterContract(),
            start=dt.date(2026, 10, 1),
            end=end,
            capital=300_000,
            source="api",
            now=now,
        )


@pytest.mark.parametrize(
    "client", [EmptyMasterContract(), SdkOnlyClient(EmptyMasterContract())], ids=["pool", "sdk"]
)
def test_the_servers_unknown_symbol_answer_is_a_symbol_not_found_error(client):
    with pytest.raises(SymbolNotFoundError, match="HTTP 400: Symbol 'X' not found") as caught:
        load_week(client)
    assert (caught.value.symbol, caught.value.exchange) == ("X", "NSE")


def test_another_bad_request_stays_a_plain_data_error():
    client = FakeClient()
    client.queue(400, {"status": "error", "message": {"interval": ["Invalid interval"]}})
    with pytest.raises(DataError) as caught:
        load_week(client)
    assert not isinstance(caught.value, SymbolNotFoundError)


def test_a_contract_that_expired_before_today_says_so_and_what_to_do():
    with pytest.raises(SymbolNotFoundError) as caught:
        nifty_run(end=dt.date(2026, 11, 11), now="2026-11-12")
    assert str(caught.value) == (
        "NFO:NIFTY27OCT26FUT: the contract expired on 2026-10-27 and the OpenAlgo server no "
        "longer lists it: brokers stop serving an expired contract's history. Start the run "
        "after 2026-10-27, or run on stored bars: data= a dict by contract symbol holding "
        "NIFTY27OCT26FUT and every other contract the run trades"
    )
    assert "not found for exchange" in str(caught.value.__cause__)


def test_a_contract_not_yet_expired_keeps_the_servers_answer():
    with pytest.raises(SymbolNotFoundError, match=f"Symbol '{OCTOBER}' not found") as caught:
        nifty_run(end=dt.date(2026, 10, 20), now="2026-10-21")
    assert "expired" not in str(caught.value)
