"""A run whose history before `start` the OpenAlgo server cannot serve (2.2.1).

The bars of the run come from the local cache, filled by an earlier run that needed no
history. Then the broker login expires, and the server answers every history request
with HTTP 500. A run that needs history for its indicators stops with ServerError
instead of trading from the cached bars alone and waiting inside them for its
indicators. A broker that refuses only older dates still answers for the run's own
days: the run keeps going with the history it could load, and says why.
"""

import datetime as dt
import warnings

import pytest
from data_fixtures.fakes import FakeClient, FakeResponse
from test_quickstart import QUICKSTART, historify_rows

import openalgo.backtest as bt
from openalgo.backtest.errors import DataError, ServerError, WarmUpWarning

SERIES = ("SBIN", "NSE", "D")
EXPIRED = "Error fetching historical data: API Error: Incorrect `api_key` or `access_token`."


class ExpiringLogin(FakeClient):
    """A broker behind the server whose login can expire: from then on every history
    request is answered HTTP 500 with the broker's error, as OpenAlgo passes it on."""

    expired = False

    def _history(self, payload):
        if self.expired:
            return FakeResponse(500, {"status": "error", "message": EXPIRED})
        return super()._history(payload)


class Idle(bt.Strategy):
    """The quick start's instrument with no indicator: it needs no history."""

    symbol, exchange, interval = QUICKSTART.EmaCross.symbol, "NSE", "D"

    def on_bar(self, bar):
        pass


def run(strategy, client):
    return bt.backtest(
        strategy,
        client,
        start=QUICKSTART.START,
        end=QUICKSTART.END,
        capital=QUICKSTART.CAPITAL,
        source="api",
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """The local bars cache in a temporary folder, not the user's home."""
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path))
    return tmp_path


@pytest.fixture
def expired_after_caching(cache):
    """A client whose run bars are in the cache and whose broker login has expired."""
    client = ExpiringLogin(api={SERIES: historify_rows(QUICKSTART.daily_bars())})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(Idle, client)
    client.expired = True
    client.calls.clear()
    return client


def test_a_server_that_cannot_answer_stops_a_run_that_needs_history(expired_after_caching):
    with pytest.raises(ServerError, match="logged in to the broker") as raised:
        run(QUICKSTART.EmaCross, expired_after_caching)
    message = str(raised.value)
    assert message.startswith("SBIN: the OpenAlgo server cannot answer history requests")
    assert "access_token" in message


def test_the_server_is_asked_for_the_runs_first_day_before_the_run_stops(expired_after_caching):
    with pytest.raises(ServerError):
        run(QUICKSTART.EmaCross, expired_after_caching)
    first_day = QUICKSTART.START
    asked = expired_after_caching.history_calls(source="api")
    assert asked[-1] == ("api", "D", first_day, first_day)


def test_a_run_that_needs_no_history_still_runs_from_the_cache(expired_after_caching):
    result = run(Idle, expired_after_caching)
    assert len(result.equity) > 0


def test_a_broker_that_refuses_only_older_dates_leaves_the_run_its_warning(cache):
    client = FakeClient(
        api={SERIES: historify_rows(QUICKSTART.daily_bars())},
        api_fail_before=dt.date.fromisoformat(QUICKSTART.START),
    )
    with pytest.warns(WarmUpWarning):
        result = run(QUICKSTART.EmaCross, client)
    assert any("could not be used (ServerError" in line for line in result.warnings)
    first_day = QUICKSTART.START
    assert ("api", "D", first_day, first_day) in client.history_calls(source="api")


def test_a_server_error_is_a_data_error():
    assert issubclass(ServerError, DataError)
