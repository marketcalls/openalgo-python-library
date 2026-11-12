"""`load_bars` on real data from a running OpenAlgo server (read-only).

The loads cover the last three sessions of a near-month MCX future and of an NSE
equity, at 1 minute and daily, from the broker (`source="api"`), Historify
(`source="db"`) and the automatic choice. Each load runs once per module; the local
cache is off, so nothing is written outside the test run.
"""

import datetime as dt
import time

import pandas as pd
import pytest
from live_support import (
    IST,
    live_only,
    near_month_future,
    session_is_open,
    today_ist,
)

from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data import load_bars
from openalgo.backtest.errors import NoDataError

pytestmark = [
    live_only,
    pytest.mark.filterwarnings("ignore::openalgo.backtest.reference.ReferenceDataWarning"),
]

MCX_UNDERLYING = "CRUDEOIL"
EQUITY = "SBIN"
SESSIONS = 3
# Calendar days searched back for the last SESSIONS sessions (covers holiday weeks).
LOOKBACK_DAYS = 14


def _last_sessions(exchange: str) -> tuple[dt.date, dt.date]:
    """First and last IST date of the last SESSIONS sessions that have opened."""
    calendar, today = default_calendar(), today_ist()
    frame = calendar.sessions(exchange, today - dt.timedelta(days=LOOKBACK_DAYS), today)
    opened = frame[frame["open"] <= time.time_ns()]
    return opened["date"].iloc[-SESSIONS], opened["date"].iloc[-1]


def _ist_day(stamps_ns: pd.Series) -> pd.Series:
    return pd.to_datetime(stamps_ns, unit="ns", utc=True).dt.tz_convert(IST).dt.date


@pytest.fixture(scope="module")
def mcx_symbol(client) -> str:
    return near_month_future(client, MCX_UNDERLYING, "MCX")["symbol"]


@pytest.fixture(scope="module")
def series(client, mcx_symbol):
    """load(symbol, exchange, interval, source) -> (bars, meta, load start, load end), cached."""
    loaded: dict[tuple, tuple] = {}

    def load(symbol: str, exchange: str, interval: str, source: str):
        key = (symbol, exchange, interval, source)
        if key not in loaded:
            start, end = _last_sessions(exchange)
            before = time.time_ns()
            bars, meta = load_bars(
                symbol, exchange, interval, start, end, source=source, client=client, cache=False
            )
            loaded[key] = (bars, meta, before, time.time_ns())
        return loaded[key]

    return load


def _symbol(kind: str, mcx_symbol: str) -> tuple[str, str]:
    return (mcx_symbol, "MCX") if kind == "mcx" else (EQUITY, "NSE")


@pytest.mark.parametrize("source", ["api", "auto"])
def test_mcx_minute_bars_open_and_close_with_every_session(series, mcx_symbol, source):
    bars, _, before, _ = series(mcx_symbol, "MCX", "1m", source)
    calendar = default_calendar()
    for day, rows in bars.groupby(_ist_day(bars["ts_open"])):
        session_open, session_close = calendar.session("MCX", day)
        assert rows["ts_open"].iloc[0] == session_open, f"first bar of {day}"
        if session_close <= before:
            assert rows["ts_close"].iloc[-1] == session_close, f"last bar of {day}"


@pytest.mark.parametrize("source", ["api", "auto"])
def test_equity_minute_bars_stay_inside_the_session(series, source):
    bars, _, _, _ = series(EQUITY, "NSE", "1m", source)
    calendar = default_calendar()
    for day, rows in bars.groupby(_ist_day(bars["ts_open"])):
        session_open, session_close = calendar.session("NSE", day)
        assert rows["ts_open"].iloc[0] == session_open, f"first bar of {day}"
        assert rows["ts_close"].iloc[-1] <= session_close, f"last bar of {day}"


@pytest.mark.parametrize(
    ("kind", "interval"), [("mcx", "1m"), ("mcx", "D"), ("equity", "1m"), ("equity", "D")]
)
def test_no_bar_closes_after_the_load(series, mcx_symbol, kind, interval):
    bars, _, _, after = series(*_symbol(kind, mcx_symbol), interval, "api")
    assert int(bars["ts_close"].max()) <= after


def test_the_daily_bar_of_an_open_mcx_session_is_left_out(series, mcx_symbol):
    if not session_is_open("MCX"):
        pytest.skip("the MCX session is not open now")
    bars, meta, _, _ = series(mcx_symbol, "MCX", "D", "api")
    assert today_ist() not in set(_ist_day(bars["ts_open"]))
    assert meta.obtained_end < today_ist().isoformat()


@pytest.mark.parametrize(
    ("kind", "interval"), [("mcx", "1m"), ("mcx", "D"), ("equity", "1m"), ("equity", "D")]
)
def test_a_broker_load_says_where_its_bars_came_from(series, broker, mcx_symbol, kind, interval):
    _, meta, _, _ = series(*_symbol(kind, mcx_symbol), interval, "api")
    assert (meta.source, meta.source_requested, meta.broker) == ("api", "api", broker)
    assert meta.short_history is False
    assert meta.quality.errors == []
    assert meta.rows > 0


@pytest.mark.parametrize(("kind", "interval"), [("mcx", "1m"), ("equity", "D")])
def test_auto_reports_the_probe_behind_its_choice(series, mcx_symbol, kind, interval):
    _, meta, _, _ = series(*_symbol(kind, mcx_symbol), interval, "auto")
    assert meta.source_requested == "auto"
    assert meta.auto_probe is not None
    assert meta.auto_probe.decision == meta.source


@pytest.mark.parametrize(("kind", "interval"), [("mcx", "1m"), ("equity", "D")])
def test_historify_loads_or_says_it_lacks_the_range(series, mcx_symbol, kind, interval):
    try:
        bars, meta, _, _ = series(*_symbol(kind, mcx_symbol), interval, "db")
    except NoDataError as error:
        assert "Historify" in str(error)
    else:
        assert meta.source == "db"
        assert len(bars) > 0
