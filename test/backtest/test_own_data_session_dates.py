"""Daily and longer bars indexed by naive midnight dates are session dates and need no
`tz=` (14-strategy-api 6): `client.history(...)` returns daily bars that way, and a date
has no clock time a wrong time zone could shift. They are read as IST session dates, as
`tz='Asia/Kolkata'` reads them, and the Assumptions block says so. Intraday bars, and
daily bars stamped at a time of day, still need `tz=`: a wrong guess there moves every
bar by 5 hours 30 minutes.

The data: SBIN on the five NSE sessions 2025-06-30 to 2025-07-04, flat at 100.00, each
row stamped at naive midnight of its date.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.data import Frame
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAYS = [dt.date(2025, 6, 30) + dt.timedelta(days=k) for k in range(5)]
SESSION_DATES = "SBIN: naive midnight timestamps read as IST session dates (daily bars need no tz=)"


def naive_daily_frame(at: dt.time = dt.time(0, 0)) -> pd.DataFrame:
    index = pd.DatetimeIndex([dt.datetime.combine(day, at) for day in DAYS], name="timestamp")
    prices = {name: 100.0 for name in ("open", "high", "low", "close")}
    return pd.DataFrame({**prices, "volume": 1_000.0}, index=index)


class Holds(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "D"


def run(data, **settings) -> bt.Result:
    return bt.backtest(Holds, data=data, capital=100_000, costs=None, **settings)


def test_daily_bars_on_naive_midnight_dates_run_without_tz_and_give_the_run_its_dates():
    arguments = run(naive_daily_frame()).manifest["arguments"]
    assert (arguments["start"], arguments["end"]) == ("2025-06-30", "2025-07-04")


def test_naive_midnight_dates_are_read_as_ist_session_dates():
    closes = run(naive_daily_frame()).equity.index
    assert list(closes) == [
        pd.Timestamp(dt.datetime.combine(day, dt.time(15, 30), tzinfo=IST)) for day in DAYS
    ]


def test_the_assumptions_say_the_dates_were_read_as_ist_session_dates():
    assert run(naive_daily_frame()).assumptions["timestamps"] == SESSION_DATES


def test_a_declared_tz_reads_the_same_dates_and_assumes_nothing():
    declared = run(naive_daily_frame(), tz="Asia/Kolkata")
    assert list(declared.equity.index) == list(run(naive_daily_frame()).equity.index)
    assert "timestamps" not in declared.assumptions


def test_a_csv_file_of_daily_dates_needs_no_tz(tmp_path):
    path = tmp_path / "SBIN.csv"
    frame = naive_daily_frame()
    frame.index = frame.index.strftime("%Y-%m-%d")
    frame.to_csv(path)
    assert run(str(path)).assumptions["timestamps"] == SESSION_DATES


def test_naive_daily_stamps_with_a_time_of_day_still_need_a_tz():
    with pytest.raises(ConfigError, match="carry no time zone"):
        run(naive_daily_frame(at=dt.time(9, 15)))


def test_weekly_bars_on_naive_midnight_dates_need_no_tz():
    mondays = pd.DatetimeIndex([dt.datetime(2025, 6, 30), dt.datetime(2025, 7, 7)])
    prices = {name: [100.0, 100.0] for name in ("open", "high", "low", "close")}
    frame = pd.DataFrame(prices, index=mondays)
    bars = Frame(frame).bars("SBIN", "NSE", "W", dt.date(2025, 6, 30), dt.date(2025, 7, 11))
    assert len(bars) == 2


def test_naive_midnight_stamps_of_intraday_bars_still_need_a_tz():
    with pytest.raises(ConfigError, match="carry no time zone"):
        Frame(naive_daily_frame()).bars("SBIN", "NSE", "15m", DAYS[0], DAYS[-1])
