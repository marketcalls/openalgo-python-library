"""The time zone of own data (14-strategy-api 6, FR-DATA-04): naive timestamps need
`tz=`, and with it the run takes its dates from the data; epoch numbers count from
1970-01-01 UTC, so they need no `tz=` and refuse a different one.

The data is one NSE session of 15-minute bars on 2025-07-01 (09:15 to 15:15 IST), flat
at 100.00, as IST clock times without a time zone.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.data import Files, Frame
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAY = dt.date(2025, 7, 1)
BARS = 25
# 2025-07-01 09:15 IST is 03:45 UTC: 1751341500 seconds after 1970-01-01 UTC.
FIRST_OPEN_EPOCH_SECONDS = 1_751_341_500
FIRST_OPEN_NS = FIRST_OPEN_EPOCH_SECONDS * 1_000_000_000
BAR_SECONDS = 15 * 60


def naive_ist_frame() -> pd.DataFrame:
    start = dt.datetime(2025, 7, 1, 9, 15)
    index = pd.DatetimeIndex(
        [start + dt.timedelta(minutes=15 * k) for k in range(BARS)], name="timestamp"
    )
    prices = {name: 100.0 for name in ("open", "high", "low", "close")}
    return pd.DataFrame({**prices, "volume": 1_000.0}, index=index)


class Holds(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"


def run(data, **settings):
    arguments = {"data": data, "capital": 100_000, "costs": None, "base": "15m", **settings}
    return bt.backtest(Holds, **arguments)


def test_naive_timestamps_with_tz_give_the_run_its_dates():
    manifest = run(naive_ist_frame(), tz="Asia/Kolkata").manifest
    assert (manifest["arguments"]["start"], manifest["arguments"]["end"]) == (
        "2025-07-01",
        "2025-07-01",
    )


def test_naive_timestamps_in_a_column_with_tz_give_the_run_its_dates():
    frame = naive_ist_frame().reset_index()
    assert run(frame, tz="Asia/Kolkata").manifest["arguments"]["start"] == "2025-07-01"


def test_a_csv_file_with_tz_gives_the_run_its_dates(tmp_path):
    path = tmp_path / "SBIN.csv"
    naive_ist_frame().to_csv(path)
    arguments = run(str(path), tz="Asia/Kolkata").manifest["arguments"]
    assert (arguments["start"], arguments["end"]) == ("2025-07-01", "2025-07-01")


def test_a_files_reader_gives_the_run_its_dates(tmp_path):
    path = tmp_path / "SBIN.csv"
    naive_ist_frame().to_csv(path)
    arguments = run(Files(path, tz="Asia/Kolkata")).manifest["arguments"]
    assert (arguments["start"], arguments["end"]) == ("2025-07-01", "2025-07-01")


def test_naive_timestamps_without_tz_ask_for_a_time_zone_not_for_dates():
    with pytest.raises(ConfigError, match="carry no time zone") as caught:
        run(naive_ist_frame())
    assert "tz='Asia/Kolkata'" in str(caught.value)
    assert "start=" not in str(caught.value)


def epoch_columns() -> dict:
    stamps = [FIRST_OPEN_EPOCH_SECONDS + BAR_SECONDS * k for k in range(BARS)]
    prices = {name: [100.0] * BARS for name in ("open", "high", "low", "close")}
    return {"timestamp": stamps, **prices, "volume": [1_000.0] * BARS}


def first_open(**options) -> int:
    bars = Frame(epoch_columns(), unit="s", **options).bars("SBIN", "NSE", "15m", DAY, DAY)
    return int(bars["ts_open"].iloc[0])


def test_epoch_seconds_are_utc_without_a_tz():
    assert first_open() == FIRST_OPEN_NS


def test_epoch_seconds_accept_tz_utc():
    assert first_open(tz="UTC") == FIRST_OPEN_NS


def test_epoch_seconds_with_another_tz_are_a_config_error():
    with pytest.raises(ConfigError, match="1970-01-01 UTC"):
        first_open(tz="Asia/Kolkata")
