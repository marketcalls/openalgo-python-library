"""The public `bt.read_ticks` (`_read_ticks.py`, `_live/_ticks_frame.py`; the lead's decision
of 2026-11-18, unit 16): a day's recordings read back as one table, merged in the order
the ticks were received.

Recordings are made through `bt.record`'s test seam (`_record_verb.record_day`) on the
recorder tests' fakes (`recorder_fakes`): manual clocks and a feed the test plays.
Expected values come from what the test played and from the recorder's own verified
reader, never from the table's code. Nothing opens a socket.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from recorder_fakes import DAY, INFY, SBIN, Clocks, FakeFeed, Script, ist_ns, quote

import openalgo.backtest as bt
from openalgo.backtest._live import _record_verb
from openalgo.backtest._live.recorder import read_merged_ticks
from openalgo.backtest.errors import DataError

#: The table's columns for a recording in quote mode (it keeps the volume).
QUOTE_COLUMNS = [
    "exchange",
    "symbol",
    "exchange_time",
    "receive_time",
    "exchange_time_given",
    "ltp",
    "volume",
]
#: A last trade time the feed gives, epoch milliseconds: 10:00:00.100 IST.
LTT_MS = ist_ns("10:00:00.100") // 1_000_000


def record(tmp_path, start: str, until: str, plays: list[tuple[str, dict]], instruments) -> None:
    """One recording from the IST time `start` to `until` on the fakes, the feed playing
    each (IST time, quote message) of `plays`."""
    clocks = Clocks(ist_ns(start))
    feed, script = FakeFeed(), Script(clocks)
    for at, message in plays:
        script.at(ist_ns(at), lambda message=message: feed.push(message))
    _record_verb.record_day(
        None,
        instruments,
        cache_dir=tmp_path,
        until=until,
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        feed=feed,
    )


def ist(ns: list[int]) -> list[pd.Timestamp]:
    return [pd.Timestamp(value, tz="UTC").tz_convert("Asia/Kolkata") for value in ns]


@pytest.fixture
def one_day(tmp_path):
    """SBIN recorded from 10:00 to 10:01: two quotes, the first with its last trade time."""
    record(
        tmp_path,
        "10:00:00",
        "10:01",
        [
            ("10:00:00.200", quote("SBIN", 812.4, volume=1000, ltt=LTT_MS)),
            ("10:00:00.500", quote("SBIN", 812.5, volume=1010)),
        ],
        [SBIN],
    )
    return tmp_path


def test_a_recording_reads_back_whole_one_row_per_tick(one_day):
    frame = bt.read_ticks(DAY, cache_dir=one_day)
    assert list(frame.columns) == QUOTE_COLUMNS
    assert frame[["exchange", "symbol", "ltp", "volume"]].values.tolist() == [
        ["NSE", "SBIN", 812.4, 1000.0],
        ["NSE", "SBIN", 812.5, 1010.0],
    ]


def test_the_receive_times_are_the_recordings_own_in_ist(one_day):
    frame = bt.read_ticks(DAY, cache_dir=one_day)
    recorded = read_merged_ticks([SBIN], DAY, cache_dir=one_day).ticks
    assert list(frame["receive_time"]) == ist([tick.ts_recv_ns for tick in recorded])
    assert frame["receive_time"][0].utcoffset() == dt.timedelta(hours=5, minutes=30)


def test_the_exchange_time_is_the_feeds_or_the_receive_time_and_a_column_says_which(one_day):
    frame = bt.read_ticks(DAY, cache_dir=one_day)
    assert list(frame["exchange_time_given"]) == [True, False]
    assert frame["exchange_time"][0] == ist([LTT_MS * 1_000_000])[0]
    assert frame["exchange_time"][1] == frame["receive_time"][1]


def test_two_recordings_of_one_day_are_merged_in_the_order_received(tmp_path):
    record(
        tmp_path,
        "10:00:00",
        "10:01",
        [("10:00:00.200", quote("SBIN", 812.4)), ("10:00:00.300", quote("INFY", 1500.0))],
        [SBIN, INFY],
    )
    record(
        tmp_path,
        "10:05:00",
        "10:06",
        [("10:05:00.200", quote("INFY", 1501.0)), ("10:05:00.300", quote("SBIN", 812.6))],
        [SBIN, INFY],
    )
    frame = bt.read_ticks(DAY, cache_dir=tmp_path)
    assert frame[["symbol", "ltp"]].values.tolist() == [
        ["SBIN", 812.4],
        ["INFY", 1500.0],
        ["INFY", 1501.0],
        ["SBIN", 812.6],
    ]


def test_instruments_read_only_those_named(tmp_path):
    record(
        tmp_path,
        "10:00:00",
        "10:01",
        [("10:00:00.200", quote("SBIN", 812.4)), ("10:00:00.300", quote("INFY", 1500.0))],
        [SBIN, INFY],
    )
    frame = bt.read_ticks(DAY, [{"exchange": "NSE", "symbol": "INFY"}], cache_dir=tmp_path)
    assert frame[["symbol", "ltp"]].values.tolist() == [["INFY", 1500.0]]


def test_a_day_with_no_recording_raises_naming_the_folder(tmp_path):
    with pytest.raises(DataError) as missing:
        bt.read_ticks(DAY, cache_dir=tmp_path)
    assert "2026-11-16" in str(missing.value)
    assert str(tmp_path) in str(missing.value)
