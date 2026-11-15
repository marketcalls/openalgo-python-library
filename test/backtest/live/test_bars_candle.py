"""Broker candles as live bars, and the check of tick-built bars against them
(08-live-trading section 5, rule 2; D27): differences kept for the drift log, and the
candle accepted only when it starts at the expected time.

`history_frame` shapes rows as `client.history` returns them (openalgo/data.py): the
server's rows carry the candle's open in epoch seconds, and the SDK turns them into a
frame indexed by IST-aware timestamps.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from openalgo.backtest._live._types import LiveBar
from openalgo.backtest._live.bars import BarDrift, CandleCheck, candle_bar
from openalgo.backtest.errors import ConfigError, DataError, TimestampError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
MINUTE_NS = 60 * 10**9


def ist(clock: str) -> int:
    """UTC nanoseconds of an IST clock time on 2026-11-16."""
    instant = dt.datetime.combine(dt.date(2026, 11, 16), dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


def server_row(clock: str, open_, high, low, close, volume) -> dict:
    """A row of the server's history payload: the candle's open in epoch seconds."""
    return {
        "timestamp": ist(clock) // 10**9,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def history_frame(rows: list[dict]) -> pd.DataFrame:
    """The frame `client.history` builds from the payload rows of an intraday interval."""
    frame = pd.DataFrame(rows)
    stamps = pd.to_datetime(frame["timestamp"], unit="s")
    frame["timestamp"] = stamps.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    return frame.set_index("timestamp").sort_index()


def tick_bar(opens: str, ohlc: tuple[float, ...], volume: float, ticks: int) -> LiveBar:
    return LiveBar("SBIN", "NSE", ist(opens), ist(opens) + MINUTE_NS, *ohlc, volume, ticks, "ticks")


def candle(opens: str, ohlc: tuple[float, ...], volume: float) -> LiveBar:
    return LiveBar("SBIN", "NSE", ist(opens), ist(opens) + MINUTE_NS, *ohlc, volume, 0, "candle")


# ----- candle_bar -------------------------------------------------------------


def test_a_row_of_the_history_frame_becomes_a_candle_bar_stamped_by_its_open_and_close():
    frame = history_frame([server_row("09:15", 800.0, 802.0, 799.0, 801.0, 15000)])

    bar = candle_bar("SBIN", "NSE", frame.iloc[0], MINUTE_NS)

    assert bar == candle("09:15", (800.0, 802.0, 799.0, 801.0), 15000.0)


def test_a_server_payload_row_in_epoch_seconds_becomes_the_same_candle_bar():
    row = server_row("09:15", 800.0, 802.0, 799.0, 801.0, 15000)

    assert candle_bar("SBIN", "NSE", row, MINUTE_NS) == candle(
        "09:15", (800.0, 802.0, 799.0, 801.0), 15000.0
    )


def test_a_row_of_the_frame_with_its_index_reset_reads_the_timestamp_column():
    frame = history_frame([server_row("09:16", 801.0, 801.0, 800.0, 800.5, 900)]).reset_index()

    assert candle_bar("SBIN", "NSE", frame.iloc[0], MINUTE_NS).ts_open_ns == ist("09:16")


def test_a_payload_row_in_a_frame_of_floats_keeps_its_whole_epoch_seconds():
    row = pd.DataFrame([server_row("09:17", 800.0, 801.0, 799.5, 800.0, 10.5)]).astype(float)

    assert candle_bar("SBIN", "NSE", row.iloc[0], MINUTE_NS).ts_open_ns == ist("09:17")


def test_a_naive_timestamp_raises_timestamp_error_as_its_zone_is_never_guessed():
    row = {"timestamp": dt.datetime(2026, 11, 16, 9, 15), "open": 1.0, "high": 1.0}
    row |= {"low": 1.0, "close": 1.0, "volume": 0}

    with pytest.raises(TimestampError, match="no time zone"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


@pytest.mark.parametrize("stamp", [1794802500.5, "2026-11-16 09:15", None, -60, True])
def test_a_timestamp_that_is_neither_aware_nor_whole_epoch_seconds_raises(stamp):
    row = server_row("09:15", 800.0, 800.0, 800.0, 800.0, 0) | {"timestamp": stamp}

    with pytest.raises(TimestampError, match="epoch seconds"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


@pytest.mark.parametrize("value", [math.nan, 0.0, -800.0, "800", None, math.inf])
def test_an_unusable_price_raises_data_error(value):
    row = server_row("09:15", 800.0, 802.0, 799.0, 801.0, 0) | {"close": value}

    with pytest.raises(DataError, match="close must be a positive price"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


def test_a_missing_price_raises_data_error_naming_the_field():
    row = server_row("09:15", 800.0, 802.0, 799.0, 801.0, 0)
    del row["high"]

    with pytest.raises(DataError, match="no 'high' field"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


def test_a_high_and_low_that_do_not_contain_the_open_and_close_raise_data_error():
    row = server_row("09:15", 800.0, 800.5, 799.0, 801.0, 0)

    with pytest.raises(DataError, match="not a valid bar"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


@pytest.mark.parametrize("volume", ["absent", None, math.nan])
def test_a_candle_without_volume_has_zero_volume(volume):
    row = server_row("09:15", 800.0, 800.0, 800.0, 800.0, volume)
    if volume == "absent":
        del row["volume"]

    assert candle_bar("NIFTY", "NSE_INDEX", row, MINUTE_NS).volume == 0.0


def test_a_negative_volume_raises_data_error():
    row = server_row("09:15", 800.0, 800.0, 800.0, 800.0, -1)

    with pytest.raises(DataError, match="volume"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


def test_candle_bar_refuses_an_interval_that_is_not_whole_seconds():
    row = server_row("09:15", 800.0, 800.0, 800.0, 800.0, 0)

    with pytest.raises(ConfigError, match="interval_ns"):
        candle_bar("SBIN", "NSE", row, 0)


# ----- CandleCheck ------------------------------------------------------------


def test_compare_gives_the_candle_minus_the_tick_bar_for_every_field():
    built = tick_bar("09:15", (800.0, 801.5, 799.0, 801.0), 12000.0, 40)
    broker = candle("09:15", (800.05, 802.0, 798.5, 801.0), 12500.0)

    drift = CandleCheck().compare(built, broker)

    assert (drift.symbol, drift.exchange, drift.ticks) == ("SBIN", "NSE", 40)
    assert (drift.ts_open_ns, drift.candle_ts_open_ns) == (ist("09:15"), ist("09:15"))
    assert drift.open == pytest.approx(0.05)
    assert (drift.high, drift.low, drift.close, drift.volume) == (0.5, -0.5, 0.0, 500.0)
    assert drift.on_time


def test_compare_keeps_every_field_when_the_bars_agree():
    built = tick_bar("09:15", (800.0, 801.0, 799.0, 800.0), 100.0, 5)

    drift = CandleCheck().compare(built, candle("09:15", (800.0, 801.0, 799.0, 800.0), 100.0))

    assert drift == BarDrift("SBIN", "NSE", ist("09:15"), ist("09:15"), 0.0, 0.0, 0.0, 0.0, 0.0, 5)


def test_compare_records_nothing():
    check = CandleCheck()
    built = tick_bar("09:15", (800.0,) * 4, 0.0, 1)

    check.compare(built, candle("09:15", (800.0,) * 4, 0.0))

    assert check.drifts == []


def test_accepted_returns_the_candle_that_starts_on_time_and_keeps_its_drift():
    check = CandleCheck()
    built = tick_bar("09:15", (800.0,) * 4, 10.0, 2)
    broker = candle("09:15", (800.0,) * 4, 10.0)

    assert check.accepted(built, broker) is broker
    assert check.drifts == [check.compare(built, broker)]


def test_accepted_returns_the_tick_bar_and_keeps_nothing_when_the_candle_is_of_another_minute():
    check = CandleCheck()
    built = tick_bar("09:16", (800.0,) * 4, 10.0, 2)

    assert check.accepted(built, candle("09:15", (800.0,) * 4, 10.0)) is built
    assert check.drifts == []


def test_the_drifts_of_a_session_are_kept_in_the_order_the_candles_were_accepted():
    check = CandleCheck()
    for clock in ("09:15", "09:16", "09:17"):
        check.accepted(tick_bar(clock, (800.0,) * 4, 0.0, 1), candle(clock, (800.0,) * 4, 0.0))

    assert [d.ts_open_ns for d in check.drifts] == [ist("09:15"), ist("09:16"), ist("09:17")]


def test_a_candle_of_another_symbol_cannot_check_a_bar():
    built = tick_bar("09:15", (800.0,) * 4, 0.0, 1)
    other = LiveBar("INFY", "NSE", ist("09:15"), ist("09:16"), *(1500.0,) * 4, 0.0, 0, "candle")

    with pytest.raises(ConfigError, match="cannot check"):
        CandleCheck().accepted(built, other)


def test_compare_refuses_the_bars_in_the_wrong_order():
    built = tick_bar("09:15", (800.0,) * 4, 0.0, 1)

    with pytest.raises(ConfigError, match="from ticks and then a broker candle"):
        CandleCheck().compare(candle("09:15", (800.0,) * 4, 0.0), built)


def test_a_bar_drift_is_frozen():
    drift = CandleCheck().compare(
        tick_bar("09:15", (800.0,) * 4, 0.0, 1), candle("09:15", (800.0,) * 4, 0.0)
    )

    with pytest.raises(FrozenInstanceError):
        drift.close = 1.0  # type: ignore[misc]
