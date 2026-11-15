"""Broker candles as live bars, and the check of tick-built bars against them
(08-live-trading section 5, rule 2; D27): the candle's time read only from an explicit
field, its close cut at the session close, differences kept for the drift log, and the
candle accepted only when it covers the tick bar's slice.

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


def ist(clock: str, day: dt.date = dt.date(2026, 11, 16)) -> int:
    """UTC nanoseconds of an IST clock time, on 2026-11-16 unless `day` is given."""
    instant = dt.datetime.combine(day, dt.time.fromisoformat(clock), tzinfo=IST)
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


def flat_row(clock: str) -> dict:
    return server_row(clock, 800.0, 800.0, 800.0, 800.0, 0)


def history_frame(rows: list[dict]) -> pd.DataFrame:
    """The frame `client.history` builds from the payload rows of an intraday interval."""
    frame = pd.DataFrame(rows)
    stamps = pd.to_datetime(frame["timestamp"], unit="s")
    frame["timestamp"] = stamps.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    return frame.set_index("timestamp").sort_index()


def tick_bar(opens: str, ohlc: tuple[float, ...], volume: float, ticks: int) -> LiveBar:
    return LiveBar("SBIN", "NSE", ist(opens), ist(opens) + MINUTE_NS, *ohlc, volume, ticks, "ticks")


def candle(opens: str, ohlc: tuple[float, ...], volume: float, minutes: int = 1) -> LiveBar:
    closes = ist(opens) + minutes * MINUTE_NS
    return LiveBar("SBIN", "NSE", ist(opens), closes, *ohlc, volume, 0, "candle")


# ----- the candle's time ------------------------------------------------------


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


def test_a_row_of_the_backtest_bars_layout_reads_its_ts_open():
    frame = pd.DataFrame(
        {
            "ts_open": [ist("09:15"), ist("09:16")],
            "ts_close": [ist("09:16"), ist("09:17")],
            "open": [800.0, 801.0],
            "high": [801.0, 802.0],
            "low": [799.0, 800.0],
            "close": [800.5, 801.5],
            "volume": [100.0, 200.0],
        }
    )

    bar = candle_bar("SBIN", "NSE", frame.iloc[1], MINUTE_NS)

    assert (bar.ts_open_ns, bar.ts_close_ns) == (ist("09:16"), ist("09:17"))


@pytest.mark.parametrize("position", [0, 1])
def test_a_row_position_is_never_read_as_a_time(position):
    frame = pd.DataFrame([flat_row("09:15"), flat_row("09:16")]).drop(columns="timestamp")

    with pytest.raises(TimestampError, match="pass the timestamp"):
        candle_bar("SBIN", "NSE", frame.iloc[position], MINUTE_NS)


def test_a_naive_timestamp_raises_timestamp_error_as_its_zone_is_never_guessed():
    row = flat_row("09:15") | {"timestamp": dt.datetime(2026, 11, 16, 9, 15)}

    with pytest.raises(TimestampError, match="no time zone"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


@pytest.mark.parametrize(
    "stamp",
    [
        ist("09:15") // 10**6,
        ist("09:15") // 10**9 + 0.5,
        5,
        int(dt.datetime(1999, 12, 31, tzinfo=dt.UTC).timestamp()),
        int(dt.datetime(2100, 1, 1, tzinfo=dt.UTC).timestamp()),
        "2026-11-16 09:15",
        None,
        True,
    ],
    ids=["milliseconds", "fraction", "tiny", "1999", "2100", "string", "none", "bool"],
)
def test_a_timestamp_that_is_not_aware_or_whole_epoch_seconds_from_2000_to_2100_raises(stamp):
    row = flat_row("09:15") | {"timestamp": stamp}

    with pytest.raises(TimestampError, match="epoch seconds"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


def test_a_ts_open_in_seconds_rather_than_nanoseconds_raises():
    row = flat_row("09:15")
    row["ts_open"] = row.pop("timestamp")

    with pytest.raises(TimestampError, match="UTC nanoseconds"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


# ----- the candle's close -----------------------------------------------------


def test_the_candle_of_the_sessions_last_slice_closes_at_the_session_close():
    row = flat_row("15:15")

    bar = candle_bar("SBIN", "NSE", row, 60 * MINUTE_NS)

    assert (bar.ts_open_ns, bar.ts_close_ns) == (ist("15:15"), ist("15:30"))


@pytest.mark.parametrize("clock", ["15:30", "09:14"])
def test_a_candle_outside_the_session_raises_data_error(clock):
    with pytest.raises(DataError, match="outside the NSE session"):
        candle_bar("SBIN", "NSE", flat_row(clock), MINUTE_NS)


def test_a_candle_on_a_day_without_a_session_raises_data_error():
    sunday = ist("10:00", dt.date(2026, 11, 15)) // 10**9
    row = flat_row("10:00") | {"timestamp": sunday}

    with pytest.raises(DataError, match="outside the NSE session"):
        candle_bar("SBIN", "NSE", row, MINUTE_NS)


# ----- the candle's values ----------------------------------------------------


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


@pytest.mark.parametrize("interval_ns", [0, 1.5e9, 24 * 60 * MINUTE_NS])
def test_candle_bar_refuses_an_interval_that_is_not_whole_seconds_shorter_than_a_day(
    interval_ns,
):
    with pytest.raises(ConfigError, match="interval_ns"):
        candle_bar("SBIN", "NSE", flat_row("09:15"), interval_ns)


# ----- CandleCheck ------------------------------------------------------------


def test_compare_gives_the_candle_minus_the_tick_bar_for_every_field():
    built = tick_bar("09:15", (800.0, 801.5, 799.0, 801.0), 12000.0, 40)
    broker = candle("09:15", (800.05, 802.0, 798.5, 801.0), 12500.0)

    drift = CandleCheck().compare(built, broker)

    assert (drift.symbol, drift.exchange, drift.ticks) == ("SBIN", "NSE", 40)
    assert (drift.ts_open_ns, drift.ts_close_ns) == (ist("09:15"), ist("09:16"))
    assert (drift.candle_ts_open_ns, drift.candle_ts_close_ns) == (ist("09:15"), ist("09:16"))
    assert drift.open == pytest.approx(0.05)
    assert (drift.high, drift.low, drift.close, drift.volume) == (0.5, -0.5, 0.0, 500.0)
    assert drift.same_slice


def test_compare_keeps_every_field_when_the_bars_agree():
    built = tick_bar("09:15", (800.0, 801.0, 799.0, 800.0), 100.0, 5)

    drift = CandleCheck().compare(built, candle("09:15", (800.0, 801.0, 799.0, 800.0), 100.0))

    slice_ = (ist("09:15"), ist("09:16"))
    assert drift == BarDrift("SBIN", "NSE", *slice_, *slice_, 0.0, 0.0, 0.0, 0.0, 0.0, 5)


def test_compare_records_nothing():
    check = CandleCheck()

    check.compare(tick_bar("09:15", (800.0,) * 4, 0.0, 1), candle("09:15", (800.0,) * 4, 0.0))

    assert check.drifts == []


def test_accepted_returns_the_candle_of_the_same_slice_and_keeps_its_drift():
    check = CandleCheck()
    built = tick_bar("09:15", (800.0,) * 4, 10.0, 2)
    broker = candle("09:15", (800.0,) * 4, 10.0)

    assert check.accepted(built, broker) is broker
    assert check.drifts == [check.compare(built, broker)]


def test_accepted_returns_the_tick_bar_and_keeps_nothing_for_a_candle_of_another_minute():
    check = CandleCheck()
    built = tick_bar("09:16", (800.0,) * 4, 10.0, 2)

    assert check.accepted(built, candle("09:15", (800.0,) * 4, 10.0)) is built
    assert check.drifts == []


def test_a_candle_of_another_interval_is_never_accepted_for_the_bar():
    check = CandleCheck()
    built = tick_bar("09:15", (800.0,) * 4, 10.0, 2)
    five_minutes = candle("09:15", (790.0, 830.0, 780.0, 820.0), 1e6, minutes=5)

    assert check.accepted(built, five_minutes) is built
    assert not check.compare(built, five_minutes).same_slice
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
