"""Timestamp normalisation into the canonical layout (FR-DATA-05, FR-DATA-06).

Includes the Phase 0 exit criterion: the same 1-minute series served in the
Historify shape and in the broker shape normalises to identical timestamps and
passes calendar validation, for NSE and for MCX, against the engine's own calendar.

Expected values are written out from session times (NSE 09:15 to 15:30 IST), never
computed by the code under test.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import (
    NS,
    FakeCalendar,
    FakeClient,
    at_ist,
    broker_shape,
    daily_rows,
    days_between,
    historify_shape,
    ist_frame,
    normalise,
    open_ns,
    real_calendar,
    server_load,
    shifted,
    truth_rows,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from openalgo.backtest._calendar_types import CalendarWarning
from openalgo.backtest._layout import BAR_COLUMNS
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data import load_bars
from openalgo.backtest.data.providers import validate_layout
from openalgo.backtest.errors import ConfigError, DataError, TimestampError

MON = dt.date(2024, 1, 1)  # a Monday
FRI = dt.date(2024, 1, 5)
WEEK = days_between(MON, FRI)
LONG_AFTER = at_ist(dt.date(2024, 3, 15), 18, 0)  # "now" for loads of finished history
IST_SHIFT_SECONDS = 19_800
MINUTE = 60
NSE_MINUTES_PER_DAY = 375  # 09:15 to 15:30


def _epoch_day(day):
    """Epoch seconds of 00:00 UTC on `day`: the OpenAlgo stamp of a daily row."""
    return (day - dt.date(1970, 1, 1)).days * 86_400


def _bar(stamp):
    return {"timestamp": stamp, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 5}


def _own(rows, interval="1m", **options):
    return load_bars(
        "X", "NSE", interval, MON, FRI, data=ist_frame(rows), calendar=FakeCalendar(),
        now=LONG_AFTER, **options,
    )  # fmt: skip


# --- the two raw conventions of OpenAlgo --------------------------------------------


@pytest.mark.parametrize(
    "calendar, exchange, preopen_minutes",
    [
        (default_calendar(), "NSE", 8),
        (default_calendar(), "MCX", 0),
        (real_calendar(), "NSE", 8),
        (FakeCalendar(), "MCX", 0),
    ],
    ids=["NSE built-in calendar", "MCX built-in calendar", "NSE regular hours", "MCX protocol"],
)
def test_historify_and_broker_shapes_normalise_to_identical_timestamps(
    calendar, exchange, preopen_minutes
):
    truth = truth_rows(calendar, exchange, WEEK)
    client = FakeClient(
        db={("X", exchange, "1m"): historify_shape(truth, calendar, exchange, preopen_minutes)},
        api={("X", exchange, "1m"): broker_shape(truth)},
    )
    loads = {
        source: server_load(
            client,
            "1m",
            MON,
            FRI,
            calendar=calendar,
            now_ns=LONG_AFTER,
            exchange=exchange,
            source=source,
            cache=False,
        ).bars  # fmt: skip
        for source in ("db", "api")
    }
    assert loads["db"]["ts_open"].tolist() == open_ns(truth)
    assert loads["api"]["ts_open"].tolist() == open_ns(truth)
    assert loads["db"]["ts_close"].tolist() == loads["api"]["ts_close"].tolist()
    # calendar validation: every day starts at the session open, minute by minute
    for day in WEEK:
        session_open = calendar.session(exchange, day)[0]
        first_of_day = loads["db"][loads["db"]["ts_open"] >= session_open].iloc[0]
        assert first_of_day["ts_open"] == session_open
        assert first_of_day["ts_close"] == session_open + MINUTE * NS


def test_historify_preopen_bars_are_dropped_and_counted():
    calendar = FakeCalendar()
    rows = historify_shape(truth_rows(calendar, "NSE", WEEK), calendar, "NSE", preopen_minutes=8)
    bars, report = normalise(rows, calendar=calendar)
    assert len(bars) == NSE_MINUTES_PER_DAY * 5
    assert report.preopen_dropped == 8 * 5


def test_result_is_in_the_canonical_layout():
    bars, _ = normalise(truth_rows(FakeCalendar(), "NSE", [MON]))
    validate_layout(bars)
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert [str(bars[name].dtype) for name in BAR_COLUMNS] == ["int64"] * 2 + ["float64"] * 6
    assert bars.attrs == {"symbol": "X", "exchange": "NSE", "interval": "1m"}


# --- intraday: bar open, bar close, session filter ----------------------------------


def test_intraday_stamp_is_the_bar_open_and_the_bar_closes_one_interval_later():
    bars, _ = normalise(truth_rows(FakeCalendar(), "NSE", [MON], step=300), interval="5m")
    assert bars["ts_open"].iloc[0] == at_ist(MON, 9, 15)
    assert bars["ts_close"].iloc[0] == at_ist(MON, 9, 20)
    assert bars["ts_open"].iloc[-1] == at_ist(MON, 15, 25)
    assert bars["ts_close"].iloc[-1] == at_ist(MON, 15, 30)


def test_last_bar_of_the_session_closes_at_the_session_close():
    hourly = truth_rows(FakeCalendar(), "NSE", [MON], step=3600)  # 09:15, 10:15, ... 15:15
    bars, _ = normalise(hourly, interval="1h")
    assert bars["ts_open"].iloc[-1] == at_ist(MON, 15, 15)
    assert bars["ts_close"].iloc[-1] == at_ist(MON, 15, 30)  # not 16:15


def test_label_close_reads_the_stamp_as_the_bar_close():
    close_stamped = shifted(truth_rows(FakeCalendar(), "NSE", [MON]), MINUTE)
    bars, _ = normalise(close_stamped, label="close")
    assert bars["ts_open"].iloc[0] == at_ist(MON, 9, 15)
    assert bars["ts_close"].iloc[0] == at_ist(MON, 9, 16)
    assert len(bars) == NSE_MINUTES_PER_DAY


def test_close_labelled_last_bar_of_a_session_opens_at_the_last_grid_point():
    session_open, session_close = FakeCalendar().session("NSE", MON)
    half_hour = 1800 * NS
    closes = [*range(session_open + half_hour, session_close, half_hour), session_close]
    values = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}
    frame = pd.DataFrame(values, index=pd.to_datetime(closes, unit="ns", utc=True))
    bars = load_bars(
        "X", "NSE", "30m", MON, MON, data=frame, label="close", calendar=FakeCalendar(),
        now=LONG_AFTER,
    ).bars  # fmt: skip
    assert bars["ts_open"].iloc[-2:].tolist() == [at_ist(MON, 14, 45), at_ist(MON, 15, 15)]
    assert bars["ts_close"].iloc[-1] == at_ist(MON, 15, 30)


def test_unknown_label_is_a_config_error():
    with pytest.raises(ConfigError, match="label"):
        normalise(truth_rows(FakeCalendar(), "NSE", [MON]), label="middle")


@pytest.mark.parametrize(
    "hour, minute", [(15, 40), (20, 15)], ids=["closing session", "long after the close"]
)
def test_postclose_bars_are_dropped_and_counted(hour, minute):
    rows = truth_rows(FakeCalendar(), "NSE", WEEK)
    late_bar = dict(rows[0], timestamp=at_ist(MON, hour, minute) // NS)
    bars, report = normalise(rows + [late_bar])
    assert len(bars) == NSE_MINUTES_PER_DAY * 5
    assert report.postclose_dropped == 1


def test_out_of_session_bars_can_be_kept_on_request():
    calendar = FakeCalendar()
    rows = historify_shape(truth_rows(calendar, "NSE", [MON]), calendar, "NSE", preopen_minutes=8)
    bars, report = normalise(rows, calendar=calendar, regular_session_only=False)
    assert len(bars) == NSE_MINUTES_PER_DAY + 8
    assert report.preopen_dropped == 0
    assert report.notes == [
        "8 bars outside the regular session were kept (regular_session_only=False)"
    ]


def test_bars_on_a_closed_day_are_dropped_and_counted():
    rows = truth_rows(FakeCalendar(), "NSE", WEEK)
    saturday = [dict(row, timestamp=row["timestamp"] + 5 * 86_400) for row in rows[:10]]
    bars, report = normalise(rows + saturday)
    assert len(bars) == NSE_MINUTES_PER_DAY * 5
    assert report.closed_day_dropped == 10
    assert report.closed_day_examples == ["2024-01-06"]


def test_late_first_bar_on_one_day_is_counted_not_raised():
    rows = truth_rows(FakeCalendar(), "NSE", WEEK)
    before_0947 = at_ist(MON, 9, 47) // NS
    thin_monday = [
        row for row in rows if not at_ist(MON, 9, 15) // NS <= row["timestamp"] < before_0947
    ]
    bars, report = normalise(thin_monday)
    assert len(bars) == NSE_MINUTES_PER_DAY * 5 - 32
    assert (report.late_start_days, report.session_days) == (1, 5)
    assert report.late_start_examples == ["2024-01-01"]


def test_thin_instrument_without_an_opening_bar_on_most_days_loads_with_a_count():
    days = [MON, dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    rows = truth_rows(FakeCalendar(), "NSE", days)
    opens = {at_ist(MON, 9, 15) // NS, at_ist(dt.date(2024, 1, 3), 9, 15) // NS}
    result = _own([row for row in rows if row["timestamp"] not in opens])
    assert len(result.bars) == 3 * NSE_MINUTES_PER_DAY - 2
    assert result.meta.quality.count("late_session_start") == 2


# --- intraday: a wrong interval or a shifted clock is an error, never repaired ------


def test_bars_finer_than_the_declared_interval_are_an_error():
    with pytest.raises(TimestampError, match="^NSE:X 5m: 300 bars are not on the NSE bar lattice"):
        normalise(truth_rows(FakeCalendar(), "NSE", [MON]), interval="5m")


@pytest.mark.parametrize("true_step, declared", [(300, "1m"), (900, "5m"), (3600, "15m")])
def test_bars_coarser_than_the_declared_interval_are_an_error(true_step, declared):
    coarse = truth_rows(FakeCalendar(), "NSE", [MON, dt.date(2024, 1, 2)], step=true_step)
    with pytest.raises(TimestampError, match="coarser than the declared interval"):
        normalise(coarse, interval=declared)


def test_a_few_one_minute_bars_that_happen_to_be_five_minutes_apart_are_not_coarser_data():
    rows = truth_rows(FakeCalendar(), "NSE", [MON])
    sparse = [rows[0], rows[5], rows[25]]  # 09:15, 09:20, 09:40
    bars, _ = normalise(sparse)
    assert bars["ts_open"].tolist() == open_ns(sparse)


@pytest.mark.parametrize(
    "exchange, shift_seconds, direction",
    [
        ("NSE", IST_SHIFT_SECONDS, "late"),
        ("MCX", IST_SHIFT_SECONDS, "late"),
        ("NSE", -IST_SHIFT_SECONDS, "early"),
        ("MCX", -IST_SHIFT_SECONDS, "early"),
    ],
)
def test_offset_of_five_thirty_is_an_error_that_names_the_cause(exchange, shift_seconds, direction):
    calendar = FakeCalendar()
    off = shifted(truth_rows(calendar, exchange, WEEK), shift_seconds)
    with pytest.raises(TimestampError) as caught:
        normalise(off, exchange=exchange, calendar=calendar)
    message = str(caught.value)
    assert f"5 hours 30 minutes {direction}" in message
    assert "never shifted silently" in message


def test_bars_stamped_at_their_close_but_read_as_open_are_an_error():
    close_stamped = shifted(truth_rows(FakeCalendar(), "NSE", WEEK), MINUTE)
    with pytest.raises(TimestampError, match="^NSE:X 1m: the bars look labelled by their close"):
        normalise(close_stamped)


def test_close_stamped_bars_whose_last_bar_is_short_are_an_error_too():
    # NSE 30-minute bars close at 09:45, 10:15, ..., 15:15 and, for the short last bar,
    # at 15:30. Read as opens, the first bar of each day is 09:45 and the last 15:30.
    days = [MON, dt.date(2024, 1, 2)]
    rows = truth_rows(FakeCalendar(), "NSE", days, step=1800)
    close_stamped = [
        dict(row, timestamp=min(row["timestamp"] + 1800, _close_of(row["timestamp"])))
        for row in rows
    ]
    with pytest.raises(TimestampError, match="the bars look labelled by their close time"):
        normalise(close_stamped, interval="30m")


def _close_of(stamp):
    """Epoch seconds of the NSE close (15:30 IST) on the IST date of `stamp`."""
    ist_date = dt.date(1970, 1, 1) + dt.timedelta(seconds=stamp + IST_SHIFT_SECONDS)
    return at_ist(ist_date, 15, 30) // NS


def test_open_stamped_series_without_any_opening_bar_is_not_taken_for_close_stamped():
    rows = truth_rows(FakeCalendar(), "NSE", WEEK)
    openings = {at_ist(day, 9, 15) // NS for day in WEEK}
    thin = [row for row in rows if row["timestamp"] not in openings]
    bars, report = normalise(thin)
    assert len(bars) == 5 * (NSE_MINUTES_PER_DAY - 1)
    assert report.late_start_days == 5


def test_bars_stamped_at_their_open_but_declared_close_are_an_error():
    with pytest.raises(TimestampError, match="look stamped at their open"):
        normalise(truth_rows(FakeCalendar(), "NSE", WEEK), label="close")


def test_offset_error_from_a_server_names_the_broker_and_the_remedy():
    calendar = FakeCalendar()
    late = shifted(truth_rows(calendar, "NSE", WEEK), IST_SHIFT_SECONDS)
    client = FakeClient(broker="compositedge", api={("X", "NSE", "1m"): broker_shape(late)})
    with pytest.raises(TimestampError) as caught:
        server_load(
            client, "1m", MON, FRI, calendar=calendar, now_ns=LONG_AFTER, source="api", cache=False
        )
    message = str(caught.value)
    assert "broker compositedge" in message
    assert "XTS family" in message
    assert "intraday_shift_seconds=19800" in message


def test_offset_error_from_own_data_names_the_time_zone_to_declare():
    frame = ist_frame(truth_rows(FakeCalendar(), "NSE", WEEK))
    frame.index = frame.index.tz_localize(None)  # IST clock times without a zone
    with pytest.raises(TimestampError, match="declare tz='Asia/Kolkata'"):
        load_bars(
            "X",
            "NSE",
            "1m",
            MON,
            FRI,
            data=frame,
            tz="UTC",
            calendar=FakeCalendar(),
            now=LONG_AFTER,
        )


def test_declared_broker_shift_is_removed_and_reported():
    calendar = FakeCalendar()
    truth = truth_rows(calendar, "NSE", WEEK)
    delivered_late = broker_shape(shifted(truth, IST_SHIFT_SECONDS))
    client = FakeClient(broker="compositedge", api={("X", "NSE", "1m"): delivered_late})
    result = server_load(
        client, "1m", MON, FRI, calendar=calendar, now_ns=LONG_AFTER, source="api", cache=False,
        provider_options={"intraday_shift_seconds": IST_SHIFT_SECONDS},
    )  # fmt: skip
    assert result.bars["ts_open"].tolist() == open_ns(truth)
    assert result.meta.normalisation.shift_applied_seconds == IST_SHIFT_SECONDS


def test_wrongly_declared_shift_fails_the_calendar_check():
    calendar = FakeCalendar()
    client = FakeClient(api={("X", "NSE", "1m"): broker_shape(truth_rows(calendar, "NSE", WEEK))})
    with pytest.raises(TimestampError, match="5 hours 30 minutes early"):
        server_load(
            client, "1m", MON, FRI, calendar=calendar, now_ns=LONG_AFTER, source="api", cache=False,
            provider_options={"intraday_shift_seconds": IST_SHIFT_SECONDS},
        )  # fmt: skip


def test_inconclusive_offset_check_is_passed_on_and_kept_in_the_warnings():
    # 200 one-minute bars from 15:00: 30 inside the session, 170 after its close. Moved
    # 5 hours 30 minutes earlier all would fit, but not decisively: the calendar warns.
    rows = [_bar(at_ist(MON, 15, 0) // NS + minute * MINUTE) for minute in range(200)]
    with pytest.warns(CalendarWarning, match="inconclusive"):
        result = _own(rows)
    assert len(result.bars) == 30  # 15:00 to 15:29
    assert result.meta.normalisation.postclose_dropped == 170
    assert any("offset check was inconclusive" in note for note in result.meta.warnings)


# --- daily and longer ---------------------------------------------------------------


def test_daily_row_maps_to_the_session_of_its_ist_date():
    bars, report = normalise([_bar(_epoch_day(MON))], interval="D")
    assert bars["ts_open"].tolist() == [at_ist(MON, 9, 15)]
    assert bars["ts_close"].tolist() == [at_ist(MON, 15, 30)]
    assert report.notes == []


def test_daily_row_stamped_at_ist_midnight_maps_to_the_same_date():
    ist_midnight = _epoch_day(MON) - IST_SHIFT_SECONDS  # 18:30 UTC the day before
    bars, _ = normalise([_bar(ist_midnight)], interval="D")
    assert bars["ts_open"].tolist() == [at_ist(MON, 9, 15)]


def test_daily_rows_with_mixed_stamp_conventions_are_an_error():
    rows = daily_rows(FakeCalendar(), "NSE", WEEK)
    rows[2] = dict(rows[2], timestamp=rows[2]["timestamp"] - IST_SHIFT_SECONDS)
    with pytest.raises(TimestampError, match="mix timestamp conventions"):
        normalise(rows, interval="D")


def test_daily_rows_dated_one_day_late_are_an_error():
    january = daily_rows(FakeCalendar(), "NSE", days_between(MON, dt.date(2024, 1, 31)))
    with pytest.raises(TimestampError, match="one day late"):
        normalise(shifted(january, 86_400), interval="D")


def test_daily_row_on_a_closed_day_is_dropped_and_counted():
    saturday = dt.date(2024, 1, 6)
    rows = daily_rows(FakeCalendar(), "NSE", WEEK) + [_bar(_epoch_day(saturday))]
    bars, report = normalise(rows, interval="D")
    assert len(bars) == 5
    assert report.closed_day_dropped == 1
    assert report.closed_day_examples == ["2024-01-06"]


def test_intraday_rows_declared_daily_are_an_error():
    with pytest.raises(TimestampError, match="mix timestamp conventions"):
        normalise(truth_rows(FakeCalendar(), "NSE", [MON]), interval="D")


@pytest.mark.parametrize(
    "interval, stamp_day, first_session, last_session",
    [
        # a week stamped at its Monday, here a holiday: the bar opens on Tuesday
        ("W", dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 5)),
        # the Friday of this week is a holiday: the bar closes on Thursday
        ("W", dt.date(2024, 1, 22), dt.date(2024, 1, 22), dt.date(2024, 1, 25)),
        # a week stamped at its last day, as pandas does
        ("W", dt.date(2024, 1, 14), dt.date(2024, 1, 8), dt.date(2024, 1, 12)),
        ("M", dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 31)),
        ("M", dt.date(2024, 2, 29), dt.date(2024, 2, 1), dt.date(2024, 2, 29)),
        ("Q", dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 3, 29)),
        ("Y", dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 12, 31)),
    ],
)
def test_own_weekly_and_longer_bars_span_the_first_and_last_session_of_their_period(
    interval, stamp_day, first_session, last_session
):
    calendar = FakeCalendar(holidays=[dt.date(2024, 1, 1), dt.date(2024, 1, 26)])
    bars, _ = normalise([_bar(_epoch_day(stamp_day))], interval=interval, calendar=calendar)
    assert bars["ts_open"].tolist() == [at_ist(first_session, 9, 15)]
    assert bars["ts_close"].tolist() == [at_ist(last_session, 15, 30)]


def test_several_rows_in_one_period_are_an_error():
    with pytest.raises(TimestampError, match="finer than the declared interval"):
        normalise(daily_rows(FakeCalendar(), "NSE", WEEK), interval="W")


# --- the requested dates and the partial candle -------------------------------------


def test_rows_outside_the_requested_dates_are_left_out_before_anything_is_counted():
    calendar = FakeCalendar()
    rows = historify_shape(truth_rows(calendar, "NSE", WEEK), calendar, "NSE", preopen_minutes=8)
    tuesday = dt.date(2024, 1, 2)
    bars, report = normalise(rows, calendar=calendar, start=tuesday, end=tuesday)
    assert bars["ts_open"].iloc[0] == at_ist(tuesday, 9, 15)
    assert (len(bars), report.rows_in, report.preopen_dropped) == (375, 383, 8)


@pytest.mark.parametrize(
    "now_hm, expected_rows, dropped",
    [((11, 0), 4, 1), ((15, 29), 4, 1), ((15, 30), 5, 0), ((18, 0), 5, 0)],
    ids=["mid session", "one minute before close", "at the close", "evening"],
)
def test_daily_candle_of_today_is_dropped_until_the_session_has_closed(
    now_hm, expected_rows, dropped
):
    rows = daily_rows(FakeCalendar(), "NSE", WEEK)  # Friday is "today"
    bars, report = normalise(rows, interval="D", now_ns=at_ist(FRI, *now_hm))
    assert len(bars) == expected_rows
    assert report.partial_dropped == dropped


def test_intraday_bar_that_has_not_closed_is_dropped():
    rows = truth_rows(FakeCalendar(), "NSE", [MON])
    thirty_seconds_into_the_1000_bar = at_ist(MON, 10, 0, 30)
    bars, report = normalise(rows, now_ns=thirty_seconds_into_the_1000_bar)
    assert bars["ts_close"].iloc[-1] == at_ist(MON, 10, 0)
    assert len(bars) == 45
    assert report.partial_dropped == NSE_MINUTES_PER_DAY - 45


def test_own_period_bar_of_a_period_that_is_still_running_is_a_partial_bar():
    monthly = [_bar(_epoch_day(dt.date(2024, 1, 1))), _bar(_epoch_day(dt.date(2024, 2, 1)))]
    bars, report = normalise(monthly, interval="M", now_ns=at_ist(dt.date(2024, 2, 15), 12))
    assert bars["ts_close"].tolist() == [at_ist(dt.date(2024, 1, 31), 15, 30)]
    assert report.partial_dropped == 1


# --- columns: open interest, volume, duplicates -------------------------------------


def test_open_interest_of_zero_on_every_row_becomes_nan():
    rows = [dict(row, oi=0) for row in truth_rows(FakeCalendar(), "NSE", [MON])]
    bars, report = normalise(rows)
    assert bars["oi"].isna().all()
    assert report.oi_zero_to_nan == NSE_MINUTES_PER_DAY


def test_series_with_real_open_interest_keeps_its_zeros():
    rows = [dict(row, oi=0) for row in truth_rows(FakeCalendar(), "NSE", [MON])]
    rows[10]["oi"] = 12_500
    bars, report = normalise(rows)
    assert bars["oi"].iloc[10] == 12_500
    assert bars["oi"].iloc[0] == 0.0
    assert report.oi_zero_to_nan == 0


def test_missing_volume_and_open_interest_are_nan_and_noted():
    rows = [
        {k: v for k, v in row.items() if k != "volume"}
        for row in truth_rows(FakeCalendar(), "NSE", [MON])
    ]
    bars, report = normalise(rows)
    assert bars["volume"].isna().all()
    assert bars["oi"].isna().all()
    assert report.notes == ["the source has no volume column: volume is NaN"]


def test_duplicate_timestamps_keep_the_first_row_and_are_counted():
    rows = truth_rows(FakeCalendar(), "NSE", [MON])
    same_again = rows[3]
    different_close = dict(rows[5], close=rows[5]["close"] + 0.25)
    bars, report = normalise(rows + [same_again, different_close])
    assert len(bars) == NSE_MINUTES_PER_DAY
    assert bars["close"].iloc[5] == rows[5]["close"]
    assert (report.duplicates_removed, report.duplicates_conflicting) == (2, 1)


@settings(max_examples=25, deadline=None)
@given(order=st.permutations(list(range(60))), repeats=st.lists(st.integers(0, 59), max_size=10))
def test_row_order_and_repeated_rows_do_not_change_the_bars(order, repeats):
    rows = truth_rows(FakeCalendar(), "NSE", [MON])
    first_hour, rest = rows[:60], rows[60:]
    scrambled = [first_hour[i] for i in order] + [first_hour[i] for i in repeats] + rest
    bars, report = normalise(scrambled)
    assert bars["ts_open"].tolist() == open_ns(rows)
    assert bars["close"].tolist() == [row["close"] for row in rows]
    assert report.duplicates_removed == len(repeats)


# --- the layout check ---------------------------------------------------------------


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda bars: bars.drop(columns=["oi"]), "columns must be exactly"),
        (lambda bars: bars.astype({"ts_open": "float64"}), "int64 UTC nanoseconds"),
        (lambda bars: bars.astype({"volume": "int64"}), "must be float64"),
        (lambda bars: bars.iloc[::-1].reset_index(drop=True), "sorted by ts_open"),
        (lambda bars: bars.set_index(np.arange(5, 5 + len(bars))), "RangeIndex"),
        (lambda bars: bars.assign(ts_close=bars["ts_open"]), "ts_close must be after"),
        (lambda bars: bars.to_dict("list"), "expected a pandas DataFrame"),
    ],
    ids=["missing column", "float time", "int volume", "unsorted", "index", "zero length", "dict"],
)
def test_validate_layout_rejects_a_frame_that_is_not_canonical(damage, message):
    bars, _ = normalise(truth_rows(FakeCalendar(), "NSE", [MON])[:10])
    with pytest.raises(DataError, match=message):
        validate_layout(damage(bars))
