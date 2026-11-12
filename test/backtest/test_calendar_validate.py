"""Checking an intraday series against the session calendar (FR-DATA-06).

Bars sit on the lattice: the exchange's regular open plus whole intervals. A 5 hour 30
minute offset, bars stamped at their close, bars off the lattice and bars on closed
days are errors whose messages name the likely cause; nothing is shifted silently.
"""

import datetime as dt

import numpy as np
import pytest
from calendar_fixtures.builders import NS_PER_MINUTE, at_ist, bars_frame, nse_session, opens_between

from openalgo.backtest.calendar import Calendar, CalendarWarning
from openalgo.backtest.errors import ConfigError, TimestampError

MONDAY = dt.date(2025, 6, 2)
WEEK = [MONDAY + dt.timedelta(days=offset) for offset in range(5)]
SATURDAY = dt.date(2025, 6, 7)
IST_SHIFT_NS = (5 * 60 + 30) * NS_PER_MINUTE

# Every key of the check report, as documented in openalgo.backtest._calendar_series.
REPORT_KEYS = {
    "bars", "days", "session_days", "closed_days", "first_closed_day", "closed_day_bars",
    "days_starting_at_open", "days_starting_late", "first_late_day", "days_without_session_bars",
    "days_with_bar_at_close", "first_bar_at_close", "pre_open_bars", "post_close_bars",
    "off_lattice_bars", "first_off_lattice", "bars_in_session", "missing_bars", "missing_days",
    "duplicate_bars", "template_only_days", "unverified_hours_days", "close_labelled", "offset",
    "offset_check", "offset_direction", "offset_reason", "finding", "placement",
}  # fmt: skip


@pytest.fixture
def cal() -> Calendar:
    return Calendar(builtin=False)


def week_of_minutes() -> np.ndarray:
    return np.concatenate([nse_session(day) for day in WEEK])


def test_the_report_has_exactly_the_documented_keys(cal):
    assert (
        set(cal.check_intraday(nse_session(MONDAY), exchange="NSE", interval="1m")) == REPORT_KEYS
    )


def test_a_series_on_the_lattice_is_valid(cal):
    report = cal.validate_intraday(week_of_minutes(), exchange="NSE", interval="1m")
    assert (report["bars"], report["days_starting_at_open"], report["missing_bars"]) == (
        5 * 375,
        5,
        0,
    )
    assert (report["offset_check"], report["finding"]) == ("aligned", None)


def test_exchange_and_interval_default_to_the_frame_attrs(cal):
    bars = bars_frame(nse_session(MONDAY), exchange="NSE", interval="1m")
    assert cal.validate_intraday(bars)["bars"] == 375


def test_exchange_and_interval_are_required_for_a_bare_array(cal):
    with pytest.raises(ConfigError, match="exchange and interval are required"):
        cal.validate_intraday(nse_session(MONDAY))


def test_timestamps_5h30_late_raise_a_timestamp_error_naming_the_cause(cal):
    with pytest.raises(TimestampError) as raised:
        cal.validate_intraday(week_of_minutes() + IST_SHIFT_NS, exchange="NSE", interval="1m")
    message = str(raised.value)
    assert "5 hours 30 minutes late" in message
    assert "IST clock times were stored as if they were UTC" in message
    assert "2025-06-02 14:45:00 IST" in message  # where the first bar landed
    assert "never shifted silently" in message


def test_timestamps_5h30_early_raise_a_timestamp_error(cal):
    with pytest.raises(TimestampError, match="5 hours 30 minutes early"):
        cal.validate_intraday(week_of_minutes() - IST_SHIFT_NS, exchange="NSE", interval="1m")


def test_early_mcx_stamps_on_two_days_raise_a_timestamp_error(cal):
    days = [dt.date(2025, 6, 3), dt.date(2025, 6, 5)]
    stamps = np.concatenate([opens_between(day, (9, 0), (23, 30)) for day in days]) - IST_SHIFT_NS
    with pytest.raises(TimestampError, match="5 hours 30 minutes early"):
        cal.validate_intraday(stamps, exchange="MCX", interval="1m")


def test_one_shifted_day_only_warns_with_the_reason(cal):
    with pytest.warns(CalendarWarning, match="inconclusive"):
        report = cal.validate_intraday(
            nse_session(MONDAY) + IST_SHIFT_NS, exchange="NSE", interval="1m"
        )
    assert (report["offset_check"], report["offset_direction"]) == ("inconclusive", "late_5h30")


def test_bars_labelled_by_their_close_raise_a_timestamp_error(cal):
    five = np.concatenate([nse_session(day, step_minutes=5) for day in WEEK])
    with pytest.raises(TimestampError, match="labelled by their close time"):
        cal.validate_intraday(five + 5 * NS_PER_MINUTE, exchange="NSE", interval="5m")


def test_close_labelled_bars_are_a_finding_distinct_from_an_offset(cal):
    stamps = week_of_minutes() + NS_PER_MINUTE
    report = cal.check_intraday(stamps, exchange="NSE", interval="1m")
    assert (report["finding"], report["offset"], report["close_labelled"]) == (
        "close_labelled",
        None,
        True,
    )


def test_a_shifted_series_is_never_reported_as_close_labelled(cal):
    report = cal.check_intraday(week_of_minutes() + IST_SHIFT_NS, exchange="NSE", interval="1m")
    assert (report["finding"], report["close_labelled"]) == ("late_5h30", False)


def test_bars_anchored_to_the_hour_raise_a_timestamp_error(cal):
    hourly = np.array([at_ist(MONDAY, hour, 0) for hour in range(10, 16)], dtype=np.int64)
    with pytest.raises(TimestampError, match="not on the NSE bar lattice"):
        cal.validate_intraday(hourly, exchange="NSE", interval="1h")


def test_bars_on_a_closed_day_raise_and_say_how_to_add_the_session(cal):
    with pytest.raises(TimestampError) as raised:
        cal.validate_intraday(nse_session(SATURDAY), exchange="NSE", interval="1m")
    assert "2025-06-07" in str(raised.value) and "add_session" in str(raised.value)


def test_bars_on_a_closed_day_pass_when_allowed(cal):
    report = cal.validate_intraday(
        nse_session(SATURDAY), exchange="NSE", interval="1m", allow_closed_days=True
    )
    assert (report["closed_days"], report["first_closed_day"]) == (1, SATURDAY)


def test_a_special_session_in_the_calendar_makes_its_bars_valid(cal):
    cal.add_session("NSE", SATURDAY, "09:15", "15:30")
    assert (
        cal.validate_intraday(nse_session(SATURDAY), exchange="NSE", interval="1m")["closed_days"]
        == 0
    )


def test_server_lattice_bars_of_a_special_session_are_valid(cal):
    sunday = dt.date(2026, 11, 8)
    cal.add_session("NSE", sunday, "14:20", "15:30")
    stamps = opens_between(sunday, (14, 15), (15, 30), step_minutes=15)  # 14:15 straddles the open
    assert cal.validate_intraday(stamps, exchange="NSE", interval="15m")["bars_in_session"] == 5


def test_a_late_first_bar_is_counted_but_valid(cal):
    sparse = np.array([at_ist(MONDAY, 11, 3), at_ist(MONDAY, 15, 10)], dtype=np.int64)
    report = cal.validate_intraday(sparse, exchange="NSE", interval="1m")
    assert (report["days_starting_late"], report["first_late_day"]) == (1, MONDAY)
    assert report["missing_bars"] == 375 - 2


def test_check_counts_an_offset_without_raising(cal):
    report = cal.check_intraday(week_of_minutes() + IST_SHIFT_NS, exchange="NSE", interval="1m")
    assert (report["offset"], report["offset_check"]) == ("late_5h30", "shifted")
    assert report["bars_in_session"] == 5 * 45  # 14:45 to 15:30 each day


def test_check_counts_bars_outside_the_session_and_places_each_bar(cal):
    opens = np.concatenate(
        [
            [at_ist(MONDAY, 9, 7), at_ist(MONDAY, 9, 8)],
            nse_session(MONDAY),
            [at_ist(MONDAY, 15, 40)],
        ]
    )
    report = cal.check_intraday(opens, exchange="NSE", interval="1m")
    assert (report["pre_open_bars"], report["post_close_bars"], report["bars_in_session"]) == (
        2,
        1,
        375,
    )
    assert report["placement"].tolist() == [1, 1] + [0] * 375 + [2]  # PLACEMENTS codes


def test_check_reports_days_that_rest_on_the_weekday_rule(cal):
    assert (
        cal.check_intraday(week_of_minutes(), exchange="NSE", interval="1m")["template_only_days"]
        == 5
    )


def test_datetime64_timestamps_are_accepted(cal):
    stamps = nse_session(MONDAY).astype("datetime64[ns]")
    assert cal.validate_intraday(stamps, exchange="NSE", interval="1m")["bars"] == 375


def test_a_missing_timestamp_raises_a_timestamp_error(cal):
    stamps = nse_session(MONDAY).astype("datetime64[ns]")
    stamps[3] = np.datetime64("NaT", "ns")
    with pytest.raises(TimestampError, match="NaT"):
        cal.check_intraday(stamps, exchange="NSE", interval="1m")


def test_unsorted_timestamps_raise_a_timestamp_error(cal):
    with pytest.raises(TimestampError, match="not sorted"):
        cal.check_intraday(nse_session(MONDAY)[::-1].copy(), exchange="NSE", interval="1m")


def test_a_daily_interval_is_a_config_error(cal):
    with pytest.raises(ConfigError, match="not intraday"):
        cal.validate_intraday(nse_session(MONDAY), exchange="NSE", interval="D")


def test_float_timestamps_are_a_config_error(cal):
    with pytest.raises(ConfigError, match="int64 UTC nanoseconds"):
        cal.validate_intraday(nse_session(MONDAY).astype(float), exchange="NSE", interval="1m")
