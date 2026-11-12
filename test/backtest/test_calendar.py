"""Session calendar: regular hours, weekends, dated rules and the session queries.

These tests use `Calendar(builtin=False)`, so only the regular hours and the rules a
test adds are in play (FR-MKT-02). Expected instants are written out with `at_ist`,
which is plain epoch arithmetic. The regular hours are the rows of the kernel's
`regular_hours.csv`, whose sources are cited there.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from calendar_fixtures.builders import at_ist

from openalgo.backtest._calendar_convert import native_attr
from openalgo.backtest._protocols import SessionCalendar
from openalgo.backtest.calendar import Calendar, default_calendar
from openalgo.backtest.errors import CalendarError, ConfigError

MONDAY = dt.date(2025, 6, 2)
SATURDAY = dt.date(2025, 6, 7)
SUNDAY = dt.date(2025, 6, 8)


@pytest.fixture
def cal() -> Calendar:
    return Calendar(builtin=False)


def session_at(day: dt.date, open_: tuple[int, int], close: tuple[int, int]) -> tuple[int, int]:
    return at_ist(day, *open_), at_ist(day, *close)


def test_calendar_satisfies_the_session_calendar_protocol(cal):
    assert isinstance(cal, SessionCalendar)


@pytest.mark.parametrize(
    ("exchange", "open_", "close"),
    [
        ("NSE", (9, 15), (15, 30)),
        ("BSE", (9, 15), (15, 30)),
        ("NFO", (9, 15), (15, 30)),
        ("BFO", (9, 15), (15, 30)),
        ("CDS", (9, 0), (17, 0)),
        ("BCD", (9, 0), (17, 0)),
        ("MCX", (9, 0), (23, 30)),
        ("NSE_INDEX", (9, 15), (15, 30)),
        ("BSE_INDEX", (9, 15), (15, 30)),
    ],
)
def test_a_weekday_without_rules_has_the_regular_session(cal, exchange, open_, close):
    assert cal.session(exchange, MONDAY) == session_at(MONDAY, open_, close)


@pytest.mark.parametrize("day", [SATURDAY, SUNDAY])
def test_weekends_are_closed(cal, day):
    assert cal.session("NSE", day) is None
    assert not cal.is_open("MCX", day)


@pytest.mark.parametrize("exchange", ["nse", "NCDEX", "", None])
def test_an_unknown_exchange_is_a_calendar_error(cal, exchange):
    with pytest.raises(CalendarError, match="unknown exchange"):
        cal.session(exchange, MONDAY)


@pytest.mark.parametrize("day", ["02-06-2025", 20250602, None])
def test_an_unreadable_date_is_a_config_error(cal, day):
    with pytest.raises(ConfigError, match="as a date"):
        cal.session("NSE", day)


@pytest.mark.parametrize("day", [dt.date(1600, 1, 3), dt.date(2300, 1, 2), dt.date(9999, 12, 31)])
def test_a_date_outside_the_supported_range_is_a_calendar_error(cal, day):
    with pytest.raises(CalendarError, match="outside the supported range"):
        cal.session("NSE", day)


def test_adding_a_rule_on_an_unsupported_date_is_a_calendar_error(cal):
    with pytest.raises(CalendarError, match="outside the supported range"):
        cal.add_closure("NSE", dt.date(2300, 1, 2))


@pytest.mark.parametrize(
    "day",
    [
        "2025-06-02",
        dt.datetime(2025, 6, 2, 10, 30),
        dt.datetime(2025, 6, 1, 20, 0, tzinfo=dt.UTC),  # 01:30 IST on 2 June
        pd.Timestamp("2025-06-02 09:15", tz="Asia/Kolkata"),
        np.datetime64("2025-06-02"),
    ],
)
def test_dates_are_read_as_ist_dates_from_common_types(cal, day):
    assert cal.session("NSE", day) == session_at(MONDAY, (9, 15), (15, 30))


@pytest.mark.parametrize(
    ("day", "close"),
    [
        (dt.date(2026, 7, 31), (15, 30)),
        (dt.date(2026, 8, 3), (15, 40)),  # NSE/FAOP/74467
        (dt.date(2026, 12, 4), (15, 40)),
        (dt.date(2026, 12, 7), (15, 45)),  # NSE/FAOP/77356
    ],
)
def test_equity_derivatives_close_later_from_each_cited_date(cal, day, close):
    assert cal.session("NFO", day) == session_at(day, (9, 15), close)
    assert cal.session("BFO", day) == session_at(day, (9, 15), close)
    assert cal.session("NSE", day) == session_at(day, (9, 15), (15, 30))


@pytest.mark.parametrize(
    ("day", "close"),
    [
        (dt.date(2025, 3, 7), (23, 55)),  # last trading day before the US switch
        (dt.date(2025, 3, 10), (23, 30)),
        (dt.date(2025, 10, 31), (23, 30)),
        (dt.date(2025, 11, 3), (23, 55)),
    ],
)
def test_the_mcx_close_follows_us_daylight_saving_and_is_unverified(cal, day, close):
    info = cal.session_info("MCX", day)
    assert (info.open, info.close) == session_at(day, (9, 0), close)
    assert info.hours_unverified


@pytest.mark.parametrize(
    ("exchange", "day", "unverified"),
    [
        ("NSE", dt.date(2026, 4, 27), True),  # before NSE/CMTR/73927
        ("NSE", dt.date(2026, 4, 28), False),
        ("NFO", dt.date(2026, 12, 7), False),
        ("BFO", dt.date(2026, 12, 7), True),  # no BSE notice was read
    ],
)
def test_hours_are_flagged_unless_an_exchange_document_states_them(cal, exchange, day, unverified):
    assert cal.session_info(exchange, day).hours_unverified is unverified


def test_a_template_session_reports_regular_kind_and_template_coverage(cal):
    info = cal.session_info("NSE", MONDAY)
    assert (info.exchange, info.date, info.kind, info.coverage) == (
        "NSE",
        MONDAY,
        "regular",
        "template",
    )


def test_a_closure_closes_only_the_exchanges_it_names(cal):
    cal.add_closure(["NSE", "NFO"], "2026-03-03", name="Holi")
    assert cal.session("NSE", "2026-03-03") is None
    assert cal.session("NSE_INDEX", "2026-03-03") is None
    assert cal.session("NFO", "2026-03-03") is None
    assert cal.is_open("BSE", "2026-03-03")
    assert cal.is_open("MCX", "2026-03-03")


def test_a_special_session_opens_a_weekend_day(cal):
    sunday = dt.date(2026, 11, 8)
    cal.add_session("NSE", sunday, "14:20", "15:30", name="Muhurat")
    info = cal.session_info("NSE", sunday)
    assert (info.open, info.close) == session_at(sunday, (14, 20), (15, 30))
    assert (info.kind, info.coverage) == ("special", "sourced")
    assert cal.session("MCX", sunday) is None


def test_an_unverified_rule_reports_unverified_coverage(cal):
    cal.add_session("NSE", SATURDAY, "09:15", "15:30", verified=False)
    assert cal.coverage("NSE", SATURDAY) == "unverified"


def test_a_session_can_flag_its_times_as_unverified(cal):
    cal.add_session("NSE", SATURDAY, "09:15", "12:30", hours_unverified=True)
    assert cal.session_info("NSE", SATURDAY).hours_unverified


@pytest.mark.parametrize(
    ("open_", "close"), [("18:00", "24:15"), ("15:30", "09:15"), ("09:15", "09:15")]
)
def test_session_times_must_be_ordered_and_end_by_midnight(cal, open_, close):
    with pytest.raises((CalendarError, ConfigError)):
        cal.add_session("MCX", MONDAY, open_, close)


def test_a_session_may_close_at_midnight(cal):
    cal.add_session("MCX", MONDAY, "18:00", "24:00")
    next_day = MONDAY + dt.timedelta(days=1)
    assert cal.session("MCX", MONDAY) == (at_ist(MONDAY, 18), at_ist(next_day, 0))


def test_an_evening_session_without_a_close_ends_at_the_regular_close(cal):
    winter = dt.date(2025, 12, 1)
    cal.add_session("MCX", winter, "17:00")
    assert cal.session("MCX", winter) == session_at(winter, (17, 0), (23, 55))


def test_new_mcx_hours_reach_an_evening_session_added_earlier(cal):
    cal.add_session("MCX", MONDAY, "17:00")
    cal.set_regular_hours("MCX", "09:00", "23:00")
    assert cal.session("MCX", MONDAY) == session_at(MONDAY, (17, 0), (23, 0))


def test_regular_hours_set_without_a_date_apply_to_every_date_as_verified(cal):
    cal.set_regular_hours("MCX", "09:00", "23:55")
    info = cal.session_info("MCX", MONDAY)
    assert (info.open, info.close, info.hours_unverified) == (
        *session_at(MONDAY, (9, 0), (23, 55)),
        False,
    )


def test_regular_hours_set_from_a_date_apply_from_that_date_only(cal):
    cal.set_regular_hours("CDS", "09:00", "15:00", effective_from="2030-01-01")
    assert cal.session("CDS", "2029-12-31") == session_at(dt.date(2029, 12, 31), (9, 0), (17, 0))
    assert cal.session("CDS", "2030-01-01") == session_at(dt.date(2030, 1, 1), (9, 0), (15, 0))


def test_sessions_lists_one_row_per_session_in_the_range(cal):
    cal.add_closure("NSE", "2026-03-03")
    frame = cal.sessions("NSE", "2026-03-01", "2026-03-08")  # Sunday to Sunday
    expected_days = [dt.date(2026, 3, day) for day in (2, 4, 5, 6)]
    assert frame["date"].tolist() == expected_days
    assert frame["open"].tolist() == [at_ist(day, 9, 15) for day in expected_days]
    assert frame["close"].tolist() == [at_ist(day, 15, 30) for day in expected_days]
    assert list(frame.columns) == ["date", "open", "close", "kind", "coverage", "hours_unverified"]
    assert frame.attrs["exchange"] == "NSE"


def test_a_reversed_range_is_a_calendar_error(cal):
    with pytest.raises(CalendarError, match="start 2026-03-08 is after end 2026-03-01"):
        cal.sessions("NSE", "2026-03-08", "2026-03-01")


@pytest.mark.parametrize(
    ("exchange", "day", "entry_until", "square_off", "relative"),
    [
        ("NSE", MONDAY, (15, 0), (15, 10), False),
        ("NSE", dt.date(2026, 9, 7), (14, 50), (15, 0), False),  # from 2026-08-03
        ("NFO", dt.date(2026, 8, 3), (15, 0), (15, 10), False),
        ("CDS", MONDAY, (16, 30), (16, 40), False),
        ("MCX", MONDAY, (23, 0), (23, 10), True),  # close 23:30
        ("MCX", dt.date(2025, 1, 6), (23, 25), (23, 35), True),  # close 23:55
    ],
)
def test_default_intraday_times_per_segment(cal, exchange, day, entry_until, square_off, relative):
    times = cal.intraday_times(exchange, day)
    assert (times.entry_until, times.square_off) == (
        at_ist(day, *entry_until),
        at_ist(day, *square_off),
    )
    assert times.relative_to_close is relative


def test_intraday_times_of_an_evening_session_are_relative_to_its_close(cal):
    day = dt.date(2024, 11, 1)
    cal.add_session("NSE", day, "18:00", "19:00")
    times = cal.intraday_times("NSE", day)
    assert (times.entry_until, times.square_off) == (at_ist(day, 18, 30), at_ist(day, 18, 40))
    assert times.relative_to_close


@pytest.mark.parametrize(
    ("exchange", "day", "unverified"),
    [("MCX", MONDAY, True), ("NSE", MONDAY, False), ("NSE", dt.date(2026, 9, 7), False)],
)
def test_intraday_times_say_when_they_rest_on_unconfirmed_data(cal, exchange, day, unverified):
    assert cal.intraday_times(exchange, day).unverified is unverified


def test_a_closed_day_has_no_intraday_times(cal):
    assert cal.intraday_times("NSE", SATURDAY) is None


def test_the_default_calendar_is_shared_and_read_only():
    shared = default_calendar()
    assert default_calendar() is shared
    for change in (
        lambda: shared.add_closure("NSE", MONDAY),
        lambda: shared.add_holidays([]),
        lambda: shared.set_regular_hours("NSE", "09:15", "15:30"),
    ):
        with pytest.raises(CalendarError, match="read-only"):
            change()


def test_the_kernel_object_of_the_default_calendar_is_read_only_too():
    kernel_error = native_attr("NativeCalendarError")
    with pytest.raises(kernel_error, match="read-only"):
        default_calendar()._cal.set_closed("NSE", 20_241, 2)  # the package's own handle


def test_a_copy_of_the_default_calendar_can_be_edited():
    clone = default_calendar().copy()
    clone.add_closure("NSE", MONDAY)
    assert clone.session("NSE", MONDAY) is None
    assert default_calendar().is_open("NSE", MONDAY)


def test_holidays_lists_the_dated_rules_of_a_year(cal):
    cal.add_closure("NSE", "2026-03-03", name="Holi")
    cal.add_session("NSE", "2026-11-08", "14:20", "15:30", name="Muhurat")
    frame = cal.holidays("NSE", 2026)
    assert frame["date"].tolist() == [dt.date(2026, 3, 3), dt.date(2026, 11, 8)]
    assert frame["name"].tolist() == ["Holi", "Muhurat"]
    assert frame["open"].tolist() == [None, "14:20"]
    assert frame["close"].tolist() == [None, "15:30"]


def test_holiday_names_belong_to_the_exchange_they_were_given_for(cal):
    cal.add_closure("NSE", "2026-03-03", name="Holi")
    cal.add_closure("MCX", "2026-03-03", name="MCX morning closed")
    assert cal.holidays("NSE_INDEX", 2026)["name"].tolist() == ["Holi"]
    assert cal.holidays("MCX", 2026)["name"].tolist() == ["MCX morning closed"]
