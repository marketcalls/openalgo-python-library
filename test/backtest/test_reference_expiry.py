"""Contracts and their dates from the weekday rules, contract dates and a session calendar.

The rules used here are the small made-up table in test_reference_helpers.py: NIFTY
weekly and monthly on Thursday, moving to Tuesday from 1 September 2025; BANKNIFTY
weekly on Wednesday with the monthly contract on the last Thursday, and no weekly
series after 13 November 2024.
"""

import pytest
from test_reference_helpers import RULES, FakeCalendar, build_tables, day, write_file

from openalgo.backtest._protocols import SessionCalendar
from openalgo.backtest.errors import CalendarError, ConfigError, ReferenceDataError
from openalgo.backtest.reference import Expiry, ReferenceTables


@pytest.fixture
def tables(tmp_path):
    return build_tables(tmp_path, only=["rules.csv"])


def _dates(*texts):
    return [day(text) for text in texts]


def test_fake_calendar_satisfies_the_session_calendar_protocol():
    assert isinstance(FakeCalendar(), SessionCalendar)


# ---- the weekday rule ---------------------------------------------------------------------


def test_weekly_expiries_fall_on_every_rule_weekday(tables):
    found = tables.expiries("NIFTY", "NFO", "2024-11-01", "2024-11-30", FakeCalendar())
    assert found == _dates("2024-11-07", "2024-11-14", "2024-11-21", "2024-11-28")


def test_monthly_expiry_is_the_last_rule_weekday_of_the_month(tables):
    found = tables.expiries(
        "NIFTY", "NFO", "2024-10-01", "2024-12-31", FakeCalendar(), series="monthly"
    )
    assert found == _dates("2024-10-31", "2024-11-28", "2024-12-26")


def test_last_weekly_date_of_the_month_is_reported_as_the_monthly_expiry(tables):
    schedule = tables.expiry_schedule("NIFTY", "NFO", "2024-11-21", "2024-11-28", FakeCalendar())
    assert [(item.last_trading_day, item.series) for item in schedule] == [
        (day("2024-11-21"), "weekly"),
        (day("2024-11-28"), "monthly"),
    ]


def test_weekly_series_filter_leaves_out_the_monthly_expiry(tables):
    found = tables.expiries(
        "NIFTY", "NFO", "2024-11-01", "2024-11-30", FakeCalendar(), series="weekly"
    )
    assert found == _dates("2024-11-07", "2024-11-14", "2024-11-21")


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ("2024-11-07", "2024-11-14", ["2024-11-07", "2024-11-14"]),  # both ends inclusive
        ("2024-11-08", "2024-11-13", []),
        ("2024-11-14", "2024-11-14", ["2024-11-14"]),
    ],
)
def test_requested_range_is_inclusive_at_both_ends(tables, start, end, expected):
    assert tables.expiries("NIFTY", "NFO", start, end, FakeCalendar()) == _dates(*expected)


def test_calendar_is_asked_about_the_derivatives_exchange(tables):
    calendar = FakeCalendar()
    tables.expiries("NIFTY", "NSE_INDEX", "2024-11-01", "2024-11-30", calendar)
    assert {exchange for exchange, _ in calendar.calls} == {"NFO"}


def test_expiry_record_carries_the_contract_its_dates_and_the_source_row(tables):
    (expiry,) = tables.expiry_schedule("NIFTY", "NFO", "2024-11-14", "2024-11-14", FakeCalendar())
    assert expiry == Expiry(
        underlying="NIFTY",
        exchange="NFO",
        series="weekly",
        contract_date=day("2024-11-14"),
        expiry=day("2024-11-14"),
        last_trading_day=day("2024-11-14"),
        source="test",
        status="user",
    )


# ---- holidays -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "holidays,expiry",
    [
        (["2024-11-14"], "2024-11-13"),  # Thursday closed: Wednesday
        (["2024-11-14", "2024-11-13"], "2024-11-12"),  # two closed days: Tuesday
        (["2024-11-14", "2024-11-13", "2024-11-12", "2024-11-11"], "2024-11-08"),  # over a weekend
    ],
)
def test_expiry_on_a_holiday_moves_to_the_previous_trading_day(tables, holidays, expiry):
    calendar = FakeCalendar(holidays=_dates(*holidays))
    assert tables.expiries("NIFTY", "NFO", "2024-11-08", "2024-11-15", calendar) == [day(expiry)]


def test_moved_expiry_keeps_its_contract_date(tables):
    calendar = FakeCalendar(holidays=_dates("2024-11-14"))
    (expiry,) = tables.expiry_schedule("NIFTY", "NFO", "2024-11-11", "2024-11-15", calendar)
    assert (expiry.expiry, expiry.last_trading_day, expiry.contract_date) == (
        day("2024-11-13"),
        day("2024-11-13"),
        day("2024-11-14"),
    )


def test_expiry_moved_back_into_the_range_from_just_after_it_is_included(tables):
    # The rule date (Thursday 14) is outside the range; the actual expiry (13) is inside.
    calendar = FakeCalendar(holidays=_dates("2024-11-14"))
    found = tables.expiries("NIFTY", "NFO", "2024-11-11", "2024-11-13", calendar)
    assert found == _dates("2024-11-13")


def test_expiry_moved_out_of_the_range_is_left_out(tables):
    calendar = FakeCalendar(holidays=_dates("2024-11-14"))
    assert tables.expiries("NIFTY", "NFO", "2024-11-14", "2024-11-15", calendar) == []


def test_calendar_with_no_session_for_weeks_is_a_calendar_error(tables):
    class Closed:
        def session(self, exchange, on_day):
            return None

        def coverage(self, exchange, on_day):
            return "sourced"

    with pytest.raises(CalendarError, match="no NFO session"):
        tables.expiries("NIFTY", "NFO", "2024-11-01", "2024-11-30", Closed())


# ---- rule changes ---------------------------------------------------------------------------


def test_weekday_change_takes_effect_from_the_first_expiry_on_or_after_its_date(tables):
    found = tables.expiries("NIFTY", "NFO", "2025-08-18", "2025-09-12", FakeCalendar())
    assert found == _dates("2025-08-21", "2025-08-28", "2025-09-02", "2025-09-09")


def test_monthly_expiry_follows_the_weekday_in_force_for_that_month(tables):
    found = tables.expiries(
        "NIFTY", "NFO", "2025-08-01", "2025-10-31", FakeCalendar(), series="monthly"
    )
    assert found == _dates("2025-08-28", "2025-09-30", "2025-10-28")


@pytest.mark.parametrize(
    "on_date,series,weekday",
    [
        ("2025-08-31", "weekly", "THU"),
        ("2025-09-01", "weekly", "TUE"),
        ("2019-02-10", "weekly", "NONE"),
        ("2024-02-29", "monthly", "THU"),
    ],
)
def test_expiry_rule_reports_the_weekday_in_force(tables, on_date, series, weekday):
    underlying = "BANKNIFTY" if series == "monthly" else "NIFTY"
    assert tables.expiry_rule(underlying, "NFO", on_date, series) == weekday


def test_series_not_yet_listed_gives_no_weekly_expiries(tables):
    # Weekly contracts start on 11 February 2019 in this table; January has the monthly only.
    found = tables.expiries("NIFTY", "NFO", "2019-01-01", "2019-02-28", FakeCalendar())
    assert found == _dates("2019-01-31", "2019-02-14", "2019-02-21", "2019-02-28")


def test_discontinued_weekly_series_stops_after_its_last_expiry(tables):
    calendar = FakeCalendar(holidays=_dates("2024-12-25"))
    found = tables.expiries("BANKNIFTY", "NFO", "2024-11-01", "2024-12-31", calendar)
    assert found == _dates("2024-11-06", "2024-11-13", "2024-11-27", "2024-12-24")


def test_no_separate_weekly_expiry_in_the_week_of_the_monthly_expiry(tables):
    # Weekly on Wednesday, monthly on the last Thursday (28 September 2023): Wednesday
    # 27 September has no weekly contract.
    schedule = tables.expiry_schedule(
        "BANKNIFTY", "NFO", "2023-09-04", "2023-10-06", FakeCalendar()
    )
    assert [(item.last_trading_day, item.series) for item in schedule] == [
        (day("2023-09-06"), "weekly"),
        (day("2023-09-13"), "weekly"),
        (day("2023-09-20"), "weekly"),
        (day("2023-09-28"), "monthly"),
        (day("2023-10-04"), "weekly"),
    ]


def test_weekly_date_is_dropped_even_when_the_monthly_expiry_lies_outside_the_range(tables):
    # Wednesday 27 September 2023 is in range, Thursday 28 (the monthly) is not.
    found = tables.expiries("BANKNIFTY", "NFO", "2023-09-25", "2023-09-27", FakeCalendar())
    assert found == []


# ---- contract dates ------------------------------------------------------------------------

CONTRACT_HEADER = (
    "underlying,exchange,contract,valid_from,valid_to,expiry_date,last_trading_day,source\n"
)


def _with_contracts(tmp_path, body):
    files = [
        write_file(tmp_path, "rules.csv", RULES),
        write_file(tmp_path, "dates.csv", CONTRACT_HEADER + body),
    ]
    return ReferenceTables(user_files=files, builtin=False)


def test_contract_row_replaces_the_rule_date(tmp_path):
    body = "NIFTY,NFO,2024-11-14,2019-01-01,,2024-11-12,2024-11-12,exchange notice\n"
    tables = _with_contracts(tmp_path, body)
    (expiry,) = tables.expiry_schedule("NIFTY", "NFO", "2024-11-11", "2024-11-15", FakeCalendar())
    assert (expiry.expiry, expiry.contract_date, expiry.source) == (
        day("2024-11-12"),
        day("2024-11-14"),
        "exchange notice",
    )


def test_contract_row_is_used_without_asking_the_calendar(tmp_path):
    body = "NIFTY,NFO,2024-11-14,2019-01-01,,2024-11-12,2024-11-12,exchange notice\n"
    calendar = FakeCalendar()
    _with_contracts(tmp_path, body).expiries("NIFTY", "NFO", "2024-11-11", "2024-11-15", calendar)
    assert ("NFO", day("2024-11-14")) not in calendar.calls


def test_contract_row_may_set_a_date_later_than_the_rule_date(tmp_path):
    body = "NIFTY,NFO,2024-11-14,2019-01-01,,2024-11-15,2024-11-15,exchange notice\n"
    tables = _with_contracts(tmp_path, body)
    assert tables.expiries("NIFTY", "NFO", "2024-11-15", "2024-11-15", FakeCalendar()) == _dates(
        "2024-11-15"
    )


def test_contract_dates_are_those_in_force_on_the_trade_date(tmp_path):
    body = (
        "NIFTY,NFO,2024-11-14,2019-01-01,2024-11-08,2024-11-14,2024-11-14,listing\n"
        "NIFTY,NFO,2024-11-14,2024-11-09,,2024-11-13,2024-11-13,re-dating notice\n"
    )
    tables = _with_contracts(tmp_path, body)

    def listed(as_of):
        schedule = tables.expiry_schedule(
            "NIFTY", "NFO", "2024-11-11", "2024-11-15", FakeCalendar(), as_of=as_of
        )
        return [item.expiry for item in schedule]

    assert listed("2024-11-08") == _dates("2024-11-14")
    assert listed("2024-11-11") == _dates("2024-11-13")
    assert listed(None) == _dates("2024-11-13")  # the final dates


def test_listed_expiry_and_last_trading_day_may_differ(tmp_path):
    body = "NIFTY,NFO,2024-11-14,2019-01-01,,2024-11-14,2024-11-12,exchange notice\n"
    tables = _with_contracts(tmp_path, body)
    (expiry,) = tables.expiry_schedule("NIFTY", "NFO", "2024-11-11", "2024-11-15", FakeCalendar())
    assert (expiry.expiry, expiry.last_trading_day) == (day("2024-11-14"), day("2024-11-12"))
    assert tables.expiries("NIFTY", "NFO", "2024-11-11", "2024-11-15", FakeCalendar()) == _dates(
        "2024-11-12"
    )


# ---- coverage and arguments -------------------------------------------------------------------


@pytest.mark.parametrize(
    "underlying,start,end,uncovered",
    [
        ("NIFTY", "2018-12-01", "2019-01-31", "2018-12-01"),  # starts before the first rule
        ("BANKNIFTY", "2023-08-01", "2023-09-30", "2023-08-01"),
        ("FINNIFTY", "2024-01-01", "2024-01-31", "2024-01-01"),  # no rule at all
    ],
)
def test_range_the_rules_do_not_cover_raises_and_names_the_first_uncovered_day(
    tables, underlying, start, end, uncovered
):
    with pytest.raises(ReferenceDataError) as excinfo:
        tables.expiries(underlying, "NFO", start, end, FakeCalendar())
    assert "expiry_rules" in str(excinfo.value) and uncovered in str(excinfo.value)


def test_expiries_without_a_calendar_are_refused(tables):
    with pytest.raises(ConfigError, match="session calendar"):
        tables.expiries("NIFTY", "NFO", "2024-11-01", "2024-11-30", None)


def test_reversed_range_is_refused(tables):
    with pytest.raises(ConfigError, match="before start"):
        tables.expiries("NIFTY", "NFO", "2024-11-30", "2024-11-01", FakeCalendar())


def test_unknown_series_is_refused(tables):
    with pytest.raises(ConfigError, match="series must be"):
        tables.expiries("NIFTY", "NFO", "2024-11-01", "2024-11-30", FakeCalendar(), series="daily")
