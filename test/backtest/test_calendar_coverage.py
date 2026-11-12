"""Coverage of calendar answers, the annualisation factor, and copying another calendar."""

import datetime as dt
import warnings

import pytest
from calendar_fixtures.builders import at_ist

from openalgo.backtest.calendar import Calendar, CalendarWarning
from openalgo.backtest.errors import CalendarError

MONDAY = dt.date(2025, 6, 2)
SUNDAY = dt.date(2025, 6, 8)
FOUR_WEEKS = ("2025-06-02", "2025-06-29")  # Monday to Sunday, 28 calendar days


@pytest.fixture
def cal() -> Calendar:
    return Calendar(builtin=False)


def test_a_date_without_data_is_template(cal):
    assert cal.coverage("NSE", MONDAY) == "template"


def test_a_year_can_be_asked_for_as_an_int(cal):
    assert cal.coverage("NSE", 2025) == "template"


def test_a_rule_reports_its_own_coverage(cal):
    cal.add_closure("NSE", MONDAY, verified=False)
    assert cal.coverage("NSE", MONDAY) == "unverified"


def test_sessions_per_year_scales_the_count_to_365_25_days(cal):
    cal.add_closure("NSE", "2025-06-03")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalendarWarning)
        result = cal.sessions_per_year("NSE", *FOUR_WEEKS)
    assert (result.sessions, result.calendar_days) == (19, 28)
    assert result.value == pytest.approx(19 * 365.25 / 28)


def test_sessions_per_year_uses_the_union_of_the_exchanges(cal):
    cal.add_closure("NSE", "2025-06-03")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalendarWarning)
        result = cal.sessions_per_year(["NSE", "MCX"], *FOUR_WEEKS)
    assert result.sessions == 20


def test_sessions_per_year_warns_and_names_template_years(cal):
    with pytest.warns(CalendarWarning, match="NSE 2025"):
        result = cal.sessions_per_year("NSE", *FOUR_WEEKS)
    assert (result.coverage, result.template_only) == ("template", [("NSE", 2025)])


def test_sessions_per_year_is_silent_on_a_covered_window():
    cal = Calendar()  # the built-in lists cover NSE in 2025
    with warnings.catch_warnings():
        warnings.simplefilter("error", CalendarWarning)
        result = cal.sessions_per_year("NSE", "2025-01-01", "2025-12-31")
    assert result.coverage == "sourced"


def test_sessions_per_year_refuses_a_window_shorter_than_four_weeks(cal):
    with pytest.raises(CalendarError, match="too short to annualise"):
        cal.sessions_per_year("NSE", "2025-06-02", "2025-06-28")


class OneHourCalendar:
    """A `SessionCalendar` that is not the engine's: Mondays trade 10:00 to 11:00 IST."""

    def __init__(self, coverage: str | None = None) -> None:
        if coverage is not None:
            self.coverage = lambda exchange, day: coverage

    def session(self, exchange: str, day: dt.date) -> tuple[int, int] | None:
        return (at_ist(day, 10, 0), at_ist(day, 11, 0)) if day.weekday() == 0 else None


def test_from_sessions_copies_another_calendar_date_by_date():
    copied = Calendar.from_sessions(OneHourCalendar(), "NSE", MONDAY, SUNDAY)
    assert copied.session("NSE", MONDAY) == (at_ist(MONDAY, 10, 0), at_ist(MONDAY, 11, 0))
    assert len(copied.sessions("NSE", MONDAY, SUNDAY)) == 1  # the other days are closed


def test_from_sessions_leaves_dates_outside_the_range_on_regular_hours():
    copied = Calendar.from_sessions(OneHourCalendar(), "NSE", MONDAY, MONDAY)
    next_monday = MONDAY + dt.timedelta(days=7)
    assert copied.session("NSE", next_monday) == (
        at_ist(next_monday, 9, 15),
        at_ist(next_monday, 15, 30),
    )


def test_a_copy_without_source_coverage_is_never_sourced():
    copied = Calendar.from_sessions(OneHourCalendar(), "NSE", MONDAY, SUNDAY)
    assert {copied.coverage("NSE", MONDAY + dt.timedelta(days=k)) for k in range(7)} == {
        "unverified"
    }
    assert copied.session_info("NSE", MONDAY).hours_unverified


@pytest.mark.parametrize("level", ["unverified", "sourced"])
def test_a_copy_carries_the_coverage_of_its_source(level):
    copied = Calendar.from_sessions(OneHourCalendar(level), "NSE", MONDAY, SUNDAY)
    assert copied.coverage("NSE", MONDAY) == level


def test_a_template_date_equal_to_the_weekday_rule_stays_template():
    class Regular:
        def session(self, exchange, day):
            return (at_ist(day, 9, 15), at_ist(day, 15, 30)) if day.weekday() < 5 else None

        def coverage(self, exchange, day):
            return "template"

    copied = Calendar.from_sessions(Regular(), "NSE", MONDAY, SUNDAY)
    assert copied.coverage("NSE", MONDAY) == "template"


def test_an_unknown_coverage_from_the_source_is_a_calendar_error():
    with pytest.raises(CalendarError, match="coverage must be one of"):
        Calendar.from_sessions(OneHourCalendar("verified"), "NSE", MONDAY, MONDAY)


def test_from_sessions_rejects_a_session_outside_its_date():
    class Overnight:
        def session(self, exchange: str, day: dt.date) -> tuple[int, int]:
            return at_ist(day, 22, 0), at_ist(day + dt.timedelta(days=1), 2, 0)

    with pytest.raises(CalendarError, match="must lie inside its IST date"):
        Calendar.from_sessions(Overnight(), "MCX", MONDAY, MONDAY)
