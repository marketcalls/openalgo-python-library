"""The preflight's check of the package calendar against OpenAlgo's market holidays
(`_live/_preflight_calendar.py`; 18-multi-strategy 12.2, unit M11).

The day is Monday 2026-11-16, 10:15:20 IST. From it to the year's end the package calendar
closes NSE and NFO on Tuesday 2026-11-24 (Guru Nanak's birthday), when MCX keeps its
evening session, and every exchange on Friday 2026-12-25 (Christmas); OpenAlgo's 2026
table (`fake_openalgo.holidays`) holds the same two rows.
"""

from __future__ import annotations

import pytest
from fake_openalgo.holidays import HOLIDAYS_2026, holidays_answer
from preflight_fakes import NOW_NS, UNREACHABLE, FakeOpenAlgo, make_plan, sdk_http_error

from openalgo.backtest._live._preflight_calendar import HOLIDAYS, holiday_check
from openalgo.backtest._live._preflight_plan import WARNING
from openalgo.backtest._live.preflight import check_preflight
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import PreflightError

CALENDAR = default_calendar()


def without(date: str) -> dict:
    """OpenAlgo's 2026 answer with the row of `date` left out."""
    rows = tuple(row for row in HOLIDAYS_2026 if row["date"] != date)
    return holidays_answer(2026, rows)


def closing_today(exchanges: list[str]) -> dict:
    """OpenAlgo's 2026 answer with Monday 2026-11-16 closed for `exchanges`."""
    today = {
        "date": "2026-11-16",
        "description": "A closure only OpenAlgo has",
        "holiday_type": "TRADING_HOLIDAY",
        "closed_exchanges": exchanges,
        "open_exchanges": [],
    }
    return holidays_answer(2026, (*HOLIDAYS_2026, today))


def check(answer: dict | None, exchanges=("NSE",)):
    client = FakeOpenAlgo(holidays=answer)
    return holiday_check(client, exchanges, NOW_NS, CALENDAR), client


def test_a_table_that_agrees_passes_with_no_days_named():
    (result, later), _ = check(None, ("NSE", "NFO", "MCX"))
    assert result.ok and not result.message.startswith(WARNING)
    assert later == ()


def test_the_check_reads_the_current_years_holidays_once():
    _, client = check(None)
    assert client.calls == [("holidays", 2026)]


def test_a_later_day_that_disagrees_is_a_warning_naming_the_day():
    (result, later), _ = check(without("2026-12-25"))
    assert result.ok and result.message.startswith(WARNING)
    assert later == (
        "NSE on 2026-12-25: the package calendar is closed, OpenAlgo's holidays opens 09:15 "
        "to 15:30 IST",
    )


def test_a_disagreement_about_today_refuses_the_start():
    (result, later), _ = check(closing_today(["NSE", "BSE", "NFO", "BFO"]))
    assert not result.ok
    assert "disagree about today (NSE on 2026-11-16: the package calendar opens" in result.message
    assert "OpenAlgo's holidays is closed)" in result.message
    assert later == ()


def test_only_the_sessions_exchanges_are_compared():
    # OpenAlgo's table leaves MCX open on Christmas; an NSE session does not trade MCX.
    rows = tuple(
        row | {"closed_exchanges": [e for e in row["closed_exchanges"] if e != "MCX"]}
        if row["date"] == "2026-12-25"
        else row
        for row in HOLIDAYS_2026
    )
    (nse, _), _ = check(holidays_answer(2026, rows), ("NSE",))
    (mcx, later), _ = check(holidays_answer(2026, rows), ("MCX",))
    assert nse.ok and not nse.message.startswith(WARNING)
    assert mcx.message.startswith(WARNING) and later[0].startswith("MCX on 2026-12-25")


def test_mcxs_evening_on_an_equity_holiday_agrees():
    # On 2026-11-24 OpenAlgo opens MCX 17:00 to 23:55, which the calendar takes at MCX's
    # regular close of the date.
    (result, later), _ = check(None, ("MCX",))
    assert result.ok and later == ()


@pytest.mark.parametrize(
    "answer",
    [UNREACHABLE, sdk_http_error(404, {"message": "Not Found"})],
    ids=["OpenAlgo unreachable", "a server without the endpoint"],
)
def test_holidays_that_cannot_be_read_are_a_warning_not_a_refusal(answer):
    (result, later), _ = check(answer)
    assert result.ok and result.message.startswith(WARNING)
    assert "the package calendar was not checked" in result.message
    assert later == ()


# ----- through the preflight -----------------------------------------------------------------


def test_the_report_keeps_the_later_days_that_disagree():
    client = FakeOpenAlgo(holidays=without("2026-11-24"))
    report = check_preflight(client, make_plan(), now=lambda: NOW_NS).report()
    assert report.calendar_disagrees == (
        "NSE on 2026-11-24: the package calendar is closed, OpenAlgo's holidays opens 09:15 "
        "to 15:30 IST",
    )
    assert any("disagree on later days" in note for note in report.warnings)


def test_a_disagreement_about_today_fails_the_preflight():
    client = FakeOpenAlgo(holidays=closing_today(["NSE"]))
    outcome = check_preflight(client, make_plan(), now=lambda: NOW_NS)
    assert outcome.check(HOLIDAYS).ok is False
    with pytest.raises(PreflightError, match="disagree about today"):
        outcome.report()
