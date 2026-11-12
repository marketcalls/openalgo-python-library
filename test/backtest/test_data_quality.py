"""Data quality checks and their report (PRD 06 section 5, the Phase 0 part of
FR-DATA-10): each check, its severity and its count.

The reference series is one NSE day of 1-minute bars (375 bars from 09:15 IST).
"""

import dataclasses
import datetime as dt
import json

import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    at_ist,
    canonical_frame,
    daily_rows,
    days_between,
    historify_shape,
    ist_frame,
    normalise,
    truth_rows,
)

from openalgo.backtest.data import QualityReport, check_bars, load_bars
from openalgo.backtest.errors import DataQualityError

MON = dt.date(2024, 1, 1)
FRI = dt.date(2024, 1, 5)
CAL = FakeCalendar()
LONG_AFTER = at_ist(dt.date(2024, 3, 15), 18, 0)
DAY_ROWS = truth_rows(CAL, "NSE", [MON])


def _checked(rows, interval="1m", **options):
    """Normalise raw rows and run the quality checks on the result."""
    bars, normalisation = normalise(rows, interval=interval, calendar=CAL, **options)
    return check_bars(bars, calendar=CAL, normalisation=normalisation)


def _issue(report, check):
    found = [issue for issue in report.issues if issue.check == check]
    assert len(found) == 1, f"expected one {check} issue, got {[i.check for i in report.issues]}"
    return found[0]


def _with(rows, position, **changes):
    changed = list(rows)
    changed[position] = dict(rows[position], **changes)
    return changed


# --- a clean series -----------------------------------------------------------------


def test_clean_series_has_no_issues():
    report = _checked([dict(row, oi=100) for row in DAY_ROWS])
    assert report.ok
    assert report.issues == []
    assert report.rows == 375


def test_report_names_the_series_it_describes():
    report = _checked(DAY_ROWS)
    assert (report.symbol, report.exchange, report.interval) == ("X", "NSE", "1m")


# --- errors -------------------------------------------------------------------------


@pytest.mark.parametrize("bad_price", [0.0, -5.0], ids=["zero", "negative"])
def test_price_that_is_not_positive_is_an_error(bad_price):
    report = _checked(_with(DAY_ROWS, 7, low=bad_price))
    issue = _issue(report, "non_positive_price")
    assert (issue.severity, issue.count) == ("error", 1)
    assert issue.examples == ("2024-01-01 09:22 IST",)
    assert not report.ok


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize(
    "bad_price", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_price_that_is_not_finite_is_an_error(column, bad_price):
    report = _checked(_with(DAY_ROWS, 7, **{column: bad_price}))
    issue = _issue(report, "non_finite_price")
    assert (issue.severity, issue.count) == ("error", 1)
    assert issue.examples == ("2024-01-01 09:22 IST",)
    assert report.count("non_positive_price") == 0  # reported once, as non-finite


@pytest.mark.parametrize(
    "changes",
    [
        {"high": 100.0, "open": 101.0, "close": 100.5, "low": 99.0},  # high below the open
        {"high": 102.0, "open": 100.0, "close": 103.0, "low": 99.0},  # high below the close
        {"low": 101.0, "open": 100.0, "close": 102.0, "high": 103.0},  # low above the open
        {"low": 101.5, "open": 102.0, "close": 101.0, "high": 103.0},  # low above the close
    ],
    ids=["high<open", "high<close", "low>open", "low>close"],
)
def test_high_or_low_that_contradicts_open_and_close_is_an_error(changes):
    report = _checked(_with(DAY_ROWS, 3, **changes))
    issue = _issue(report, "ohlc_inconsistent")
    assert (issue.severity, issue.count) == ("error", 1)
    assert issue.examples == ("2024-01-01 09:18 IST",)


def test_row_with_an_unusable_price_is_not_reported_twice():
    report = _checked(_with(DAY_ROWS, 3, high=-1.0))  # negative, and below open and close
    assert report.count("non_positive_price") == 1
    assert report.count("ohlc_inconsistent") == 0


# --- warnings -----------------------------------------------------------------------


def test_duplicate_timestamps_are_a_warning_with_their_count():
    different = dict(DAY_ROWS[5], close=DAY_ROWS[5]["close"] + 0.25)
    report = _checked(DAY_ROWS + [DAY_ROWS[3], DAY_ROWS[4], different])
    issue = _issue(report, "duplicate_timestamps")
    assert (issue.severity, issue.count) == ("warning", 3)
    assert issue.message == (
        "3 duplicate timestamps: 3 removed on load (the first row of each timestamp was "
        "kept), 1 of them carried different values"
    )


def test_duplicates_still_in_a_frame_are_found_without_a_normalisation_report():
    bars = canonical_frame(DAY_ROWS + [DAY_ROWS[0]])
    bars.attrs.update(symbol="X", exchange="NSE", interval="1m")
    issue = _issue(check_bars(bars.sort_values("ts_open"), calendar=CAL), "duplicate_timestamps")
    assert issue.message == "1 duplicate timestamp: 1 still present"


def test_missing_intraday_bars_are_a_warning_with_their_count():
    gone = {at_ist(MON, 10, 0) // 10**9, at_ist(MON, 10, 1) // 10**9, at_ist(MON, 14, 30) // 10**9}
    report = _checked([row for row in DAY_ROWS if row["timestamp"] not in gone])
    issue = _issue(report, "missing_bars")
    assert (issue.severity, issue.count) == ("warning", 3)
    assert report.ok  # a warning does not fail the load


def test_session_day_without_any_bar_is_a_missing_session():
    week = truth_rows(CAL, "NSE", days_between(MON, FRI))
    wednesday = dt.date(2024, 1, 3)
    without_wednesday = [
        row for row in week
        if not at_ist(wednesday, 0) // 10**9 <= row["timestamp"] < at_ist(wednesday, 24) // 10**9
    ]  # fmt: skip
    report = _checked(without_wednesday)
    issue = _issue(report, "missing_sessions")
    assert (issue.severity, issue.count, issue.examples) == ("warning", 1, ("2024-01-03",))
    assert report.count("missing_bars") == 0  # the four days that have bars are complete


def test_missing_daily_bars_are_counted_against_the_calendar():
    days = [day for day in days_between(MON, dt.date(2024, 1, 31)) if day.day not in (10, 11)]
    issue = _issue(_checked(daily_rows(CAL, "NSE", days), interval="D"), "missing_sessions")
    assert (issue.severity, issue.count) == ("warning", 2)
    assert issue.examples == ("2024-01-10", "2024-01-11")
    assert issue.message.startswith("2 of 23 session days")


def test_bars_off_the_lattice_in_a_frame_checked_directly_are_a_warning():
    rows = _with(DAY_ROWS, 10, timestamp=DAY_ROWS[10]["timestamp"] + 30)  # 09:25:30
    bars = canonical_frame(rows)
    bars.attrs.update(symbol="X", exchange="NSE", interval="1m")
    issue = _issue(check_bars(bars, calendar=CAL), "off_lattice_bars")
    assert (issue.severity, issue.count) == ("warning", 1)


def test_late_first_bar_of_a_session_is_a_warning():
    without_first_ten = DAY_ROWS[10:] + truth_rows(CAL, "NSE", [dt.date(2024, 1, 2)])
    issue = _issue(_checked(without_first_ten), "late_session_start")
    assert (issue.severity, issue.count) == ("warning", 1)
    assert issue.examples == ("2024-01-01",)


def test_bars_dropped_after_the_close_are_a_warning():
    after_close = dict(DAY_ROWS[0], timestamp=at_ist(MON, 15, 40) // 10**9)
    issue = _issue(_checked(DAY_ROWS + [after_close]), "postclose_dropped")
    assert (issue.severity, issue.count) == ("warning", 1)


def test_rows_dropped_on_a_closed_day_are_a_warning_with_the_date():
    saturday_bar = dict(DAY_ROWS[0], timestamp=at_ist(dt.date(2024, 1, 6), 9, 15) // 10**9)
    issue = _issue(_checked(DAY_ROWS + [saturday_bar]), "closed_day_dropped")
    assert (issue.severity, issue.count, issue.examples) == ("warning", 1, ("2024-01-06",))


# --- info ---------------------------------------------------------------------------


def test_open_interest_of_zero_converted_to_nan_is_reported_as_info():
    issue = _issue(_checked([dict(row, oi=0) for row in DAY_ROWS]), "oi_zero_to_nan")
    assert (issue.severity, issue.count) == ("info", 375)


def test_dropped_preopen_bars_are_reported_as_info():
    rows = historify_shape(DAY_ROWS, CAL, "NSE", preopen_minutes=8)
    rows = [dict(row, oi=5) for row in rows]
    issue = _issue(_checked(rows), "preopen_dropped")
    assert (issue.severity, issue.count) == ("info", 8)


def test_dropped_partial_bar_is_reported_as_info():
    issue = _issue(_checked(DAY_ROWS, now_ns=at_ist(MON, 15, 29, 30)), "partial_dropped")
    assert (issue.severity, issue.count) == ("info", 1)


# --- checks that did not run --------------------------------------------------------


def test_report_lists_every_prd_check_this_phase_does_not_run():
    not_run = [text.split(":")[0] for text in _checked(DAY_ROWS).not_run]
    assert not_run == [
        "zero_volume_runs", "outlier_returns", "survivorship_bias", "adjustment_status",
    ]  # fmt: skip


def test_missing_bars_check_is_listed_as_not_run_for_weekly_and_longer_bars():
    epoch_day = (MON - dt.date(1970, 1, 1)).days * 86_400
    report = _checked([dict(DAY_ROWS[0], timestamp=epoch_day)], interval="W")
    assert "missing_bars: not checked for interval 'W' (only intraday and D)" in report.not_run


# --- the report object --------------------------------------------------------------


def test_summary_lists_errors_before_warnings_before_info():
    rows = [dict(row, oi=0) for row in _with(DAY_ROWS, 7, low=-1.0)][:-1] + [DAY_ROWS[3]]
    lines = _checked(rows).summary().splitlines()
    kinds = [line.split(":")[0] for line in lines]
    assert kinds[:4] == ["error", "warning", "warning", "info"]
    assert set(kinds[4:]) == {"not run"}


def test_report_is_json_serialisable():
    report = _checked(_with(DAY_ROWS, 7, low=-1.0))
    as_dict = json.loads(json.dumps(dataclasses.asdict(report)))
    assert as_dict["issues"][0] == {
        "check": "non_positive_price",
        "severity": "error",
        "count": 1,
        "message": "1 row has a price that is zero or negative",
        "examples": ["2024-01-01 09:22 IST"],
    }


def test_unknown_severity_is_rejected():
    with pytest.raises(ValueError, match="unknown severity"):
        QualityReport().add("x", "fatal", 1, "message")


# --- load_bars and error severity ---------------------------------------------------


def _load(frame, **options):
    return load_bars(
        "X", "NSE", "1m", MON, MON, data=frame, calendar=CAL, now=LONG_AFTER, **options
    )


def test_load_stops_on_an_error_severity_issue_and_carries_the_report():
    frame = ist_frame(_with(DAY_ROWS, 7, low=-1.0))
    with pytest.raises(DataQualityError, match="1 row has a price that is zero") as caught:
        _load(frame)
    assert caught.value.report.count("non_positive_price") == 1


def test_load_with_strict_false_returns_the_bars_and_the_errors():
    result = _load(ist_frame(_with(DAY_ROWS, 7, low=-1.0)), strict=False)
    assert len(result.bars) == 375
    assert [issue.check for issue in result.meta.quality.errors] == ["non_positive_price"]


def test_bars_of_a_running_session_that_cannot_have_closed_yet_are_not_missing():
    thirty_seconds_into_the_1000_bar = at_ist(MON, 10, 0, 30)
    result = load_bars(
        "X", "NSE", "1m", MON, MON, data=ist_frame(DAY_ROWS), calendar=CAL,
        now=thirty_seconds_into_the_1000_bar,
    )  # fmt: skip
    assert len(result.bars) == 45
    assert result.meta.quality.count("missing_bars") == 0


def test_load_does_not_stop_on_warnings():
    result = _load(ist_frame(DAY_ROWS[:100] + DAY_ROWS[110:]))
    assert result.meta.quality.count("missing_bars") == 10
    assert result.meta.quality.ok
