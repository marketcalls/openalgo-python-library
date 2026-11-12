"""Split and bonus detection from the bars (FR-DATA-10, FR-MKT-10; PRD 06 section 5):
which session gaps are reported, at what severity, and with what evidence.

The series are those of `split_series`: daily NSE bars of one share from 2024-07-01 to
2024-07-12, at 2,000.00 up to 2024-07-05 unless a test says otherwise, with the gap at
the open of Monday 2024-07-08.
"""

import dataclasses
import datetime as dt
import json

import numpy as np
import pytest
from data_fixtures.fakes import NS, at_ist
from split_series import BEFORE, CAL, EX_DATE, SESSIONS, daily, frame, gap_at_ex_date, issue

from openalgo.backtest.data import check_bars
from openalgo.backtest.data.splits import FACTOR_TOLERANCE, USUAL_FACTORS, WIDEST_PRICE_BAND

MINUTE_NS = 60 * NS


def checked(prices, exchange="NSE", interval="D", calendar=CAL):
    return check_bars(daily(prices, exchange, interval), calendar=calendar)


# --- what is reported ---------------------------------------------------------------


def test_a_halving_overnight_that_holds_is_a_suspected_split_or_bonus_at_error_severity():
    report = checked(gap_at_ex_date(1_001.50, 1_010.00))
    found = issue(report, "split_or_bonus")
    assert (found.severity, found.count, found.examples) == ("error", 1, ("2024-07-08",))
    assert not report.ok


def test_the_suspected_gap_carries_its_date_ratio_and_prices():
    gap = checked(gap_at_ex_date(1_001.50, 1_010.00)).price_gaps[0]
    assert (gap.ex_date, gap.status, gap.factor) == ("2024-07-08", "suspected", 2.0)
    assert (gap.previous_close, gap.open, gap.session_close) == (2_000.0, 1_001.50, 1_010.0)
    assert gap.ratio == pytest.approx(2_000.0 / 1_001.50)
    assert gap.ex_open_ns == at_ist(EX_DATE, 9, 15)


def test_the_error_message_names_the_date_the_ratio_the_reading_and_the_fix():
    message = issue(checked(gap_at_ex_date(1_001.50, 1_010.00)), "split_or_bonus").message
    assert "2024-07-08: the previous close 2000.00 against the open 1001.50 is a ratio" in message
    assert "of 1.997, near 2 shares for every 1 (a 1:1 bonus, or a split from face" in message
    assert "the session closed at 1010.00, keeping the new level" in message
    assert "CorporateAction.split(ex_date, from_face_value, to_face_value)" in message
    assert "CorporateAction.none(ex_date) when the move is genuine" in message


# Previous close 3,000.00 (consolidations 30.00) against an ex-date open and close at
# the price divided by the factor; the factor and its reading are the table's.
@pytest.mark.parametrize(
    ("before", "ex_price", "factor", "reading"),
    [
        (3_000.0, 2_250.0, 4 / 3, "a 1:3 bonus"),
        (3_000.0, 2_000.0, 3 / 2, "a 1:2 bonus"),
        (3_000.0, 1_800.0, 5 / 3, "a 2:3 bonus"),
        (3_000.0, 1_500.0, 2.0, "a 1:1 bonus, or a split from face value 10 to 5 or 2 to 1"),
        (3_000.0, 1_200.0, 5 / 2, "a 3:2 bonus, or a split from face value 5 to 2"),
        (3_000.0, 1_000.0, 3.0, "a 2:1 bonus"),
        (3_000.0, 750.0, 4.0, "a 3:1 bonus"),
        (3_000.0, 600.0, 5.0, "a 4:1 bonus, or a split from face value 10 to 2 or 5 to 1"),
        (3_000.0, 300.0, 10.0, "a split from face value 10 to 1"),
        (30.0, 60.0, 1 / 2, "a consolidation from face value 5 to 10 or 1 to 2"),
        (30.0, 75.0, 2 / 5, "a consolidation from face value 2 to 5"),
        (30.0, 150.0, 1 / 5, "a consolidation from face value 2 to 10 or 1 to 5"),
        (30.0, 300.0, 1 / 10, "a consolidation from face value 1 to 10"),
    ],
)
def test_every_usual_factor_is_recognised_with_its_reading(before, ex_price, factor, reading):
    gaps = checked(gap_at_ex_date(ex_price, ex_price, before=before)).price_gaps
    assert [(g.status, g.factor, g.reading) for g in gaps] == [("suspected", factor, reading)]


@pytest.mark.parametrize(
    ("previous_close", "reported"),
    [(2_080.0, True), (1_920.0, True), (2_120.0, False), (1_880.0, False)],
    ids=["4% above 2", "4% below 2", "6% above 2", "6% below 2"],
)
def test_a_gap_is_near_a_factor_within_five_percent_of_it(previous_close, reported):
    report = checked(gap_at_ex_date(1_000.0, 1_000.0, before=previous_close))
    assert report.count("split_or_bonus") == (1 if reported else 0)


def test_a_gap_on_intraday_bars_is_read_from_the_last_and_first_bars_of_two_sessions():
    stamps = [
        stamp
        for day in (dt.date(2024, 7, 5), EX_DATE)
        for stamp in range(*CAL.session("NSE", day), MINUTE_NS)
    ]
    prices = np.where(np.array(stamps) < at_ist(EX_DATE, 0), 2_000.0, 1_000.0)
    bars = frame(stamps, MINUTE_NS, prices, prices, "NSE", "1m")
    gaps = check_bars(bars, calendar=CAL).price_gaps
    found = [(gap.ex_date, gap.ex_open_ns, gap.factor) for gap in gaps]
    assert found == [("2024-07-08", at_ist(EX_DATE, 9, 15), 2.0)]


def test_bars_out_of_time_order_are_read_in_time_order():
    reversed_bars = daily(gap_at_ex_date(1_000.0, 1_000.0)).iloc[::-1]
    gaps = check_bars(reversed_bars, calendar=CAL).price_gaps
    assert [(gap.ex_date, gap.ratio) for gap in gaps] == [("2024-07-08", 2.0)]


def test_a_gap_across_a_session_without_bars_is_dated_at_the_next_session_with_bars():
    days = [day for day in SESSIONS if day != EX_DATE]
    prices = [(BEFORE, BEFORE)] * 5 + [(1_000.0, 1_000.0)] * 4
    report = check_bars(daily(prices, days=days), calendar=CAL)
    assert [gap.ex_date for gap in report.price_gaps] == ["2024-07-09"]


# --- what is not reported -----------------------------------------------------------


def test_the_smallest_usual_factor_lies_beyond_the_widest_daily_price_band():
    smallest_split = min(factor for factor, _ in USUAL_FACTORS if factor > 1)
    largest_consolidation = max(factor for factor, _ in USUAL_FACTORS if factor < 1)
    lower_circuit_ratio = 1 / (1 - WIDEST_PRICE_BAND)  # previous close / open
    upper_circuit_ratio = 1 / (1 + WIDEST_PRICE_BAND)
    assert smallest_split * (1 - FACTOR_TOLERANCE) > lower_circuit_ratio
    assert largest_consolidation * (1 + FACTOR_TOLERANCE) < upper_circuit_ratio


@pytest.mark.parametrize(
    "prices",
    [
        gap_at_ex_date(1_600.0, 1_600.0),
        gap_at_ex_date(2_400.0, 2_400.0),
        [(BEFORE, BEFORE)] * 5 + [(1_600.0, 1_600.0)] + [(1_280.0, 1_280.0)] * 4,
    ],
    ids=["opens at the 20% lower circuit", "opens at the 20% upper circuit", "two in a row"],
)
def test_a_circuit_limit_day_is_not_a_split(prices):
    report = checked(prices)
    assert (report.issues, report.price_gaps) == ([], [])


@pytest.mark.parametrize(
    "prices",
    [
        gap_at_ex_date(1_700.0, 1_690.0),
        gap_at_ex_date(1_990.0, 1_000.0),
        gap_at_ex_date(1_000.0, 1_995.0),
    ],
    ids=["a 15% gap that holds", "an ordinary open, then the price halves", "a bad print"],
)
def test_an_ordinary_large_move_is_not_a_split(prices):
    report = checked(prices)
    assert (report.issues, report.price_gaps) == ([], [])


@pytest.mark.parametrize("exchange", ["NSE_INDEX", "BSE_INDEX", "NFO", "MCX"])
def test_an_index_or_a_derivative_is_not_checked_and_the_report_says_so(exchange):
    report = checked(gap_at_ex_date(1_000.0, 1_000.0), exchange=exchange, calendar=None)
    assert (report.count("split_or_bonus"), report.price_gaps) == (0, [])
    assert [text for text in report.not_run if text.startswith("split_or_bonus")] == [
        f"split_or_bonus: not checked on {exchange}: shares on NSE and BSE split, an index "
        "does not, and adjusting derivative contracts at a corporate action is FR-MKT-14 "
        "(Phase 6)"
    ]


def test_weekly_bars_are_not_checked_and_the_report_says_so():
    report = checked(gap_at_ex_date(1_000.0, 1_000.0), interval="W")
    assert (
        "split_or_bonus: not checked for interval 'W': a split inside a weekly or longer bar "
        "cannot be seen (load_bars checks the bars it builds them from)"
    ) in report.not_run


def test_a_gap_next_to_a_price_that_is_not_usable_is_left_to_the_price_checks():
    prices = gap_at_ex_date(1_000.0, 1_000.0)
    prices[4] = (BEFORE, float("nan"))
    report = checked(prices)
    assert (report.count("split_or_bonus"), report.count("non_finite_price")) == (0, 1)


def test_the_report_with_its_gaps_is_json_serialisable():
    report = checked(gap_at_ex_date(1_000.0, 1_000.0))
    gap = json.loads(json.dumps(dataclasses.asdict(report)))["price_gaps"][0]
    assert (gap["ex_date"], gap["status"], gap["factor"]) == ("2024-07-08", "suspected", 2.0)
