"""Declared corporate actions (PRD 06 sections 2, 5 and 7): how a declaration is built and
checked, and what it makes of a gap the split check finds (`splits` module table).

The series are those of `split_series`: daily NSE bars from 2024-07-01 to 2024-07-12 at
2,000.00 up to 2024-07-05, with the gap at the open of Monday 2024-07-08.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import at_ist
from split_series import BEFORE, CAL, EX_DATE, SESSIONS, daily, gap_at_ex_date, issue

from openalgo.backtest.data import CorporateAction, check_bars, load_bars
from openalgo.backtest.data.corporate_actions import declared
from openalgo.backtest.errors import ConfigError, DataQualityError

HALVED = gap_at_ex_date(1_000.0, 1_000.0)
FLAT = gap_at_ex_date(BEFORE, BEFORE)


def checked(prices, *actions, days=SESSIONS):
    return check_bars(daily(prices, days=days), calendar=CAL, corporate_actions=actions)


# --- building a declaration ----------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "kind", "factor"),
    [
        (CorporateAction.split("2024-07-08", 10, 2), "split", 5.0),
        (CorporateAction.split("2024-07-08", 10, 5), "split", 2.0),
        (CorporateAction.split("2024-07-08", 1, 10), "split", 0.1),
        (CorporateAction.bonus("2024-07-08", 1, 1), "bonus", 2.0),
        (CorporateAction.bonus("2024-07-08", 3, 2), "bonus", 2.5),
        (CorporateAction.bonus("2024-07-08", 1, 3), "bonus", 4 / 3),
        (CorporateAction.none("2024-07-08"), "none", 1.0),
    ],
    ids=["split 10 to 2", "split 10 to 5", "consolidation 1 to 10", "bonus 1:1", "bonus 3:2",
         "bonus 1:3", "none"],
)  # fmt: skip
def test_a_declaration_gives_its_factor_in_shares_after_per_share_before(action, kind, factor):
    assert (action.ex_date, action.kind, action.factor) == (EX_DATE, kind, factor)


@pytest.mark.parametrize(
    ("action", "words"),
    [
        (CorporateAction.split(EX_DATE, 10, 2), "a split (5 shares for every 1)"),
        (CorporateAction.split(EX_DATE, 1, 10), "a split (1 share for every 10)"),
        (CorporateAction.bonus(EX_DATE, 3, 2), "a bonus (5 shares for every 2)"),
        (CorporateAction.none(EX_DATE), "no corporate action"),
    ],
)
def test_a_declaration_describes_itself_in_whole_shares(action, words):
    assert action.describe() == words


@pytest.mark.parametrize(
    "build",
    [
        lambda: CorporateAction.split(EX_DATE, 10, 10),
        lambda: CorporateAction.split(EX_DATE, 0, 2),
        lambda: CorporateAction.split(EX_DATE, 10, float("nan")),
        lambda: CorporateAction.split(EX_DATE, "10", 2),
        lambda: CorporateAction.bonus(EX_DATE, 0, 1),
        lambda: CorporateAction.bonus(EX_DATE, 1, 1.5),
        lambda: CorporateAction.bonus(EX_DATE, True, 1),
        lambda: CorporateAction(EX_DATE, "dividend", 2.0),
        lambda: CorporateAction(EX_DATE, "none", 2.0),
        lambda: CorporateAction(EX_DATE, "bonus", 0.5),
        lambda: CorporateAction(EX_DATE, "split", float("inf")),
        lambda: CorporateAction("2024-13-01", "split", 2.0),
    ],
    ids=["equal face values", "zero face value", "NaN face value", "face value as text",
         "no bonus shares", "fractional held", "bool shares", "unknown kind",
         "none with a factor", "bonus reducing shares", "infinite factor", "bad date"],
)  # fmt: skip
def test_an_unusable_declaration_is_a_config_error(build):
    with pytest.raises(ConfigError):
        build()


@pytest.mark.parametrize(
    "actions",
    [
        CorporateAction.none(EX_DATE),
        "2024-07-08",
        [("2024-07-08", "split", 2.0)],
        [CorporateAction.bonus(EX_DATE, 1, 1), CorporateAction.bonus(EX_DATE, 1, 2)],
        [CorporateAction.none(EX_DATE), CorporateAction.split(EX_DATE, 10, 5)],
    ],
    ids=["one object, not a list", "a string", "a tuple, not a CorporateAction",
         "two bonuses on one date", "none and a split on one date"],
)  # fmt: skip
def test_an_unusable_list_of_declarations_is_a_config_error(actions):
    with pytest.raises(ConfigError):
        declared(actions)


def test_declarations_are_kept_in_ex_date_order():
    later, earlier = CorporateAction.none("2024-07-10"), CorporateAction.bonus(EX_DATE, 1, 1)
    assert declared([later, earlier]) == (earlier, later)


# --- what a declaration makes of a gap ------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [CorporateAction.bonus(EX_DATE, 1, 1), CorporateAction.split(EX_DATE, 10, 5)],
    ids=["bonus 1:1", "split 10 to 5"],
)
def test_a_declared_split_or_bonus_on_unadjusted_bars_is_a_warning_and_no_error(action):
    report = checked(HALVED, action)
    found = issue(report, "split_or_bonus_unadjusted")
    assert (found.severity, found.count, found.examples) == ("warning", 1, ("2024-07-08",))
    assert report.ok
    assert [(gap.status, gap.factor) for gap in report.price_gaps] == [("unadjusted", 2.0)]


def test_the_unadjusted_warning_says_what_is_not_adjusted_and_what_a_backtest_does():
    report = checked(HALVED, CorporateAction.bonus(EX_DATE, 1, 1))
    message = issue(report, "split_or_bonus_unadjusted").message
    assert message.startswith(
        "the bars are not adjusted for 1 declared corporate actions (2024-07-08: a bonus (2 "
        "shares for every 1); the previous close 2000.00 against the open 1000.00 is a ratio "
        "of 2.000). Adjusting for them is FR-MKT-11 (Phase 5)"
    )
    assert "a backtest that holds a position or a GTT across one stops" in message


def test_a_declared_split_whose_factor_does_not_match_the_gap_is_an_error():
    report = checked(HALVED, CorporateAction.split(EX_DATE, 10, 2))
    found = issue(report, "split_or_bonus")
    assert (found.severity, found.examples) == ("error", ("2024-07-08",))
    assert "2024-07-08: a split (5 shares for every 1) is declared, but the previous close" in (
        found.message
    )
    assert [gap.status for gap in report.price_gaps] == ["mismatch"]


def test_a_declared_bonus_on_bars_already_adjusted_for_it_is_info():
    report = checked(FLAT, CorporateAction.bonus(EX_DATE, 1, 1))
    found = issue(report, "split_or_bonus_adjusted")
    assert (found.severity, found.examples) == ("info", ("2024-07-08",))
    assert [(gap.status, gap.ratio) for gap in report.price_gaps] == [("adjusted", 1.0)]


def test_a_date_declared_without_corporate_action_reads_the_gap_as_a_genuine_move():
    report = checked(HALVED, CorporateAction.none(EX_DATE))
    found = issue(report, "price_move_acknowledged")
    assert (found.severity, found.examples) == ("info", ("2024-07-08",))
    assert report.ok
    assert [(gap.status, gap.factor) for gap in report.price_gaps] == [("acknowledged", 2.0)]


def test_a_date_declared_without_corporate_action_where_there_is_no_gap_reports_nothing():
    report = checked(FLAT, CorporateAction.none(EX_DATE))
    assert (report.issues, report.price_gaps) == ([], [])


def test_a_split_and_a_bonus_on_one_date_multiply():
    split, bonus = CorporateAction.split(EX_DATE, 2, 1), CorporateAction.bonus(EX_DATE, 1, 1)
    report = checked(gap_at_ex_date(500.0, 500.0), split, bonus)
    assert [(gap.status, gap.factor) for gap in report.price_gaps] == [("unadjusted", 4.0)]
    assert report.price_gaps[0].reading == (
        "a split (2 shares for every 1) and a bonus (2 shares for every 1)"
    )


def test_a_declaration_on_a_session_without_bars_applies_at_the_next_session_with_bars():
    days = [day for day in SESSIONS if day != EX_DATE]
    prices = [(BEFORE, BEFORE)] * 5 + [(1_000.0, 1_000.0)] * 4
    report = checked(prices, CorporateAction.bonus(EX_DATE, 1, 1), days=days)
    found = [(gap.ex_date, gap.status) for gap in report.price_gaps]
    assert found == [("2024-07-09", "unadjusted")]


def test_a_declaration_outside_the_bars_changes_nothing():
    on_the_first_session = CorporateAction.bonus("2024-07-01", 1, 1)
    after_the_last = CorporateAction.bonus("2024-08-01", 1, 1)
    report = checked(FLAT, on_the_first_session, after_the_last)
    assert (report.issues, report.price_gaps) == ([], [])


def test_a_declared_bonus_inside_the_price_band_is_read_against_its_own_factor():
    # A 1:4 bonus (factor 1.25) moves the price no further than a circuit: undetected,
    # but once declared the gap is read against it.
    report = checked(gap_at_ex_date(1_600.0, 1_600.0), CorporateAction.bonus(EX_DATE, 1, 4))
    assert [(gap.status, gap.factor) for gap in report.price_gaps] == [("unadjusted", 1.25)]


def test_none_and_a_split_that_fall_on_one_session_are_a_config_error():
    days = [day for day in SESSIONS if day not in (EX_DATE, dt.date(2024, 7, 9))]
    prices = [(BEFORE, BEFORE)] * 5 + [(1_000.0, 1_000.0)] * 3
    actions = (CorporateAction.none(EX_DATE), CorporateAction.split("2024-07-09", 10, 5))
    with pytest.raises(ConfigError, match="one says no corporate action"):
        checked(prices, *actions, days=days)


@pytest.mark.parametrize("exchange", ["NSE_INDEX", "NFO"])
def test_declarations_for_an_index_or_a_derivative_are_a_config_error(exchange):
    bars = daily(HALVED, exchange=exchange)
    with pytest.raises(ConfigError, match="corporate_actions apply to shares on NSE and BSE"):
        check_bars(bars, corporate_actions=[CorporateAction.bonus(EX_DATE, 1, 1)])


# --- load_bars ------------------------------------------------------------------------

LOAD_AFTER = at_ist(dt.date(2024, 8, 1), 18, 0)


def own_daily(prices):
    """The series as own data: daily rows stamped at 09:15 IST."""
    bars = daily(prices)
    stamps = np.array([at_ist(day, 9, 15) for day in SESSIONS], dtype="int64")
    index = pd.to_datetime(stamps, utc=True).tz_convert("Asia/Kolkata")
    return bars.drop(columns=["ts_open", "ts_close", "oi"]).set_index(index)


def load(prices, interval="D", **options):
    return load_bars(
        "SHARE", "NSE", interval, SESSIONS[0], SESSIONS[-1], data=own_daily(prices),
        calendar=CAL, now=LOAD_AFTER, base="D", **options,
    )  # fmt: skip


def test_a_load_stops_on_a_suspected_split_and_says_how_to_go_on():
    with pytest.raises(DataQualityError, match="1 suspected unadjusted split or bonus") as caught:
        load(HALVED)
    assert caught.value.report.price_gaps[0].ex_date == "2024-07-08"


def test_a_load_with_the_split_declared_returns_the_bars_with_the_warning():
    result = load(HALVED, corporate_actions=[CorporateAction.bonus(EX_DATE, 1, 1)])
    assert len(result.bars) == 10
    assert result.meta.quality.count("split_or_bonus_unadjusted") == 1


def test_weekly_bars_built_from_daily_bars_are_checked_on_the_daily_bars():
    with pytest.raises(DataQualityError, match="1 suspected unadjusted split or bonus"):
        load(HALVED, interval="W")


def test_a_load_refuses_declarations_for_an_index_before_reading_any_data():
    with pytest.raises(ConfigError, match="corporate_actions apply to shares"):
        load_bars(
            "NIFTY", "NSE_INDEX", "D", SESSIONS[0], SESSIONS[-1], data=object(),
            corporate_actions=[CorporateAction.bonus(EX_DATE, 1, 1)],
        )  # fmt: skip
