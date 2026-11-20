"""The daily summary of a live session: its figures and its plain text, in the style of
`Result.summary()` (08-live-trading section 8). The expected texts are written by hand."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import FrozenInstanceError, replace

import pytest

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.alerts import SessionSummary, summary_text
from openalgo.backtest.errors import ConfigError

A_GOOD_DAY = SessionSummary(
    day=dt.date(2026, 11, 16),
    mode=Mode.SANDBOX,
    broker="zerodha",
    strategies=("ema_cross", "orb"),
    closed_trades=4,
    gross_pnl=1250.0,
    charges=182.4,
    slippage=35.0,
    rejected_orders=1,
    unprotected_seconds=0.0,
    reconcile_mismatches=0,
    undelivered_alerts=0,
    flat=True,
)

A_BAD_DAY = SessionSummary(
    day=dt.date(2026, 11, 17),
    mode=Mode.LIVE,
    broker="angel",
    strategies=("orb",),
    closed_trades=3,
    gross_pnl=-1234.5,
    charges=210.75,
    slippage=-12.5,
    rejected_orders=2,
    unprotected_seconds=12.4,
    reconcile_mismatches=1,
    undelivered_alerts=2,
    flat=False,
)


def line(summary: SessionSummary, label: str) -> str:
    """The value on the summary's line of `label`."""
    for text in summary_text(summary).splitlines()[1:]:
        name, _, value = text.partition(" : ")
        if name.strip() == label:
            return value
    raise AssertionError(f"no {label!r} line in the summary")


def test_a_sandbox_day_reads_as_a_title_and_one_line_per_figure():
    assert summary_text(A_GOOD_DAY) == (
        "Session of 2026-11-16 (IST): SANDBOX mode, broker zerodha\n"
        "Strategies  : ema_cross, orb\n"
        "Trades      : 4 closed\n"
        "Gross P&L   : Rs 1,250.00\n"
        "Charges     : Rs 182.40 (modelled)\n"
        "Net P&L     : Rs 1,067.60\n"
        "Slippage    : Rs 35.00 worse than the backtest\n"
        "Rejected    : 1 order\n"
        "Unprotected : 0 s without a stop at the broker\n"
        "Reconcile   : 0 mismatches\n"
        "End state   : flat"
    )


def test_a_live_day_with_a_loss_that_ended_open_reads_so():
    assert summary_text(A_BAD_DAY) == (
        "Session of 2026-11-17 (IST): LIVE mode, broker angel\n"
        "Strategies  : orb\n"
        "Trades      : 3 closed\n"
        "Gross P&L   : Rs -1,234.50\n"
        "Charges     : Rs 210.75 (modelled)\n"
        "Net P&L     : Rs -1,445.25\n"
        "Slippage    : Rs 12.50 better than the backtest\n"
        "Rejected    : 2 orders\n"
        "Unprotected : 13 s without a stop at the broker\n"
        "Reconcile   : 1 mismatch\n"
        "Alerts      : 2 not delivered\n"
        "End state   : not flat: a position is still open"
    )


def test_the_daily_limits_set_follow_the_strategies_line():
    limited = replace(
        A_GOOD_DAY,
        limits=("ema_cross at most 3 trades a day", "account trade cap 4 trades a day"),
    )
    lines = summary_text(limited).splitlines()
    assert lines[1:3] == [
        "Strategies  : ema_cross, orb",
        "Limits      : ema_cross at most 3 trades a day; account trade cap 4 trades a day",
    ]
    assert "Limits" not in summary_text(A_GOOD_DAY)


def test_the_alerts_line_is_left_out_when_every_alert_reached_openalgo():
    assert "Alerts" not in summary_text(A_GOOD_DAY)


def test_one_alert_not_delivered_is_shown():
    assert line(replace(A_GOOD_DAY, undelivered_alerts=1), "Alerts") == "1 not delivered"


def test_the_net_pnl_is_the_gross_pnl_less_the_modelled_charges():
    assert A_BAD_DAY.net_pnl == pytest.approx(-1445.25)


def test_rupees_use_indian_digit_grouping():
    assert line(replace(A_GOOD_DAY, gross_pnl=1234567.891), "Gross P&L") == "Rs 12,34,567.89"


def test_slippage_with_no_fill_compared_reads_not_compared():
    assert line(replace(A_GOOD_DAY, slippage=None), "Slippage") == "not compared"


@pytest.mark.parametrize(
    ("count", "shown"),
    [
        (1, "not compared (1 MARKET fill: no bar of its decision)"),
        (2, "not compared (2 MARKET fills: no bar of their decisions)"),
    ],
    ids=["one", "two"],
)
def test_slippage_with_every_market_fill_left_out_reads_not_compared_with_their_count(count, shown):
    day = replace(A_GOOD_DAY, slippage=None, slippage_not_compared=count)
    assert line(day, "Slippage") == shown


@pytest.mark.parametrize(
    ("count", "shown"),
    [
        (
            1,
            "Rs 35.00 worse than the backtest (1 MARKET fill not compared: no bar of its decision)",
        ),
        (
            2,
            "Rs 35.00 worse than the backtest (2 MARKET fills not compared: no bar of their "
            "decisions)",
        ),
    ],
    ids=["one", "two"],
)
def test_the_market_fills_the_slippage_leaves_out_are_counted_on_its_line(count, shown):
    assert line(replace(A_GOOD_DAY, slippage_not_compared=count), "Slippage") == shown


@pytest.mark.parametrize(
    "slippage", [0.0, 0.004, -0.004], ids=["zero", "under half a paisa", "under half a paisa back"]
)
def test_slippage_that_rounds_to_no_paisa_reads_as_none(slippage):
    assert line(replace(A_GOOD_DAY, slippage=slippage), "Slippage") == "none against the backtest"


def test_a_session_with_no_strategy_names_says_none():
    assert line(replace(A_GOOD_DAY, strategies=()), "Strategies") == "none"


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [(0.0, "0 s"), (0.4, "1 s"), (7.6, "8 s"), (12.0, "12 s")],
    ids=["none", "under a second", "a part second", "whole seconds"],
)
def test_the_seconds_unprotected_round_up_so_any_exposure_shows(seconds, shown):
    assert line(replace(A_GOOD_DAY, unprotected_seconds=seconds), "Unprotected") == (
        f"{shown} without a stop at the broker"
    )


@pytest.mark.parametrize("field", ["gross_pnl", "charges", "slippage", "unprotected_seconds"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_figure_that_is_not_a_finite_number_is_refused(field, value):
    with pytest.raises(ConfigError, match=f"the session summary's {field} must be a finite"):
        replace(A_GOOD_DAY, **{field: value})


@pytest.mark.parametrize("field", ["gross_pnl", "charges", "unprotected_seconds"])
def test_only_the_slippage_may_be_missing(field):
    with pytest.raises(ConfigError, match=f"the session summary's {field} must be a finite"):
        replace(A_GOOD_DAY, **{field: None})


def test_negative_seconds_unprotected_are_refused():
    with pytest.raises(ConfigError, match="unprotected_seconds cannot be negative"):
        replace(A_GOOD_DAY, unprotected_seconds=-3.5)


@pytest.mark.parametrize(
    "field",
    [
        "closed_trades",
        "rejected_orders",
        "reconcile_mismatches",
        "undelivered_alerts",
        "slippage_not_compared",
    ],
)
@pytest.mark.parametrize("value", [-1, 1.5, None], ids=["negative", "a fraction", "none"])
def test_a_count_that_is_not_a_whole_number_of_at_least_zero_is_refused(field, value):
    with pytest.raises(ConfigError, match=f"the session summary's {field} must be a whole"):
        replace(A_GOOD_DAY, **{field: value})


def test_the_summary_is_plain_ascii_text():
    assert summary_text(A_GOOD_DAY).isascii()
    assert summary_text(A_BAD_DAY).isascii()


def test_a_summary_cannot_be_changed():
    with pytest.raises(FrozenInstanceError):
        A_GOOD_DAY.closed_trades = 5


# ----- a difference first seen at the close (16-session-design 5.4 "As built", round 7) ---

#: How the close names a leg whose books differed at its last read.
SHORT_4 = "StopHold SBIN NSE -4 units in OpenAlgo's books, the engine's 0"
A_SHORT_CLOSE = replace(A_BAD_DAY, reconcile_mismatches=2, differing=(SHORT_4,))


def test_the_reconcile_line_counts_and_names_the_legs_that_differ_at_the_close():
    assert line(A_SHORT_CLOSE, "Reconcile") == f"2 mismatches (at the close: {SHORT_4})"


def test_the_end_state_names_the_legs_that_differ_at_the_close():
    assert line(A_SHORT_CLOSE, "End state") == (
        f"not flat: a position is still open (at the close: {SHORT_4})"
    )


def test_several_legs_that_differ_at_the_close_are_named_in_order():
    other = "Orb INFY NSE 0 units in OpenAlgo's books, the engine's 5"
    summary = replace(A_SHORT_CLOSE, differing=(SHORT_4, other))
    assert line(summary, "Reconcile") == f"2 mismatches (at the close: {SHORT_4}; {other})"


#: How the close names a leg whose OpenAlgo books show it flat (round 8).
FLAT_AT_OPENALGO = "StopHold SBIN NSE no position in OpenAlgo's books, the engine's 10"


def test_a_flat_end_state_names_a_leg_openalgos_books_show_flat_at_the_close():
    # The engine held 10 and its stop works on at the broker: no position is open, but the
    # leg is named where the End state says flat.
    summary = replace(A_GOOD_DAY, reconcile_mismatches=1, differing=(FLAT_AT_OPENALGO,))
    assert line(summary, "End state") == f"flat (at the close: {FLAT_AT_OPENALGO})"
    assert line(summary, "Reconcile") == f"1 mismatch (at the close: {FLAT_AT_OPENALGO})"


def test_a_leg_that_differs_at_the_close_counts_among_the_mismatches():
    with pytest.raises(ConfigError, match="at least one mismatch for each leg"):
        replace(A_SHORT_CLOSE, reconcile_mismatches=0)


@pytest.mark.parametrize("words", ["", 5], ids=["empty", "not text"])
def test_a_leg_that_differs_is_named_in_words(words):
    with pytest.raises(ConfigError, match="names each leg in words"):
        replace(A_SHORT_CLOSE, differing=(words,))


#: How the close names a leg no read judged after a fill of its own orders (round 9).
NOT_CONFIRMED_0 = (
    "StopHold SBIN NSE the engine's 0 units, not confirmed, since a fill of its own orders "
    "landed during a read of the books and no later read judged the position"
)


def test_a_flat_end_state_names_a_leg_no_read_judged_after_a_fill_and_counts_no_mismatch():
    summary = replace(A_GOOD_DAY, unconfirmed=(NOT_CONFIRMED_0,))
    assert line(summary, "End state") == f"flat (at the close: {NOT_CONFIRMED_0})"
    assert line(summary, "Reconcile") == f"0 mismatches (at the close: {NOT_CONFIRMED_0})"


def test_a_leg_not_confirmed_is_named_after_the_legs_that_differ():
    summary = replace(A_SHORT_CLOSE, unconfirmed=(NOT_CONFIRMED_0,))
    assert line(summary, "Reconcile") == (
        f"2 mismatches (at the close: {SHORT_4}; {NOT_CONFIRMED_0})"
    )


@pytest.mark.parametrize("words", ["", 5], ids=["empty", "not text"])
def test_a_leg_not_confirmed_is_named_in_words(words):
    with pytest.raises(ConfigError, match="unconfirmed names each leg in words"):
        replace(A_GOOD_DAY, unconfirmed=(words,))
