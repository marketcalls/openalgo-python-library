"""The daily summary of a live session: its figures and its plain text, in the style of
`Result.summary()` (08-live-trading section 8). The expected texts are written by hand."""

from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError, replace

import pytest

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.alerts import SessionSummary, summary_text

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
        "Unprotected : 12 s without a stop at the broker\n"
        "Reconcile   : 1 mismatch\n"
        "End state   : not flat: a position is still open"
    )


def test_the_net_pnl_is_the_gross_pnl_less_the_modelled_charges():
    assert A_BAD_DAY.net_pnl == pytest.approx(-1445.25)


def test_rupees_use_indian_digit_grouping():
    assert line(replace(A_GOOD_DAY, gross_pnl=1234567.891), "Gross P&L") == "Rs 12,34,567.89"


def test_slippage_not_compared_with_a_backtest_reads_n_a():
    assert (
        line(replace(A_GOOD_DAY, slippage=None), "Slippage") == "n/a (no backtest to compare with)"
    )


@pytest.mark.parametrize(
    "slippage", [0.0, 0.004, -0.004], ids=["zero", "under half a paisa", "under half a paisa back"]
)
def test_slippage_that_rounds_to_no_paisa_reads_as_none(slippage):
    assert line(replace(A_GOOD_DAY, slippage=slippage), "Slippage") == "none against the backtest"


def test_a_session_with_no_strategy_names_says_none():
    assert line(replace(A_GOOD_DAY, strategies=()), "Strategies") == "none"


def test_the_seconds_unprotected_are_whole_seconds():
    assert line(replace(A_GOOD_DAY, unprotected_seconds=7.6), "Unprotected") == (
        "8 s without a stop at the broker"
    )


def test_the_summary_is_plain_ascii_text():
    assert summary_text(A_GOOD_DAY).isascii()
    assert summary_text(A_BAD_DAY).isascii()


def test_a_summary_cannot_be_changed():
    with pytest.raises(FrozenInstanceError):
        A_GOOD_DAY.closed_trades = 5
