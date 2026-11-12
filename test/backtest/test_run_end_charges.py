"""Charges do not depend on where a run ends when nothing happens in between: a run
that ends before the session close settles that session's contract notes at its end,
exactly as the session-close step would.

The runs use `accounting_runs.future_round_trip`: an NFO future on 2026-11-04 (session
09:15 to 15:40 IST), NRML, default charges with contract-note rounding and the default
slippage; it buys at the 09:30 open and exits at the 09:45 open. One run has bars to
the session close (385 minutes); the other has bars to 15:29 (375 minutes), so it ends
at 15:30, ten minutes before the close, with the same two fills.
"""

import pandas as pd
import pytest
from accounting_runs import future_round_trip, session_bars

# Bars from 09:15 to 15:29: the run ends at 15:30, before the 15:40 NFO close.
TO_FIFTEEN_TWENTY_NINE = 375
LEDGER = ["equity_rupees", "realized_rupees", "costs_rupees"]


@pytest.fixture(scope="module")
def runs():
    early = future_round_trip(data=session_bars(minutes=TO_FIFTEEN_TWENTY_NINE))
    return early, future_round_trip()


def test_the_two_runs_have_the_same_fills(runs):
    early, full = runs
    pd.testing.assert_frame_equal(early.fills, full.fills)


def test_the_charges_are_the_same(runs):
    early, full = runs
    pd.testing.assert_frame_equal(early.costs, full.costs)


def test_the_contract_notes_are_the_same_and_settled(runs):
    early, full = runs
    pd.testing.assert_frame_equal(early.contract_notes, full.contract_notes)
    assert list(early.contract_notes["settled"]) == [True]


def test_the_ledger_at_the_end_is_the_same(runs):
    early, full = runs
    pd.testing.assert_series_equal(
        early.equity[LEDGER].iloc[-1], full.equity[LEDGER].iloc[-1], check_names=False
    )
