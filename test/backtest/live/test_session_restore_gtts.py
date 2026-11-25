"""What a same-day restart does with each GTT the kernel restores (`_session_restore_gtts.
settle`; review finding 1 of unit N3b): by the kernel's state and the journal's GTT of it,
the journal being written before each GTT call and with each answer.

A stop GTT of 10 SBIN CNC rests at 497.00 (child limit 492.00); a modify would move it to
499.00. Expected decisions come from the module docstring's rules.
"""

from __future__ import annotations

import datetime as dt

import pytest

from openalgo.backtest._live._journal_gtt_rows import GttBinding, GttState, GttTerms
from openalgo.backtest._live._session_restore_gtts import Settle, settle
from openalgo.backtest._live._types import Purpose

RESTING = {"side": "SELL", "quantity_units": 10.0, "trigger_price": 497.0, "price": 492.0}
MODIFIED = {"side": "SELL", "quantity_units": 10.0, "trigger_price": 499.0, "price": 494.0}


def view(state, pending=None):
    """The kernel's restored GTT g1 in `state`, resting at `RESTING`."""
    return {"gtt_id": 1, "purpose": "stop", "state": state, "pending": pending, **RESTING}


def row(state, terms=RESTING, trigger_id="261116000001"):
    """The journal's GTT of g1 in `state` with `terms`."""
    return GttBinding(
        seq=1,
        day=dt.date(2026, 11, 16),
        strategy="CncHold",
        gtt_id=1,
        request_id=3,
        trigger_id=trigger_id,
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
        purpose=Purpose.STOP,
        terms=GttTerms(
            terms["side"], terms["trigger_price"], terms["price"], int(terms["quantity_units"])
        ),
        state=state,
        recorded_ns=0,
        changed_ns=0,
    )


@pytest.mark.parametrize(
    ("kernel", "journal", "expected"),
    [
        (view("Requested"), None, Settle.PLACE),
        (view("Requested"), row(GttState.PLACED), Settle.CONFIRM_PLACE),
        (view("Requested"), row(GttState.ACTIVE), Settle.CONFIRM_PLACE),
        (view("Active"), row(GttState.ACTIVE), Settle.TAKE_BACK),
        (view("Active"), row(GttState.TRIGGERED), Settle.TAKE_BACK),
        (view("ModifyPending", MODIFIED), row(GttState.ACTIVE), Settle.MODIFY),
        (view("ModifyPending", MODIFIED), row(GttState.ACTIVE, MODIFIED), Settle.CONFIRM_MODIFY),
        (view("CancelPending"), row(GttState.ACTIVE), Settle.CANCEL),
    ],
    ids=[
        "place never sent",
        "placed, bound",
        "placed, resting",
        "resting",
        "fired, child not resolved",
        "modify never sent",
        "modified",
        "cancel never sent",
    ],
)
def test_a_restored_gtt_the_journal_explains_is_taken_back(kernel, journal, expected):
    assert settle(kernel, journal).settle is expected


@pytest.mark.parametrize(
    ("kernel", "journal"),
    [
        (view("Requested"), row(GttState.REQUESTED, trigger_id=None)),
        (view("Active"), None),
        (view("Active"), row(GttState.ACTIVE, MODIFIED)),
        (view("ModifyPending", MODIFIED), row(GttState.MODIFYING)),
        (view("CancelPending"), row(GttState.CANCELLING)),
        (view("CancelPending"), None),
        (view("Triggered"), row(GttState.TRIGGERED)),
    ],
    ids=[
        "place sent, unanswered",
        "resting, not in the journal",
        "resting at other terms",
        "modify sent, unanswered",
        "cancel sent, unanswered",
        "cancel answered",
        "fired, child followed",
    ],
)
def test_a_restored_gtt_the_journal_cannot_explain_is_unsettled_with_why(kernel, journal):
    decided = settle(kernel, journal)
    assert decided.settle is Settle.UNSETTLED
    assert f"the engine's GTT was {kernel['state']}" in decided.why
