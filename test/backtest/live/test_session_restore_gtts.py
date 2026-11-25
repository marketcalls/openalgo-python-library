"""What a same-day restart does with each GTT the kernel restores (`_session_restore_gtts.
settle`; review finding 1 of unit N3b): by the kernel's state and the journal's GTT of it,
the journal being written before each GTT call and with each answer.

A stop GTT of 10 SBIN CNC rests at 497.00 (child limit 492.00); a modify would move it to
499.00. Expected decisions come from the module docstring's rules.
"""

from __future__ import annotations

import datetime as dt
import itertools
from types import SimpleNamespace

import pytest
from test_kernel_gtts import G1, at, protected

from openalgo.backtest._live._journal_gtt_rows import GttBinding, GttState, GttTerms
from openalgo.backtest._live._session_restore_gtts import Settle, restore_gtts, settle
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._venue_gtts import GttMap

DAY = dt.date(2026, 11, 16)

RESTING = {"side": "SELL", "quantity_units": 10.0, "trigger_price": 497.0, "price": 492.0}
MODIFIED = {"side": "SELL", "quantity_units": 10.0, "trigger_price": 499.0, "price": 494.0}


def view(state, pending=None):
    """The kernel's restored GTT g1 in `state`, resting at `RESTING`."""
    return {"gtt_id": 1, "purpose": "stop", "state": state, "pending": pending, **RESTING}


def row(state, terms=RESTING, trigger_id="261116000001"):
    """The journal's GTT of g1 in `state` with `terms`."""
    return GttBinding(
        seq=1,
        day=DAY,
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
        (view("CancelPending"), row(GttState.CANCELLED), Settle.CONFIRM_CANCEL),
        (view("Requested"), row(GttState.NOT_PLACED, trigger_id=None), Settle.PLACE),
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
        "cancelled",
        "place refused or not sent",
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
        (view("CancelPending"), row(GttState.FIRED)),
        (view("Active"), row(GttState.ENDED)),
        (view("Triggered"), row(GttState.TRIGGERED)),
    ],
    ids=[
        "place sent, unanswered",
        "resting, not in the journal",
        "resting at other terms",
        "modify sent, unanswered",
        "cancel sent, unanswered",
        "cancelling, not in the journal",
        "cancelling, fired",
        "resting, ended outside",
        "fired, child followed",
    ],
)
def test_a_restored_gtt_the_journal_cannot_explain_is_unsettled_with_why(kernel, journal):
    decided = settle(kernel, journal)
    assert decided.settle is Settle.UNSETTLED
    assert f"the engine's GTT was {kernel['state']}" in decided.why


class Journal:
    """The journal's GTTs of the day, as a restart reads them: `rows`, oldest first."""

    def __init__(self, rows):
        self.rows = rows

    def open_gtts(self):
        return [gtt for gtt in self.rows if not gtt.state.terminal]

    def day_gtts(self, strategy, *, day=None):
        return [gtt for gtt in self.rows if gtt.strategy == strategy]


class Session:
    """The part of a `_session_loop.Session` a restart's GTT restore uses."""

    def __init__(self, rows):
        self.parts = SimpleNamespace(journal=Journal(rows), spec=SimpleNamespace(day=DAY))
        self.events, self.alerts, self.handed = [], [], []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def alert(self, kind, text):
        self.alerts.append((kind, text))

    def hand_over(self, requests):
        self.handed.extend(requests)


def test_a_restart_after_a_confirmed_cancel_confirms_it_and_the_target_exit_goes_on():
    # Saved while the target's cancel of the stop GTT waited; the journal noted the cancel
    # confirmed before the crash. The restart confirms it, so the kernel places its exit.
    engine = protected({"target": ("points", 2.0, None)})
    engine.push_tick(0, 502.0, at(1, 5))
    (cancel,) = engine.take_commands()
    assert (cancel["command"], engine.gtt(G1)["state"]) == ("cancel_gtt", "CancelPending")
    session = Session([row(GttState.CANCELLED)])
    locks = []
    trader = SimpleNamespace(
        name="CncHold",
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
        engine=engine,
        gtts=GttMap(
            engine,
            strategy="CncHold",
            markets=[("SBIN", "NSE")],
            product="CNC",
            request_ids=itertools.count(1),
            now=lambda: at(1, 6),
        ),
        lock=locks.append,
    )

    restore_gtts(session, trader)

    restored = [fields["settle"] for kind, fields in session.events if kind == "gtt_restored"]
    assert restored == ["confirm_cancel"]
    assert (locks, session.alerts, session.handed) == ([], [], [])
    (target_exit,) = [c for c in engine.take_commands() if c["command"] == "place"]
    assert (target_exit["role"], target_exit["quantity_units"]) == ("exit", 10.0)
