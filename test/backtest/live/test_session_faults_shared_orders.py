"""The fault rows of a shared instrument on its orders' answers, on the fake OpenAlgo
(18-multi-strategy 5.2 rows F104 and F105, 4.4; unit M8): one member's unanswered order
holds the instrument for every member, so two members' orders of one shape can never be
told apart, and an ambiguous one holds the instrument for the rest of the day: the
square-off then starts no cancel-then-exit for it, so a member's stop keeps working and is
reported as working (2.4.0 item P3-3).

The market and the strategies are `shared_samples`': SBIN at 500.00, A first in call order.
`LateTrend` is an A that buys its 100 units with a stop 3 points below (497.00) at its
second decision (14:43:01 IST); B (`SbinBreakout`) buys 50 with the same stop at its first
(14:42:01), `LateBreakout` at its second. Every test runs the session invariants, those of
groups and those restated per member among them. Expected values come from the scenario
and the fake's books.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from shared_samples import (
    A_UNITS,
    B_UNITS,
    LateBreakout,
    LateTrend,
    SbinBreakout,
    sbin_orders,
)
from stops_samples import STOP_TRIGGER, stop_world

from openalgo.backtest._live._session_resolved import AMBIGUOUS_LOCK
from openalgo.backtest._live._types import LOGGER_NAME

#: When A's stop is due: 2.5 s after its entry's decision, as for a strategy alone.
A_STOP_DUE = ist(14, 43, 3) + NS // 2


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


def session_calls(world, endpoint: str, after_ns: int) -> list:
    """The session's calls of `endpoint` on SBIN at the fake after `after_ns` (another
    client's calls name no strategy of the session)."""
    names = {"LateTrend", "SbinBreakout", "LateBreakout"}
    return [
        call
        for call in world.fake.log.calls(endpoint)
        if call.ts_ns > after_ns and call.detail["request"].get("strategy") in names
    ]


# ----- F105: an ambiguous order on a shared instrument --------------------------------------


def ambiguous_stop_of_a(world) -> None:
    """Just before A's stop is placed, another client places a SELL SL-M of A's shape (100
    units, trigger 497.00), and the answer to A's own placement is lost after it reached
    the broker: the order book shows two candidates."""
    other = world.fake.client()

    def the_same_stop_by_another_client_then_lost():
        other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="SL-M",
            product="MIS",
            quantity=A_UNITS,
            trigger_price=STOP_TRIGGER,
        )
        world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)

    world.script.at(A_STOP_DUE, the_same_stop_by_another_client_then_lost)


def run_ambiguous(world):
    """F105 to the end: past NSE's square-off at 15:00, then Ctrl+C twice (the flatten can
    send nothing for SBIN). A ends unprotected as the row asserts (18-multi-strategy 5.1 I7
    lists F105): it has no stop. B's stop works on at the broker and is reported working:
    the square-off keeps back its cancel, which could never be sent (2.4.0 item P3-3)."""
    world.trace.leave_unprotected("LateTrend", "F105")
    ambiguous_stop_of_a(world)
    world.stop_at(ist(15, 1))
    world.stop_at(ist(15, 2))
    return world.run([LateTrend, SbinBreakout])


def test_F105_an_ambiguous_stop_sends_nothing_more_for_the_instrument(world):
    run_ambiguous(world)

    # A's stop was the last order the session sent for SBIN: no square-off at 15:00, no
    # flatten, and B's stop is never cancelled.
    assert [
        c.detail["request"]["strategy"] for c in session_calls(world, "placeorder", A_STOP_DUE)
    ] == ["LateTrend"]
    for endpoint in ("modifyorder", "cancelorder"):
        assert session_calls(world, endpoint, A_STOP_DUE) == []


def test_F105_both_members_are_locked_and_one_error_alert_says_so(world, caplog):
    outcome = run_ambiguous(world)

    locks = [(e["strategy"], e["reason"]) for e in events(outcome, "lock")]
    assert ("LateTrend", AMBIGUOUS_LOCK) in locks and ("SbinBreakout", AMBIGUOUS_LOCK) in locks
    (named,) = [text for text in alerts(caplog, "error") if "cannot be told apart" in text]
    assert "LateTrend" in named and "SbinBreakout are locked" in named


def test_F105_b_keeps_its_stop_working_and_both_are_in_open_positions(world):
    outcome = run_ambiguous(world)

    (b_stop,) = [o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "SL-M"]
    assert (b_stop.quantity, b_stop.status) == (B_UNITS, "trigger pending")
    rows = {p.strategy: p for p in outcome.open_positions}
    a, b = rows["LateTrend"], rows["SbinBreakout"]
    assert (a.units, a.stop_confirmed) == (A_UNITS, False)
    assert "ambiguous stop order" in a.note
    assert (b.units, b.stop_order_id, b.stop_confirmed) == (B_UNITS, b_stop.order_id, True)
    assert world.fake.position("SBIN", "NSE") == A_UNITS + B_UNITS


#: What every member's note and square-off alert says of SBIN held for the day by A's stop,
#: before what it asks of the operator (`held`).
HELD_BY = (
    "SBIN NSE is held for the day by LateTrend's stop order, intent "
    "20261116-live-1-LateTrend-o2, which the order book cannot tell apart from another "
    "order: nothing can be sent for it today, so "
)
#: What every such note and alert ends with.
BACKSTOP = "; the broker's own MIS square-off remains the backstop"
#: What A's note and alert ask of the operator: its stop is the ambiguous order, which may
#: or may not work at the broker.
A_STEPS = "check the order book, cancelling what still works, then close it by hand in OpenAlgo"


def b_steps(b_stop_id: str) -> str:
    """What B's note and alert ask of the operator: its stop works at the broker, so a sale
    by hand before its cancel would let it sell again."""
    return f"cancel its stop {b_stop_id} at the broker, then close it by hand in OpenAlgo"


def held(steps: str) -> str:
    """A member's words on SBIN held for the day, asking `steps` of the operator."""
    return f"{HELD_BY}{steps}{BACKSTOP}"


def test_F105_each_members_note_says_the_instrument_is_held_and_bs_says_its_stop_works(world):
    outcome = run_ambiguous(world)

    (b_stop,) = [o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "SL-M"]
    rows = {p.strategy: p for p in outcome.open_positions}
    assert rows["LateTrend"].note.endswith(held(A_STEPS))
    assert rows["SbinBreakout"].note == (
        f"stop {b_stop.order_id} confirmed at the broker; {held(b_steps(b_stop.order_id))}"
    )


def test_F105_one_error_alert_per_member_at_the_square_off_says_nothing_is_sent(world, caplog):
    run_ambiguous(world)

    (b_stop,) = [o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "SL-M"]
    said = [text for text in alerts(caplog, "error") if "no square-off is sent" in text]
    # Each alert gives the advice of the member's note: B's names its working stop to
    # cancel before a sale by hand, which would otherwise sell twice.
    assert [text.split(": ", 1)[1] for text in said] == [
        f"LateTrend SBIN NSE {A_UNITS} units: no square-off is sent for it: ambiguous stop "
        f"order (intent 20261116-live-1-LateTrend-o2); {held(A_STEPS)}",
        f"SbinBreakout SBIN NSE {B_UNITS} units: no square-off is sent for it: stop "
        f"{b_stop.order_id} confirmed at the broker; {held(b_steps(b_stop.order_id))}",
    ]
    assert not [text for text in alerts(caplog, "error") if "cannot be sent now" in text]


def test_F105_a_is_unprotected_for_the_ambiguous_order_and_b_never_after_its_stop(world):
    outcome = run_ambiguous(world)

    reasons = {(w.strategy, w.reason) for w in outcome.unprotected}
    assert ("LateTrend", "ambiguous order") in reasons
    # B's stop protects it from its confirmation to the end: the square-off starts no
    # cancel of it.
    b_windows = [w for w in outcome.unprotected if w.strategy == "SbinBreakout"]
    assert [w for w in b_windows if w.from_ns > A_STOP_DUE] == []


# ----- F104: never two unbound intents on one instrument ------------------------------------


def run_a_unanswered(world):
    """A's entry (14:43:01) reaches the broker with no answer and the order book cannot be
    read for a while; B decides an entry of its own in the same step."""
    faults = world.fake.faults

    def a_unanswered():
        faults.bad_gateway_page("placeorder", reached_broker=True)
        for _ in range(8):
            faults.gateway_error("orderbook")

    world.script.at(ist(14, 43), a_unanswered)
    world.stop_at(ist(14, 46))
    return world.run([LateTrend, LateBreakout], on_stop="keep")


def test_F104_no_member_sends_on_the_instrument_while_anothers_intent_is_unbound(world):
    outcome = run_a_unanswered(world)

    (bound,) = [
        e
        for e in events(outcome, "resolved")
        if e["strategy"] == "LateTrend" and e["resolution"] == "bound"
    ]
    a_entry = sbin_orders(world, "LateTrend")[0]
    assert bound["order_id"] == a_entry.order_id
    # Never two unbound intents on SBIN: nothing is placed there until A's is bound.
    between = [c for c in world.fake.log.calls("placeorder") if a_entry.placed_ns < c.ts_ns]
    assert [c.ts_ns >= bound["ts_ns"] for c in between] == [True] * len(between)


def test_F104_an_entry_held_behind_anothers_unbound_intent_is_not_sent_later(world):
    # 16-session-design 2.4: an entry the order path's hold refuses is not tried again
    # (its moment has passed); B sees it rejected.
    outcome = run_a_unanswered(world)

    assert sbin_orders(world, "LateBreakout") == []
    orders = outcome.outputs["LateBreakout"]["orders"]
    assert [code for code in orders["reject_code"] if code is not None] == ["venue"]
    refused = [e for e in events(outcome, "not_sent") if e["strategy"] == "LateBreakout"]
    assert [e["reason"] for e in refused] == ["refused"]
