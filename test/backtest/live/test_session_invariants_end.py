"""The session invariants' checks at the end of a run and over its event stream and the fake's
log (`session_invariants`, 16-session-design 7.1, unit 14), on runs written by hand, and
what the trace records (`session_trace`) on the fake OpenAlgo: each check finds what it
must and nothing more. The session's client is "s", another client of the account "o";
`StopHold` holds SBIN long.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from fake_openalgo.log import CALL, FakeLog
from session_invariants import (
    _at_the_end,
    _books_agree,
    _modifies_and_cancels_answered,
    _no_orphan,
    _nothing_after_a_guard_lock,
    _priced_fills,
)
from session_trace import Closing, OrderChange, Run, SessionTrace

from openalgo.backtest._live._session_close import OpenPosition
from openalgo.backtest._live._types import AlertKind, Mode
from openalgo.backtest._live._venue_requests import ModifyRequest

SBIN = (Mode.LIVE, "SBIN", "NSE")
SIDES = {SBIN: "BUY"}
NS = 10**9


def run_of(*, events=(), alerts=(), outcome=None, held=(), traders=(), locked_at=None) -> Run:
    """A run of a live session with what the test gives it."""
    session = SimpleNamespace(
        parts=SimpleNamespace(mode=Mode.LIVE),
        traders=traders,
        reconcile=SimpleNamespace(guard=SimpleNamespace(left=lambda: list(held))),
    )
    return Run(
        session, frozenset({"s"}), list(events), list(alerts), outcome=outcome, locked_at=locked_at
    )


def outcome_of(*positions: OpenPosition, reconciliations=(), differing=()) -> Any:
    """A close's outcome with the open positions, reconciliations and named legs given."""
    summary = SimpleNamespace(differing=tuple(differing), unconfirmed=())
    return SimpleNamespace(
        open_positions=positions, reconciliations=tuple(reconciliations), summary=summary
    )


def leg(units: float = 10, *, covers: int = 10, order_ids=("s1",)) -> Closing:
    """`StopHold`'s leg as its close stopped it."""
    return Closing("StopHold", "SBIN", "NSE", units, True, frozenset(order_ids), covers)


def stop(*, quantity: int = 10, order_id: str = "s1", position: int = 10) -> OrderChange:
    """`StopHold`'s SL-M working at the fake."""
    return OrderChange(
        9, 9, Mode.LIVE, order_id, "s", "StopHold", "SBIN", "NSE", "SELL", "SL-M",
        "trigger pending", quantity, 0, 0, 0.0, position,
    )  # fmt: skip


def trace_of(unprotected=None, waived=None) -> Any:
    return SimpleNamespace(unprotected=unprotected or {}, waived=waived or {})


def at_the_end(run: Run, working, position: int, *, trace=None, closing=None) -> None:
    _at_the_end(
        trace or trace_of(), run, {"StopHold": closing or leg()}, working, {SBIN: position},
        SIDES, {},
    )  # fmt: skip


# ----- no orphan ------------------------------------------------------------------------------


def test_an_order_working_at_the_fake_that_the_engine_does_not_work_is_an_orphan():
    with pytest.raises(AssertionError, match="orphan"):
        _no_orphan(run_of(), {"StopHold": leg(order_ids=())}, [stop()], {SBIN: 10}, {})


def test_a_stop_working_for_a_flat_position_is_found():
    with pytest.raises(AssertionError, match="flat position"):
        _no_orphan(run_of(), {"StopHold": leg()}, [stop(position=0)], {SBIN: 0}, {})


def test_a_stop_left_on_a_position_another_client_sold_flat_is_the_operators():
    _no_orphan(run_of(), {"StopHold": leg()}, [stop(position=0)], {SBIN: 0}, {SBIN: 10})


# ----- protected ------------------------------------------------------------------------------


def test_a_position_with_its_stop_confirmed_for_every_unit_is_protected():
    at_the_end(run_of(), [stop()], 10)


def test_a_position_whose_stop_the_session_has_not_confirmed_is_found():
    with pytest.raises(AssertionError, match="0 confirmed"):
        at_the_end(run_of(), [stop()], 10, closing=leg(covers=0))


#: The open position and the ERROR alert a leg left unprotected must have.
UNPROTECTED_ROW = OpenPosition("StopHold", "SBIN", "NSE", 10, "no stop at the broker")
NAMING_ALERT = (AlertKind.ERROR, "LIVE (StopHold): StopHold SBIN NSE 10 units, no stop")


def test_a_leg_left_unprotected_is_named_by_its_row_and_an_error_alert():
    run = run_of(alerts=[NAMING_ALERT], outcome=outcome_of(UNPROTECTED_ROW))
    at_the_end(run, [], 10, trace=trace_of({"StopHold": "F80"}))


def test_a_leg_left_unprotected_with_no_error_alert_naming_it_is_found():
    run = run_of(outcome=outcome_of(UNPROTECTED_ROW))
    with pytest.raises(AssertionError, match="no ERROR alert names it"):
        at_the_end(run, [], 10, trace=trace_of({"StopHold": "F80"}))


def test_a_leg_left_unprotected_that_open_positions_does_not_list_is_found():
    run = run_of(alerts=[NAMING_ALERT], outcome=outcome_of())
    with pytest.raises(AssertionError, match="open_positions lists"):
        at_the_end(run, [], 10, trace=trace_of({"StopHold": "F60"}))


def test_a_leg_left_unprotected_the_fake_holds_flat_needs_no_exemption():
    run = run_of(alerts=[NAMING_ALERT], outcome=outcome_of(UNPROTECTED_ROW))
    with pytest.raises(AssertionError, match="holds no units"):
        at_the_end(run, [], 0, trace=trace_of({"StopHold": "F80"}))


def test_exits_left_working_above_the_position_are_found_at_the_end():
    with pytest.raises(AssertionError, match="exits of 10 units"):
        at_the_end(run_of(), [stop(position=6)], 6)


# ----- books agree ----------------------------------------------------------------------------


def test_a_difference_of_the_books_that_no_reconciliation_holds_is_found():
    with pytest.raises(AssertionError, match="no reconciliation"):
        _books_agree(run_of(outcome=outcome_of()), {"StopHold": leg()}, {SBIN: 6})


def test_a_difference_of_the_books_a_reconciliation_holds_is_said():
    row = SimpleNamespace(strategy="StopHold")
    run = run_of(outcome=outcome_of(reconciliations=[row]))
    _books_agree(run, {"StopHold": leg()}, {SBIN: 6})


def test_a_difference_of_the_books_the_close_names_is_said():
    named = "at the close: StopHold SBIN NSE 6 units in OpenAlgo's books, the engine's 10"
    run = run_of(outcome=outcome_of(differing=[named]))
    _books_agree(run, {"StopHold": leg()}, {SBIN: 6})


# ----- every modify and cancel answered -------------------------------------------------------


def request(kind: str, request_id: int = 1) -> dict[str, Any]:
    return {
        "kind": "request", "type": kind, "request_id": request_id, "strategy": "StopHold",
        "kernel_order": 2,
    }  # fmt: skip


CLOSED = {"kind": "closed"}


def test_a_modify_with_no_answer_by_the_close_is_found():
    with pytest.raises(AssertionError, match="no apply_modify"):
        _modifies_and_cancels_answered(run_of(events=[request("ModifyRequest"), CLOSED]))


def test_a_modify_answered_once_is_answered():
    answer = {"kind": "settled", "request_id": 1, "applied": "modified"}
    _modifies_and_cancels_answered(run_of(events=[request("ModifyRequest"), answer, CLOSED]))


def test_a_modify_cut_short_by_a_crash_is_not_judged():
    _modifies_and_cancels_answered(run_of(events=[request("ModifyRequest")]))


def test_a_cancel_at_the_end_of_the_run_is_answered_by_the_end():
    _modifies_and_cancels_answered(run_of(events=[request("CancelRequest"), CLOSED]))


def test_a_modify_the_exit_guard_held_to_the_close_with_no_news_is_found():
    intent = SimpleNamespace(strategy="StopHold", purpose=SimpleNamespace(value="stop"))
    held = ModifyRequest(3, 2, intent, "o2", None, None, 0)
    with pytest.raises(AssertionError, match="held by the exit guard"):
        _modifies_and_cancels_answered(run_of(events=[CLOSED], held=[held]))


# ----- nothing after a guard lock -------------------------------------------------------------


def logged(*calls: tuple[str, str, int]) -> Any:
    """A fake whose log holds `calls` (endpoint, client, time), in order."""
    log = FakeLog()
    for what, client, ts_ns in calls:
        log.record(ts_ns, CALL, what, client=client)
    return SimpleNamespace(log=log)


def test_a_book_read_at_the_very_instant_of_the_guard_lock_is_found():
    # Finding 3 of unit 14 round 1: the lock is judged by the log's order, not its time.
    fake = logged(("analyzer", "s", 5 * NS), ("positionbook", "s", 5 * NS))
    with pytest.raises(AssertionError, match="positionbook"):
        _nothing_after_a_guard_lock(run_of(locked_at=1), fake, None)


def test_calls_before_the_lock_and_another_clients_after_it_are_not_the_sessions():
    fake = logged(
        ("positionbook", "s", 4 * NS), ("analyzer", "s", 5 * NS), ("placeorder", "o", 5 * NS)
    )
    _nothing_after_a_guard_lock(run_of(locked_at=2), fake, None)


def test_a_guard_read_after_the_lock_is_no_order_and_no_book_read():
    fake = logged(("analyzer", "s", 5 * NS), ("analyzer", "s", 10 * NS))
    _nothing_after_a_guard_lock(run_of(locked_at=1), fake, None)


# ----- priced fills ---------------------------------------------------------------------------


def broker_fills(*fills: tuple[str, float]) -> Any:
    return SimpleNamespace(
        fills=lambda mode: [SimpleNamespace(order_id=o, price=p) for o, p in fills]
    )


def test_a_fill_adopted_from_other_orders_at_no_price_is_found():
    adopted = {"kind": "adopted", "side": "SELL", "units": 4, "price": 0.0}
    with pytest.raises(AssertionError, match="adopted a fill at no price"):
        _priced_fills(run_of(events=[adopted]), broker_fills())


def test_a_fill_not_flagged_estimated_at_a_price_the_broker_never_filled_is_found():
    fill = {"kind": "fill", "order_id": "e", "price": 501.0}
    with pytest.raises(AssertionError, match="not flagged estimated"):
        _priced_fills(run_of(events=[fill]), broker_fills(("e", 500.0)))


def test_a_fill_flagged_estimated_may_differ_from_the_brokers_price():
    fill = {"kind": "fill", "order_id": "e", "price": 501.0, "estimated": True}
    _priced_fills(run_of(events=[fill]), broker_fills(("e", 500.0)))


# ----- what the trace records -----------------------------------------------------------------


@pytest.fixture
def market(fake_openalgo):
    """The fake with SBIN known and last traded at 812.00, its order changes traced."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo, SessionTrace(fake_openalgo)


def stop_then_modified(fake: Any) -> str:
    """An SL-M SELL of 10 SBIN at 800.00, then modified to 25 units; its order ID."""
    client = fake.client()
    terms = {
        "strategy": "StopHold", "symbol": "SBIN", "action": "SELL", "exchange": "NSE",
        "price_type": "SL-M", "product": "MIS",
    }  # fmt: skip
    order_id = client.placeorder(**terms, quantity=10, trigger_price=800.0)["orderid"]
    answer = client.modifyorder(
        order_id=order_id, **terms, quantity=25, price=0, trigger_price=800.0
    )
    assert answer["status"] == "success"
    return order_id


@pytest.mark.parametrize("analyzer", [True, False], ids=["sandbox", "live"])
def test_a_modify_reaches_the_trace_once_in_either_book(market, analyzer):
    # Finding 4 of unit 14 round 1: the sandbox announces no modify on its order stream.
    fake, trace = market
    fake.set_analyzer(analyzer)
    order_id = stop_then_modified(fake)
    assert [c.quantity for c in trace.changes if c.order_id == order_id] == [10, 25]
