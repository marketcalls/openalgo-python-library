"""The venue map (`venue_map`, 16-session-design 2.5 and 2.7): the kernel's commands become
the order worker's requests, a modify or cancel of an order not bound yet waits for the
bind, and every answer of the worker becomes the one kernel call it calls for; each
Modify gets exactly one result (R8), whatever the outcome.

The kernel is the real binding's session (`venue_samples`); the test plays the worker.
"""

from __future__ import annotations

import pytest
from venue_samples import Rig, accepted, order, rejected, unanswered, update

from openalgo.backtest._live._order_book import Resolution, Resolved
from openalgo.backtest._live._order_check import OrderCheck
from openalgo.backtest._live._types import Mode, OrderStatus, Purpose
from openalgo.backtest._live._venue_requests import (
    CancelRequest,
    ModifyRequest,
    OrderTerms,
    PlaceRequest,
    Priority,
)
from openalgo.backtest._live.inbox import NotSent, NotSentReason, SendResult, Settled
from openalgo.backtest._live.venue_map import Applied
from openalgo.backtest.errors import LiveError

LIMIT_ENTRY = order("BUY", 10.0, order_type="LIMIT", price=450.0)
LIMIT_CHANGE = order("BUY", 20.0, order_type="LIMIT", price=455.0)
PLACE_O1 = {
    "command": "place",
    "order_id": 1,
    "instrument": 0,
    "product": "MIS",
    "side": "BUY",
    "role": "entry",
    "reason": None,
    "quantity_units": 10.0,
    "order_type": "MARKET",
    "price": None,
    "trigger_price": None,
}


def placed(rig: Rig, request=LIMIT_ENTRY) -> tuple[int, PlaceRequest]:
    """A placement handed to the worker, not answered yet."""
    kernel_order = rig.submit(request)
    (place,) = rig.wait()
    return kernel_order, place


def bound(rig: Rig, request=LIMIT_ENTRY, order_id: str = "B1") -> int:
    """A placement the broker accepted as `order_id`."""
    kernel_order, place = placed(rig, request)
    assert rig.map.sent(SendResult(place.request_id, accepted(order_id))).applied is Applied.BOUND
    return kernel_order


def modified(rig: Rig, kernel_order: int) -> ModifyRequest:
    """At the next decision point, a modify of `kernel_order` to 20 units at 455.00."""
    rig.next_close()
    rig.engine.modify(kernel_order, LIMIT_CHANGE)
    (modify,) = rig.wait()
    return modify


def long_position(rig: Rig) -> None:
    """Hold 100 SBIN bought at 500.00, at the next decision point."""
    kernel_order = bound(rig, order("BUY", 100.0), order_id="B0")
    rig.map.news(update("B0", OrderStatus.COMPLETE, filled=100, average=500.0))
    rig.next_close()
    assert rig.status(kernel_order) == "Filled"


def test_a_placement_becomes_a_request_with_its_intent_and_priority():
    rig = Rig()
    kernel_order, place = placed(rig)

    assert (place.kernel_order, place.priority) == (kernel_order, Priority.ENTER)
    intent = place.intent
    assert (intent.intent_id, intent.purpose, intent.price_type, intent.price) == (
        f"20250701-live-1-ema_cross-o{kernel_order}",
        Purpose.ENTRY,
        "LIMIT",
        450.0,
    )
    assert rig.map.owns_request(place.request_id)


def test_an_exit_the_session_asks_for_is_a_protective_placement():
    rig = Rig()
    long_position(rig)
    rig.wait()

    assert rig.engine.request_exit(0, "MIS", "operator_stop") == "sent"
    (place,) = rig.map.requests(rig.engine.take_commands())

    assert (place.intent.purpose, place.intent.action, place.priority) == (
        Purpose.RISK_EXIT,
        "SELL",
        Priority.PROTECT,
    )


def test_a_modify_of_an_order_not_bound_waits_for_the_bind():
    rig = Rig()
    kernel_order, place = placed(rig)
    rig.next_close()
    rig.engine.modify(kernel_order, LIMIT_CHANGE)

    assert rig.wait() == []
    handled = rig.map.sent(SendResult(place.request_id, accepted("B1")))

    (modify,) = handled.requests
    assert isinstance(modify, ModifyRequest)
    assert (modify.order_id, modify.terms, modify.replaces) == (
        "B1",
        OrderTerms(20, "LIMIT", 455.0, 0.0),
        OrderTerms(10, "LIMIT", 450.0, 0.0),
    )


def test_what_waited_for_the_bind_is_released_in_the_order_the_kernel_gave_it():
    rig = Rig()
    kernel_order, place = placed(rig)
    rig.next_close()
    rig.engine.modify(kernel_order, LIMIT_CHANGE)
    rig.engine.cancel(kernel_order)
    assert rig.wait() == []

    handled = rig.map.sent(SendResult(place.request_id, accepted("B1")))

    assert [type(request) for request in handled.requests] == [ModifyRequest, CancelRequest]


def test_an_unanswered_placement_releases_what_waited_when_the_order_book_binds_it():
    rig = Rig()
    kernel_order, place = placed(rig)
    assert rig.map.sent(SendResult(place.request_id, unanswered())).applied is Applied.NOTHING
    rig.next_close()
    rig.engine.cancel(kernel_order)
    assert rig.wait() == []

    handled = rig.map.resolved(Resolution(place.intent, Resolved.BOUND, order_id="B1"))

    assert handled.applied is Applied.BOUND
    assert [(type(r), r.order_id) for r in handled.requests] == [(CancelRequest, "B1")]
    assert rig.map.kernel_order_of("B1") == kernel_order


def with_a_waiting_cancel(rig: Rig) -> tuple[int, PlaceRequest]:
    """A placement not bound yet, and a cancel of it waiting for the bind."""
    kernel_order, place = placed(rig)
    rig.next_close()
    rig.engine.cancel(kernel_order)
    assert rig.wait() == []
    return kernel_order, place


def rejected_by_venue(rig: Rig, kernel_order: int) -> bool:
    row = rig.engine.order(kernel_order)
    return (row["status"], row["reject_code"]) == ("Rejected", "venue")


def test_a_refused_placement_is_rejected_and_drops_what_waited():
    rig = Rig()
    kernel_order, place = with_a_waiting_cancel(rig)

    handled = rig.map.sent(SendResult(place.request_id, rejected()))

    assert (handled.applied, handled.requests) == (Applied.REJECTED, ())
    assert rejected_by_venue(rig, kernel_order)


def test_a_placement_the_order_book_never_held_is_rejected_and_drops_what_waited():
    rig = Rig()
    kernel_order, place = with_a_waiting_cancel(rig)
    rig.map.sent(SendResult(place.request_id, unanswered()))

    handled = rig.map.resolved(Resolution(place.intent, Resolved.NOT_PLACED))

    assert (handled.applied, handled.requests) == (Applied.REJECTED, ())
    assert rejected_by_venue(rig, kernel_order)


def test_an_ambiguous_placement_stays_unbound():
    rig = Rig()
    kernel_order, place = placed(rig)

    handled = rig.map.resolved(Resolution(place.intent, Resolved.AMBIGUOUS, candidates=("B1",)))

    assert handled.applied is Applied.NOTHING
    assert (rig.map.order_id_of(kernel_order), rig.status(kernel_order)) == ("", "Submitted")


CHECK_NEW = OrderCheck("B1", OrderStatus.OPEN, 20, None, 455.0, 0.0, "LIMIT")
CHECK_OLD = OrderCheck("B1", OrderStatus.OPEN, 10, None, 450.0, 0.0, "LIMIT")
CHECK_DONE = OrderCheck("B1", OrderStatus.COMPLETE, 10, None, 450.0, 0.0, "LIMIT")


@pytest.mark.parametrize(
    ("answer", "applied", "units"),
    [
        (lambda m: SendResult(m.request_id, accepted("B1")), Applied.MODIFIED, 20.0),
        (lambda m: SendResult(m.request_id, rejected()), Applied.MODIFY_REFUSED, 10.0),
        (lambda m: NotSent(m, NotSentReason.REFUSED, "silent"), Applied.MODIFY_REFUSED, 10.0),
        (lambda m: NotSent(m, NotSentReason.LOCKED), Applied.MODIFY_REFUSED, 10.0),
        (lambda m: NotSent(m, NotSentReason.ORDER_ENDED), Applied.MODIFY_REFUSED, 10.0),
        (lambda m: Settled(m, CHECK_NEW), Applied.MODIFIED, 20.0),
        (lambda m: Settled(m, CHECK_OLD), Applied.MODIFY_REFUSED, 10.0),
        (lambda m: Settled(m, None), Applied.MODIFY_REFUSED, 10.0),
    ],
    ids=[
        "accepted",
        "refused",
        "not sent",
        "locked",
        "order ended",
        "row shows it",
        "row shows the old terms",
        "no row in time",
    ],
)
def test_every_outcome_of_a_modify_gives_the_kernel_one_result(answer, applied, units):
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)
    message = answer(modify)

    handled = route(rig, message)
    again = route(rig, message if not isinstance(message, SendResult) else Settled(modify, None))

    assert (handled.applied, rig.units(kernel_order)) == (applied, units)
    assert again.applied is Applied.NOTHING


def route(rig: Rig, message):
    if isinstance(message, SendResult):
        return rig.map.sent(message)
    if isinstance(message, NotSent):
        return rig.map.not_sent(message)
    return rig.map.settled(message)


@pytest.mark.parametrize("reason", [NotSentReason.HELD, NotSentReason.SILENT])
def test_a_modify_put_back_by_the_worker_waits_for_its_result(reason):
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)

    held = rig.map.not_sent(NotSent(modify, reason, "unbound intent"))
    result = rig.map.sent(SendResult(modify.request_id, accepted("B1")))

    assert (held.applied, result.applied, rig.units(kernel_order)) == (
        Applied.NOTHING,
        Applied.MODIFIED,
        20.0,
    )


def test_a_modify_whose_order_ended_is_settled_by_its_news():
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)

    ended = rig.map.settled(Settled(modify, CHECK_DONE))
    rig.map.news(update("B1", OrderStatus.COMPLETE, quantity=10, filled=10, average=450.0))

    assert ended.applied is Applied.NOTHING
    assert rig.status(kernel_order) == "Filled"
    assert rig.map.sent(SendResult(modify.request_id, accepted("B1"))).applied is Applied.NOTHING


def test_news_showing_the_modified_quantity_confirms_it_before_fills_beyond_the_old_units():
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)
    rig.map.sent(SendResult(modify.request_id, unanswered()))

    result = rig.map.news(update("B1", quantity=20, filled=15, average=455.0))

    assert result.mismatch is False
    assert (rig.units(kernel_order), rig.engine.order(kernel_order)["filled_units"]) == (20.0, 15.0)
    assert rig.map.settled(Settled(modify, CHECK_NEW)).applied is Applied.NOTHING


def cancelled(rig: Rig, kernel_order: int) -> CancelRequest:
    rig.next_close()
    rig.engine.cancel(kernel_order)
    (cancel,) = rig.wait()
    return cancel


def test_an_accepted_cancel_counts_once_the_cancelled_news_comes():
    rig = Rig()
    kernel_order = bound(rig)
    cancel = cancelled(rig, kernel_order)

    answer = rig.map.sent(SendResult(cancel.request_id, accepted("B1")))
    working = rig.status(kernel_order)
    rig.map.news(update("B1", OrderStatus.CANCELLED, quantity=10))

    assert (answer.applied, working, rig.status(kernel_order)) == (
        Applied.NOTHING,
        "Submitted",
        "Canceled",
    )


def test_a_cancel_the_row_shows_still_working_is_refused_so_it_can_be_sent_again():
    rig = Rig()
    kernel_order = bound(rig)
    cancel = cancelled(rig, kernel_order)
    rig.map.sent(SendResult(cancel.request_id, rejected("429 too many requests")))

    handled = rig.map.settled(Settled(cancel, CHECK_OLD))

    assert handled.applied is Applied.CANCEL_REFUSED
    rig.engine.cancel(kernel_order)
    (again,) = rig.map.requests(rig.engine.take_commands())
    assert isinstance(again, CancelRequest) and again.request_id != cancel.request_id


def test_a_cancel_withdrawn_before_it_was_sent_leaves_the_order_working_with_nothing_waiting():
    # The exit guard held it back and withdraws it (a sale by hand adopted after the
    # session's last lines): the kernel takes it as refused, and the order is no longer
    # changing, so a cancel asked for later is a new request.
    rig = Rig()
    kernel_order = bound(rig)
    cancel = cancelled(rig, kernel_order)

    taken = rig.map.withdraw_cancel(cancel)

    assert (taken, rig.status(kernel_order)) == (True, "Submitted")
    assert not (rig.map.changing(kernel_order) or rig.map.owns_request(cancel.request_id))
    rig.engine.cancel(kernel_order)
    (again,) = rig.map.requests(rig.engine.take_commands())
    assert isinstance(again, CancelRequest) and again.request_id != cancel.request_id


def test_a_withdrawn_cancel_restored_waits_for_its_answer_again():
    # The exit guard holds it again (the adoption it was withdrawn for was refused): should
    # it be sent, its answer is this map's, and the CANCELLED news ends the order.
    rig = Rig()
    kernel_order = bound(rig)
    cancel = cancelled(rig, kernel_order)
    rig.map.withdraw_cancel(cancel)

    rig.map.restore_cancel(cancel)

    assert rig.map.changing(kernel_order) and rig.map.owns_request(cancel.request_id)
    answer = rig.map.sent(SendResult(cancel.request_id, accepted("B1")))
    rig.map.news(update("B1", OrderStatus.CANCELLED, quantity=10))
    assert (answer.applied, rig.status(kernel_order)) == (Applied.NOTHING, "Canceled")


@pytest.mark.parametrize(
    "message",
    [
        lambda cancel: NotSent(cancel, NotSentReason.REFUSED, "unbound intent"),
        lambda cancel: NotSent(cancel, NotSentReason.ORDER_ENDED),
        lambda cancel: Settled(cancel, CHECK_DONE),
    ],
    ids=["refused before it was sent", "order ended", "row terminal"],
)
def test_a_cancel_changes_nothing_until_a_row_or_the_news_decides(message):
    rig = Rig()
    kernel_order = bound(rig)
    cancel = cancelled(rig, kernel_order)

    handled = route(rig, message(cancel))

    assert (handled.applied, rig.status(kernel_order)) == (Applied.NOTHING, "Submitted")


@pytest.mark.parametrize(
    "reason",
    [NotSentReason.STALE, NotSentReason.REFUSED, NotSentReason.LOCKED],
)
def test_an_entry_not_sent_is_rejected(reason):
    rig = Rig()
    kernel_order, place = placed(rig)

    handled = rig.map.not_sent(NotSent(place, reason))

    assert (handled.applied, rig.status(kernel_order)) == (Applied.REJECTED, "Rejected")


@pytest.mark.parametrize(
    "reason",
    [NotSentReason.HELD, NotSentReason.SILENT, NotSentReason.LOCKED],
)
def test_an_exit_held_or_locked_out_stays_working(reason):
    rig = Rig()
    long_position(rig)
    kernel_order, place = placed(rig, order("SELL", 100.0, order_type="LIMIT", price=510.0))

    handled = rig.map.not_sent(NotSent(place, reason))

    assert (place.priority, handled.applied, rig.status(kernel_order)) == (
        Priority.PROTECT,
        Applied.NOTHING,
        "Submitted",
    )


def test_a_terminal_news_ends_what_waited_for_a_result():
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)

    rig.map.news(update("B1", OrderStatus.CANCELLED, quantity=10))

    assert rig.status(kernel_order) == "Canceled"
    assert rig.map.sent(SendResult(modify.request_id, accepted("B1"))).applied is Applied.NOTHING


def test_a_terminal_word_the_kernel_refuses_leaves_the_modify_waiting_for_its_result():
    rig = Rig()
    kernel_order = bound(rig)
    modify = modified(rig, kernel_order)
    # Paused at a bar close the kernel refuses every result: here the stand-in for any
    # refusal of a terminal word but the race of an order it had ended.
    rig.next_close()
    refused = rig.map.news(update("B1", OrderStatus.CANCELLED, quantity=10))
    rig.wait()

    answer = rig.map.sent(SendResult(modify.request_id, accepted("B1")))

    assert (refused.mismatch, answer.applied, rig.units(kernel_order)) == (
        True,
        Applied.MODIFIED,
        20.0,
    )


def test_intent_ids_carry_the_journal_sessions_mode_and_start():
    rig = Rig(mode=Mode.SANDBOX, starts=3)
    kernel_order, place = placed(rig)
    assert place.intent.intent_id == f"20250701-sandbox-3-ema_cross-o{kernel_order}"


@pytest.mark.parametrize(
    ("call", "words"),
    [
        (lambda rig: rig.map.requests([{"command": "expire", "order_id": 1}]), "unknown command"),
        (lambda rig: rig.map.requests([{"command": "cancel", "order_id": 99}]), "o99"),
        (lambda rig: rig.map.news(update("B404")), "B404"),
        (lambda rig: rig.map.sent(SendResult(404, accepted("B1"))), "404"),
        (lambda rig: rig.map.requests([PLACE_O1, PLACE_O1]), "placed twice"),
        (lambda rig: rig.map.not_sent(NotSent(object(), NotSentReason.LOCKED)), "order requests"),
    ],
    ids=[
        "unknown command",
        "order not placed here",
        "news of another order",
        "foreign answer",
        "placed twice",
        "not an order request",
    ],
)
def test_what_is_not_this_maps_is_refused(call, words):
    with pytest.raises(LiveError, match=words):
        call(Rig())
