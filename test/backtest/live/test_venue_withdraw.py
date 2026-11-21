"""An entry or add the kernel cancels while its placement waits for the order worker is
withdrawn, not sent (`_venue_withdraw`, `_worker_queue.WithdrawPlace`; 18-multi-strategy
2.5a rules 5, 6 and 12).

The venue map asks for the withdrawal with the cancel (the cancel itself still waits for
the bind, should the worker have taken the placement already); the worker's intake takes
the placement off its queue while it waits there and answers `NotSent(cancelled)`, which
the venue map gives the kernel as the cancel it asked for. The kernel is the real
binding's session (`venue_samples`); the tests play the worker and the loop.
"""

from __future__ import annotations

import pytest
from venue_samples import Rig, accepted, order, unanswered, update

from openalgo.backtest._live._types import OrderIntent, OrderStatus, Purpose
from openalgo.backtest._live._venue_requests import Applied, PlaceRequest, Priority
from openalgo.backtest._live._worker_queue import (
    CANCELLED_WORDS,
    Incoming,
    WithdrawPlace,
    WorkerQueue,
)
from openalgo.backtest._live.inbox import NotSent, NotSentReason, SendResult

LIMIT_ENTRY = order("BUY", 10.0, order_type="LIMIT", price=450.0)


def placed(rig: Rig, request=LIMIT_ENTRY) -> tuple[int, PlaceRequest]:
    """A placement handed to the worker, not answered yet."""
    kernel_order = rig.submit(request)
    (place,) = rig.wait()
    return kernel_order, place


def cancelled_at_next_close(rig: Rig, kernel_order: int) -> list:
    """The requests of the kernel's cancel of `kernel_order` at the next decision point."""
    rig.next_close()
    rig.engine.cancel(kernel_order)
    return rig.wait()


# ----- the venue map asks ------------------------------------------------------------


def test_the_cancel_of_an_entry_not_answered_yet_asks_the_worker_to_withdraw_it():
    rig = Rig()
    kernel_order, place = placed(rig)

    assert cancelled_at_next_close(rig, kernel_order) == [WithdrawPlace(place)]


def test_the_cancel_of_an_entry_sent_with_no_answer_asks_for_nothing():
    # Sent, it may be at the broker: the cancel waits for the order book's bind.
    rig = Rig()
    kernel_order, place = placed(rig)
    rig.map.sent(SendResult(place.request_id, unanswered()))

    assert cancelled_at_next_close(rig, kernel_order) == []


def test_the_cancel_of_a_resting_exit_not_answered_yet_asks_for_nothing():
    # A closing order is never withdrawn: the throttle takes it first.
    rig = Rig()
    _, entry_place = placed(rig, order("BUY", 100.0))
    rig.map.sent(SendResult(entry_place.request_id, accepted("B0")))
    rig.map.news(update("B0", OrderStatus.COMPLETE, filled=100, average=500.0))
    rig.next_close()
    exit_order, exit_place = placed(rig, order("SELL", 100.0, order_type="LIMIT", price=510.0))
    assert exit_place.priority is Priority.PROTECT

    assert cancelled_at_next_close(rig, exit_order) == []


def test_a_withdrawn_entry_is_cancelled_in_the_kernel_and_drops_what_waited():
    rig = Rig()
    kernel_order, place = placed(rig)
    cancelled_at_next_close(rig, kernel_order)

    handled = rig.map.not_sent(NotSent(place, NotSentReason.CANCELLED, CANCELLED_WORDS))

    assert (handled.applied, handled.requests) == (Applied.CANCELLED, ())
    assert rig.status(kernel_order) == "Canceled"
    assert not rig.map.owns_request(place.request_id)
    assert kernel_order not in rig.engine.working_orders()


# ----- an entry waiting when the strategy's entries lock (2.4.0 item M8-X2) ---------


def test_an_entry_not_answered_yet_is_withdrawn_once_its_entries_lock():
    rig = Rig()
    _, place = placed(rig)
    assert rig.map.withdraw_locked() == []
    rig.engine.lock_entries("feed_dead")

    (withdrawal,) = rig.map.withdraw_locked()

    assert (withdrawal.place, withdrawal.reason) == (place, NotSentReason.ENTRIES_LOCKED)
    assert "feed_dead" in withdrawal.words
    assert rig.map.withdraw_locked() == []


def test_an_entry_answered_already_is_not_withdrawn_for_the_entry_locks():
    rig = Rig()
    _, place = placed(rig)
    rig.map.sent(SendResult(place.request_id, unanswered()))
    rig.engine.lock_entries("feed_dead")

    assert rig.map.withdraw_locked() == []


def test_an_exit_is_never_withdrawn_for_the_entry_locks():
    rig = Rig()
    _, entry_place = placed(rig, order("BUY", 100.0))
    rig.map.sent(SendResult(entry_place.request_id, accepted("B0")))
    rig.map.news(update("B0", OrderStatus.COMPLETE, filled=100, average=500.0))
    rig.next_close()
    placed(rig, order("SELL", 100.0))
    rig.engine.lock_entries("feed_dead")

    assert rig.map.withdraw_locked() == []


def test_an_entry_withdrawn_for_the_entry_locks_is_rejected_with_live_lock():
    rig = Rig()
    kernel_order, place = placed(rig)
    rig.engine.lock_entries("feed_dead")
    (withdrawal,) = rig.map.withdraw_locked()

    handled = rig.map.not_sent(NotSent(place, withdrawal.reason, withdrawal.words))

    assert (handled.applied, handled.requests) == (Applied.REJECTED, ())
    assert rig.status(kernel_order) == "Rejected"
    assert rig.engine.order(kernel_order)["reject_code"] == "live_lock"
    assert not rig.map.owns_request(place.request_id)


# ----- the worker's intake withdraws -------------------------------------------------


def entry(order: int) -> PlaceRequest:
    intent = OrderIntent(
        intent_id=f"20261116-live-1-ema_cross-o{order}",
        strategy="ema_cross",
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=10,
        price_type="MARKET",
        product="MIS",
        purpose=Purpose.ENTRY,
        created_ns=0,
    )
    return PlaceRequest(order, order, intent, Priority.ENTER)


class Intake:
    """A worker's intake (`Incoming`), its queue and the messages it posts."""

    def __init__(self) -> None:
        self.posted: list = []
        self.incoming = Incoming(self.posted.append)
        self.queue = WorkerQueue()

    def hand(self, *requests) -> None:
        for request in requests:
            self.incoming.put(request)
        self.incoming.into(self.queue)


def test_an_entry_still_waiting_is_taken_off_the_queue_and_answered_cancelled():
    intake, held = Intake(), entry(1)
    intake.hand(held)

    intake.hand(WithdrawPlace(held))

    assert intake.posted == [NotSent(held, NotSentReason.CANCELLED, CANCELLED_WORDS)]
    assert (len(intake.queue), intake.incoming.unfinished) == (0, 0)


def test_an_entry_and_its_withdrawal_handed_over_together_are_never_queued():
    intake, held = Intake(), entry(1)

    intake.hand(held, WithdrawPlace(held))

    assert intake.posted == [NotSent(held, NotSentReason.CANCELLED, CANCELLED_WORDS)]
    assert (len(intake.queue), intake.incoming.unfinished) == (0, 0)


def test_an_entry_the_worker_took_already_is_left_to_its_bind():
    intake, taken = Intake(), entry(1)
    intake.hand(taken)
    assert intake.queue.take(0) is taken

    intake.hand(WithdrawPlace(taken))

    assert intake.posted == []
    # The entry taken still counts until the worker finishes it; the withdrawal does not.
    assert intake.incoming.unfinished == 1


def test_an_entry_withdrawn_for_the_entry_locks_is_answered_with_that_reason():
    intake, held = Intake(), entry(1)
    intake.hand(held)

    intake.hand(WithdrawPlace(held, NotSentReason.ENTRIES_LOCKED, "locked"))

    assert intake.posted == [NotSent(held, NotSentReason.ENTRIES_LOCKED, "locked")]
    assert (len(intake.queue), intake.incoming.unfinished) == (0, 0)


@pytest.mark.parametrize("others", [0, 2])
def test_a_withdrawal_takes_only_its_own_entry(others):
    intake, held = Intake(), entry(1)
    rest = [entry(2 + n) for n in range(others)]
    intake.hand(held, *rest)

    intake.hand(WithdrawPlace(held))

    assert [intake.queue.take(0) for _ in rest] == rest
    assert intake.queue.take(0) is None
