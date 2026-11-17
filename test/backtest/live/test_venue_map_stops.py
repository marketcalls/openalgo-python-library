"""The venue map's answers for a leg's broker-held stop (`venue_map`; 16-session-design 4.10,
the kernel's NOT_PLACED contract in the merge notes, "For unit 8 and unit 10 (from K3)"):
a stop that never reached the broker ends as cancelled with no request from the session,
which the kernel places again once; a stop the broker refused ends rejected, which exits
the leg at market; and a cancel no read can ever settle is refused, so the stop keeps
protecting and the exit waits.

The kernel is the real binding's session (`venue_samples`); the test plays the worker.
"""

from __future__ import annotations

import pytest
from venue_samples import Rig, accepted, order, rejected, unanswered, update

from openalgo.backtest._live._order_book import Resolution, Resolved
from openalgo.backtest._live._types import OrderStatus, Purpose
from openalgo.backtest._live._venue_requests import CancelRequest, PlaceRequest
from openalgo.backtest._live.inbox import NotSent, NotSentReason, SendResult, Settled
from openalgo.backtest._live.venue_map import Applied

#: A stop 3 points below the entry.
STOP_3 = {"stop": ("points", 3.0, None)}


def holding_with_stop(rig: Rig) -> PlaceRequest:
    """Hold 100 SBIN bought at 500.00 with a stop 3 points below; the stop's placement is
    handed to the worker and not answered yet."""
    rig.submit(order("BUY", 100.0, rules=STOP_3))
    (entry,) = rig.wait()
    rig.map.sent(SendResult(entry.request_id, accepted("B0")))
    rig.map.news(update("B0", OrderStatus.COMPLETE, filled=100, average=500.0))
    (stop,) = rig.map.requests(rig.engine.take_commands())
    assert stop.intent.purpose is Purpose.STOP
    return stop


def next_placement(rig: Rig) -> PlaceRequest:
    (placement,) = rig.map.requests(rig.engine.take_commands())
    assert isinstance(placement, PlaceRequest)
    return placement


@pytest.mark.parametrize(
    "never_placed",
    [
        lambda rig, stop: (
            rig.map.sent(SendResult(stop.request_id, unanswered())),
            rig.map.resolved(Resolution(stop.intent, Resolved.NOT_PLACED)),
        ),
        lambda rig, stop: rig.map.not_sent(NotSent(stop, NotSentReason.REFUSED, "a defect")),
    ],
    ids=["the order book never held it", "the worker could not send it"],
)
def test_a_stop_that_never_reached_the_broker_is_placed_again_once(never_placed):
    rig = Rig()
    stop = holding_with_stop(rig)

    never_placed(rig, stop)

    again = next_placement(rig)
    assert (again.intent.purpose, again.intent.quantity) == (Purpose.STOP, 100)
    assert again.kernel_order != stop.kernel_order


def test_a_stop_the_broker_refused_exits_the_leg_at_market():
    rig = Rig()
    stop = holding_with_stop(rig)

    handled = rig.map.sent(SendResult(stop.request_id, rejected("trigger invalid")))

    exit_ = next_placement(rig)
    assert handled.applied is Applied.REJECTED
    assert (exit_.intent.purpose, exit_.intent.price_type) == (Purpose.RISK_EXIT, "MARKET")


def test_a_cancel_no_read_can_settle_is_refused_so_the_stop_keeps_protecting():
    rig = Rig()
    stop = holding_with_stop(rig)
    rig.map.sent(SendResult(stop.request_id, accepted("S1")))
    rig.engine.cancel(stop.kernel_order)
    (cancel,) = rig.map.requests(rig.engine.take_commands())
    assert isinstance(cancel, CancelRequest)
    rig.map.not_sent(NotSent(cancel, NotSentReason.REFUSED, "not this session's order"))

    handled = rig.map.settled(Settled(cancel, None))

    assert handled.applied is Applied.CANCEL_REFUSED
    assert rig.status(stop.kernel_order) == "Submitted"
