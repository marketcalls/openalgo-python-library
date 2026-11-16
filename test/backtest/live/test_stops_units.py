"""The stops supervisor's parts on their own (`_live/_stops_kernel.py`, `_stops_cover.py`,
`_stops_orders.py`; 16-session-design 4.2 and 4.6): the marketable limit of the price
rule, the unprotected windows, and what the supervisor knows of a leg's orders.

The marketable limit is checked against the kernel's own option stop (`test_kernel_legs`'
session, no server): both use the one 5 percent buffer.
"""

from __future__ import annotations

import pytest
from test_kernel_legs import NIFTY_CALL, STOP_3, holding

from openalgo.backtest._live._order_answers import Outcome
from openalgo.backtest._live._order_book import Resolved
from openalgo.backtest._live._stops_cover import Windows
from openalgo.backtest._live._stops_kernel import marketable_limit
from openalgo.backtest._live._stops_orders import LegOrders, Sent
from openalgo.backtest._live._types import Mode, OrderIntent, OrderStatus, OrderUpdate, Purpose
from openalgo.backtest._live._venue_requests import PlaceRequest, Priority
from openalgo.backtest._live.inbox import NotSentReason

LEG = ("ema_cross", "SBIN", "NSE")


@pytest.mark.parametrize(
    ("price", "side", "tick", "limit"),
    [
        (120.30, "SELL", 0.05, 114.25),
        (100.0, "SELL", 0.05, 95.0),
        (100.0, "BUY", 0.05, 105.0),
        (101.03, "BUY", 0.05, 106.1),
        (500.0, "SELL", 0.01, 475.0),
        (0.05, "SELL", 0.05, 0.05),
    ],
    ids=[
        "a sale rounds down to the tick",
        "a sale on the tick",
        "a purchase on the tick",
        "a purchase rounds up to the tick",
        "a paisa tick",
        "never below one tick",
    ],
)
def test_a_marketable_limit_is_5_percent_beyond_the_price_on_the_tick_away_from_the_trader(
    price, side, tick, limit
):
    assert marketable_limit(price, side, tick) == limit


def test_the_marketable_limit_uses_the_kernels_own_option_stop_buffer():
    engine = holding(STOP_3, units=75, price=123.3, instrument=NIFTY_CALL)
    (stop,) = [c for c in engine.take_commands() if c["command"] == "place"]
    assert marketable_limit(stop["trigger_price"], "SELL", 0.05) == stop["price"]


def test_a_window_opens_at_the_first_news_and_closes_at_the_last():
    windows = Windows()
    windows.note(LEG, 10, "stop unconfirmed", (100, 105))
    windows.note(LEG, 0, "", (110, 120))

    (window,) = windows.rows()
    assert (window.from_ns, window.to_ns, window.units, window.reason) == (
        100,
        120,
        10,
        "stop unconfirmed",
    )
    assert windows.seconds(10**12) == 20 / 10**9


def test_a_change_of_reason_closes_the_window_and_opens_the_next_at_the_same_instant():
    windows = Windows()
    windows.note(LEG, 10, "stop unconfirmed", (100, 100))
    windows.note(LEG, 10, "exiting", (130, 140))

    first, second = windows.rows()
    assert (first.to_ns, second.from_ns, second.to_ns) == (140, 140, None)
    assert windows.seconds(200) == (40 + 60) / 10**9


def test_a_window_still_open_counts_up_to_the_moment_asked():
    windows = Windows()
    windows.note(LEG, 5, "held cancel", (100, 100))

    assert windows.current("ema_cross").seconds(400) == 300 / 10**9
    assert windows.note(LEG, 5, "held cancel", (200, 200)) is None


def intent(purpose: Purpose, quantity: int = 10) -> OrderIntent:
    return OrderIntent(
        intent_id=f"20261116-live-1-ema_cross-o{purpose.value}",
        strategy="ema_cross",
        symbol="SBIN",
        exchange="NSE",
        action="SELL" if purpose is Purpose.STOP else "BUY",
        quantity=quantity,
        price_type="SL-M" if purpose is Purpose.STOP else "MARKET",
        product="MIS",
        purpose=purpose,
        created_ns=0,
    )


def word(status: OrderStatus, filled: int = 0) -> OrderUpdate:
    return OrderUpdate(
        order_id="S1",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        filled_quantity=filled,
        pending_quantity=10 - filled,
        average_price=0.0,
        status=status,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=0,
        source="stream",
    )


def placed_stop() -> LegOrders:
    leg = LegOrders()
    leg.handed(PlaceRequest(1, 2, intent(Purpose.STOP), Priority.PROTECT))
    return leg


def test_a_stops_placement_guards_the_leg_and_makes_it_the_legs_stop():
    leg = placed_stop()

    assert (leg.guarded, leg.stop, leg.orders[2].state) == (True, 2, Sent.HANDED)


@pytest.mark.parametrize(
    "status", [OrderStatus.TRIGGER_PENDING, OrderStatus.OPEN], ids=["trigger pending", "open"]
)
def test_a_stop_accepted_is_confirmed_by_its_first_working_word(status):
    leg = placed_stop()
    leg.placed(2, Outcome.ACCEPTED, "S1", 50)
    accepted = leg.orders[2].state

    leg.heard(2, word(status))

    assert (accepted, leg.orders[2].state, leg.orders[2].bound_ns) == (
        Sent.BOUND,
        Sent.CONFIRMED,
        50,
    )


def test_a_stop_that_ends_is_no_longer_the_legs_stop_and_its_cancel_waits_no_more():
    leg = placed_stop()
    leg.placed(2, Outcome.ACCEPTED, "S1", 50)
    leg.orders[2].cancel_sent = True

    leg.heard(2, word(OrderStatus.CANCELLED))

    assert (leg.stop, leg.orders[2].state, leg.orders[2].cancel_sent) == (None, Sent.ENDED, False)
    assert leg.orders[2].ended is OrderStatus.CANCELLED


@pytest.mark.parametrize(
    ("reason", "state"),
    [
        (NotSentReason.HELD, Sent.HELD),
        (NotSentReason.SILENT, Sent.HELD),
        (NotSentReason.REFUSED, Sent.ENDED),
        (NotSentReason.LOCKED, Sent.HANDED),
    ],
    ids=["held", "silent", "refused", "locked"],
)
def test_a_placement_not_sent_is_held_or_ended_by_why(reason, state):
    leg = placed_stop()
    request = PlaceRequest(1, 2, intent(Purpose.STOP), Priority.PROTECT)

    leg.not_sent(request, reason)

    assert leg.orders[2].state is state


def test_an_add_unanswered_on_a_guarded_leg_counts_its_units_until_the_book_settles_it():
    leg = placed_stop()
    leg.handed(PlaceRequest(3, 4, intent(Purpose.ADD, 5), Priority.ENTER))

    leg.placed(4, Outcome.UNANSWERED, "", 60)
    counted = leg.unbound_add_units
    leg.resolved(4, Resolved.BOUND, "A1", 70)

    assert (counted, leg.unbound_add_units, leg.orders[4].state) == (5, 0, Sent.BOUND)


def test_an_add_unanswered_and_not_placed_counts_no_units_after():
    leg = placed_stop()
    leg.handed(PlaceRequest(3, 4, intent(Purpose.ADD, 5), Priority.ENTER))
    leg.placed(4, Outcome.UNANSWERED, "", 60)

    leg.resolved(4, Resolved.NOT_PLACED, "", 70)

    assert (leg.unbound_add_units, leg.orders[4].state) == (0, Sent.ENDED)


def test_a_flat_leg_is_no_longer_guarded():
    leg = placed_stop()

    leg.flat()

    assert (leg.guarded, leg.unbound_add_units) == (False, 0)
