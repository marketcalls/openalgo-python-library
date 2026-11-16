"""The stops supervisor's parts on their own (`_live/_stops_kernel.py`, `_stops_cover.py`,
`_stops_orders.py`, `stops.py`; 16-session-design 4.2 and 4.6): the marketable limit of
the price rule, the unprotected windows, what the supervisor knows of a leg's orders, and
the deadline of a stop never confirmed under protect by state.

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
from openalgo.backtest._live.stops import UNPROTECTED_EXIT_AFTER_S, StopsSupervisor

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


class _Engine:
    """A kernel session whose leg stop o2 (an SL-M of 10 units) works, no cancel decided."""

    def order(self, kernel_order):
        return {"order_type": "SL-M", "quantity_units": 10, "filled_units": 0}

    def live_state(self):
        return {"desk": {"cancelling": []}}


class _Venue:
    def order_id_of(self, kernel_order):
        return "S1"


class _Trader:
    """A strategy holding 10 SBIN with its stop o2 working."""

    name, symbol, exchange, finished = "ema_cross", "SBIN", "NSE", False

    def __init__(self):
        self.engine, self.venue = _Engine(), _Venue()
        self.exits_asked: list[str] = []

    def open_units(self):
        return 10.0

    def working(self):
        return [2]

    def exit_orders(self):
        return [2]

    def request_exit(self, reason):
        self.exits_asked.append(reason)
        return "waits_for_cancels"


class _Worker:
    busy_since_ns, raw_pending = None, 0


class _Exits:
    """No exit of the session's retry schedule."""

    def halt(self, strategy=None):
        pass

    def owns(self, name):
        return False


class _Session:
    """A session of one strategy, its order worker idle; `clock` is its time."""

    locked, inbox_waiting, control = False, 0, None

    def __init__(self):
        self.parts = type("Parts", (), {"worker": _Worker()})()
        self.exits = _Exits()
        self.traders = (_Trader(),)
        self.alerts: list[str] = []
        self.clock = 0

    def now(self):
        return self.clock

    def event(self, kind, **fields):
        pass

    def alert(self, kind, text):
        self.alerts.append(text)


def test_the_unconfirmed_stop_deadline_asks_no_exit_of_a_leg_protected_by_state():
    session = _Session()
    (trader,) = session.traders
    stops = StopsSupervisor(session)
    stops.handed([PlaceRequest(1, 2, intent(Purpose.STOP), Priority.PROTECT)])
    stops.leg(trader).placed(2, Outcome.ACCEPTED, "S1", 0)
    stops.protection.begin([trader], 0)

    session.clock = round((UNPROTECTED_EXIT_AFTER_S + 1) * 10**9)
    stops.duties(session.clock)

    assert trader.exits_asked == []
    assert any(
        "not confirmed" in text and "check the order book" in text for text in session.alerts
    )
