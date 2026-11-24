"""A held cancel on an instrument two strategies of the session share, decided by the
group's two-book read (`_live/_stops_held_group.py`, `_live/_stops_held.py`;
18-multi-strategy 4.5, the lead's decision M1, unit M7), on a stand-in session.

"A" holds 10 SBIN with its stop "SA" (kernel order 2, 10 units), whose cancel the order
path holds for the trade book; "B" holds 10 with its confirmed resting stop "SB" (kernel
order 2). The read takes the position book, then the order book. Expected values are
worked by hand: signed units, long above 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from session_samples import NS

from openalgo.backtest._live._order_state import HeldFill
from openalgo.backtest._live._stops_held import HELD_CANCEL_DECIDE_AFTER_S, HeldWords
from openalgo.backtest._live._stops_held_group import decide_shared, unheard
from openalgo.backtest._live._stops_orders import Sent, Watched
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._worker_queue import ReadPositions, ReleaseCancel
from openalgo.backtest._live.inbox import PositionRows

#: When the held cancel's wait is over, and when its read was taken (half a second later).
DUE_NS = round(HELD_CANCEL_DECIDE_AFTER_S * NS)
READ_NS = DUE_NS + NS // 2


class Member:
    """A member holding `units` SBIN, its stop `stop_id` (kernel order 2, 10 units) with
    `filled` units its engine has taken; `working` its kernel's working orders."""

    symbol, exchange, product = "SBIN", "NSE", "MIS"

    def __init__(self, name: str, stop_id: str, units: int = 10, filled: int = 0):
        self.name, self.stop_id, self.units, self.filled_units = name, stop_id, units, filled
        self.finished = False
        self.working_orders = [2]
        self.changes: set[int] = set()
        self.venue = self
        self.engine = self

    # The kernel session.
    def open_units(self) -> float:
        return float(self.units)

    def working(self) -> list[int]:
        return list(self.working_orders)

    def order(self, kernel_order: int) -> dict:
        return {"quantity_units": 10, "filled_units": self.filled_units, "trigger_rupees": 497.0}

    # The venue map.
    def kernel_order_of(self, order_id: str) -> int | None:
        return 2 if order_id == self.stop_id else None

    def order_id_of(self, kernel_order: int) -> str:
        return self.stop_id if kernel_order == 2 else ""

    def intent_of(self, kernel_order: int) -> SimpleNamespace:
        return SimpleNamespace(action="SELL")

    def changing(self, kernel_order: int) -> bool:
        return kernel_order in self.changes

    def filled(self, kernel_order: int) -> int:
        return self.filled_units


def confirmed_stop(member: Member) -> SimpleNamespace:
    """The stops supervisor's leg of `member`: its stop confirmed working."""
    watched = Watched(2, Purpose.STOP, 10, "i", "SELL", Sent.CONFIRMED, member.stop_id)
    return SimpleNamespace(stop=2, orders={2: watched})


class Shared:
    """A session of A and B on SBIN: the held words read its groups, its stops supervisor's
    legs, its last tick, and the requests handed to the order worker."""

    def __init__(self, a: Member, b: Member):
        self.traders = (a, b)
        self.legs = {a.name: confirmed_stop(a), b.name: confirmed_stop(b)}
        self.groups = SimpleNamespace(
            shared=lambda trader: True, members=lambda trader: self.traders
        )
        self.stops = SimpleNamespace(leg=lambda trader: self.legs[trader.name])
        self.last_tick: dict = {}
        self.handed: list = []
        self.alerts: list[str] = []

    def hand_over(self, requests):
        self.handed.extend(requests)

    def event(self, kind, **fields):
        pass

    def alert(self, kind, text):
        self.alerts.append(text)


def books(net: int, stop_b: dict | None = None, *, orders: bool = True) -> PositionRows:
    """A two-book read at `READ_NS`: SBIN's net, and B's stop's row (working unless
    `stop_b` says otherwise); `orders` False for a read without the order book."""
    positions = {"status": "success", "data": [_position(net)]}
    row = {
        "orderid": "SB",
        "symbol": "SBIN",
        "exchange": "NSE",
        "product": "MIS",
        "action": "SELL",
        "quantity": 10,
        "pricetype": "SL-M",
        "trigger_price": 497.0,
        "order_status": "trigger pending",
        **(stop_b or {}),
    }
    orderbook = {"status": "success", "data": {"orders": [row]}} if orders else None
    return PositionRows(positions, READ_NS, orderbook)


def _position(net: int) -> dict:
    return {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": net}


def held_cancel_of_a(net: int, stop_b: dict | None = None, **options):
    """A's held cancel past its wait, then the read: the session and what it released."""
    a, b = Member("A", "SA"), Member("B", "SB")
    for name, value in options.items():
        setattr(b, name, value)
    session = Shared(a, b)
    held = HeldWords(session)
    held.update((HeldFill("SA", 0, 0),))
    held.due(DUE_NS)
    held.positions(books(net, stop_b))
    released = [
        (r.order_id, r.filled_quantity) for r in session.handed if isinstance(r, ReleaseCancel)
    ]
    return session, released


# ----- the decision ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("net", "group", "filled", "action", "decided"),
    [
        (20, 20, 0, "SELL", 0),
        (15, 20, 0, "SELL", 5),
        (10, 20, 0, "SELL", 10),
        (8, 20, 0, "SELL", None),
        (25, 20, 0, "SELL", None),
        (16, 20, 4, "SELL", 8),
        (-15, -20, 0, "BUY", 5),
        (None, 20, 0, "SELL", None),
    ],
    ids=[
        "the net is the sum: nothing filled",
        "5 fewer: the order filled 5",
        "all its units",
        "more than its units: not told",
        "more at the broker: not told",
        "4 known, 4 more",
        "a buy stop of a short group",
        "the position book unreadable",
    ],
)
def test_a_held_cancel_is_decided_against_the_members_sum(net, group, filled, action, decided):
    order = {"quantity_units": 10, "filled_units": filled}
    assert decide_shared(net, group, order, action) == decided


def test_a_shared_cancel_asks_for_the_order_book_after_the_position_book():
    session, _ = held_cancel_of_a(20)
    assert ReadPositions(orders=True) in session.handed


def test_the_two_book_read_releases_a_cancel_that_filled_nothing():
    session, released = held_cancel_of_a(20)
    assert released == [("SA", 0)]
    assert session.alerts == []


def test_the_two_book_read_releases_the_units_the_order_filled_before_its_cancel():
    _, released = held_cancel_of_a(15)
    assert released == [("SA", 5)]


@pytest.mark.parametrize(
    "stop_b",
    [{"order_status": "complete"}, {"order_status": "cancelled"}, {"filled_quantity": 4}],
    ids=["filled whole", "ended", "a book that counts a fill"],
)
def test_a_fill_of_another_members_stop_its_engine_lacks_stops_the_decision(stop_b):
    # B's stop sold 4 (or all) at the broker, unheard: the net counts it, so A's cancel
    # would be read as having filled B's units.
    session, released = held_cancel_of_a(16, stop_b)
    assert released == []
    (text,) = session.alerts
    assert "stop SA of A cancelled, its fills not confirmed (the stop SB of B shows a fill" in text
    assert "no exit of A is sent until the books tell" in text


def test_another_members_stop_shown_open_with_no_count_of_its_fills_stops_the_decision():
    # A live order book shows a stop that has triggered as open, and counts no fills: B's
    # stop may have sold 4 unheard, which the net counts, so A's cancel is not decided.
    session, released = held_cancel_of_a(16, {"order_status": "open"})
    assert released == []
    (text,) = session.alerts
    assert (
        "stop SA of A cancelled, its fills not confirmed (the order book shows the stop SB of B "
        "open and does not count its fills)"
    ) in text


@pytest.mark.parametrize(
    ("stop_b", "units", "filled"),
    [
        ({"order_status": "trigger pending"}, 10, 0),
        ({"order_status": "open", "filled_quantity": 0}, 10, 0),
        ({"order_status": "open", "filled_quantity": 4}, 6, 4),
    ],
    ids=["not triggered", "triggered, the book counts no fill", "triggered, its fill heard"],
)
def test_another_members_stop_the_book_shows_with_no_unheard_fill_lets_the_decision_go(
    stop_b, units, filled
):
    _, released = held_cancel_of_a(10 + units, stop_b, units=units, filled_units=filled)
    assert released == [("SA", 0)]


def test_a_fill_of_another_members_stop_its_engine_holds_does_not_stop_the_decision():
    # B's engine has the 4 units its stop filled: the members' sum is 16, and the net too.
    session, released = held_cancel_of_a(16, {"filled_quantity": 4}, units=6, filled_units=4)
    assert released == [("SA", 0)]


def test_another_members_order_in_flight_stops_the_decision():
    a, b = Member("A", "SA"), Member("B", "SB")
    session = Shared(a, b)
    session.legs["B"].orders[3] = Watched(3, Purpose.ENTRY, 5, "j", "BUY", Sent.UNANSWERED)
    assert unheard(session, a, books(20).orderbook) == "an order of B is in flight"


def test_another_members_working_order_but_its_resting_stop_stops_the_decision():
    a, b = Member("A", "SA"), Member("B", "SB")
    b.working_orders = [2, 3]
    session = Shared(a, b)
    assert unheard(session, a, books(20).orderbook) == "order o3 of B works and may be filling"


def test_a_change_pending_on_another_members_stop_stops_the_decision():
    a, b = Member("A", "SA"), Member("B", "SB")
    b.changes = {2}
    assert unheard(Shared(a, b), a, books(20).orderbook) == "the stop of B has a change pending"


def test_another_members_stop_missing_from_the_order_book_stops_the_decision():
    a, b = Member("A", "SA"), Member("B", "SB")
    empty = {"status": "success", "data": {"orders": []}}
    assert unheard(Shared(a, b), a, empty) == "the order book does not show the stop SB of B"


def test_an_order_book_that_cannot_be_read_stops_the_decision():
    a, b = Member("A", "SA"), Member("B", "SB")
    failed = {"status": "error", "message": "broker down"}
    assert unheard(Shared(a, b), a, failed) == (
        "the order book could not be read after the position book"
    )


def test_a_read_without_the_order_book_decides_no_shared_cancel():
    # A read another reader asked for (the kill switch's) carries the position book only.
    a, b = Member("A", "SA"), Member("B", "SB")
    session = Shared(a, b)
    held = HeldWords(session)
    held.update((HeldFill("SA", 0, 0),))
    held.positions(books(20, orders=False))
    assert [r for r in session.handed if isinstance(r, ReleaseCancel)] == []
    assert session.alerts == []


def test_another_held_word_of_the_group_keeps_the_cancel_from_being_decided():
    # B's fill held for its price is in the net and not in B's engine.
    a, b = Member("A", "SA"), Member("B", "SB")
    session = Shared(a, b)
    held = HeldWords(session)
    held.update((HeldFill("SA", 0, 0), HeldFill("SB", 4, 0)))
    held.due(DUE_NS)
    held.positions(books(16))
    assert [r for r in session.handed if isinstance(r, ReleaseCancel)] == []


def test_a_finished_member_counts_no_unheard_fill():
    a, b = Member("A", "SA"), Member("B", "SB", units=0)
    b.finished = True
    assert unheard(Shared(a, b), a, books(10).orderbook) is None
