"""What the operator is told of a position left open, on the fake OpenAlgo
(`_live/_stops_report.py`; 16-session-design 1.4 `open_positions`, 3.3 step 4, 3.5; the
unit 8 hand-off on the left-open note): each order still working toward the position
named by its broker order ID, else its side, units, purpose and intent; an order sent and
unanswered, or a market order accepted with no final word, told from one resting at the
broker; an entry still working named too; the stop, confirmed (and then that it protects
the position until it is cancelled), repriced to a resting LIMIT, not sent, or not
confirmed; and, while any order works or a stop may, the order book before closing it by
hand. A leg its confirmed stop covers is printed, and only the others are ERROR alerts.
The note of an alert sent while the session runs says to end it before any close by hand.

The session starts at 14:40:05 IST; SBIN trades at 500.00 every 10 seconds.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from session_trace import BOOKS_AGREE
from stops_samples import OPTION, STOP_POINTS, OptionHold, StopHold, option_world, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._order_answers import Outcome
from openalgo.backtest._live._stops_report import (
    CHECK_FIRST,
    STOP_PROTECTS,
    left_open,
    open_while_running,
    protected,
    stop_state,
    stop_words,
)
from openalgo.backtest._live._types import (
    LOGGER_NAME,
    Mode,
    OrderIntent,
    OrderStatus,
    OrderUpdate,
    Purpose,
)
from openalgo.backtest._live._venue_requests import OrderTerms, PlaceRequest, Priority
from openalgo.backtest._live.inbox import NotSentReason
from openalgo.backtest._live.stops import StopsSupervisor

#: The session of the supervisors built with no server: its reconciler's exit guard holds
#: no cancel back, no leg's books differ from the engine's, and each strategy is alone on
#: its instrument, which is all they ask of it (`StopsSupervisor.cancelling`,
#: `StopsSupervisor.broker_units`, `_stops_held_day.DayHolds.holder`).
NOTHING_HELD = SimpleNamespace(
    reconcile=SimpleNamespace(
        guard=SimpleNamespace(held_cancels=lambda name: frozenset()),
        broker_units=lambda name: None,
    ),
    groups=SimpleNamespace(members=lambda trader: (trader,)),
)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


class RestsAnExit(bt.Strategy):
    """Buys 10 SBIN at its first decision (no stop) and rests a LIMIT sale of them at 600
    at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2:
            self.sell(10, price=600.0)


def test_a_resting_sale_is_named_as_resting_and_the_order_book_comes_first(world):
    # The safe stop cancels the resting sale (not reduce-only, it is classed with the
    # entries), and the cancel never reaches the broker: the sale rests on.
    world.script.at(
        ist(14, 44) - NS // 2, lambda: world.fake.faults.bad_gateway_page("cancelorder")
    )
    world.stop_at(ist(14, 44))
    outcome = world.run(RestsAnExit, on_stop="keep")

    resting = next(o for o in world.orders() if o.action == "SELL")
    (left,) = outcome.open_positions
    assert resting.status == "open"
    assert f"order {resting.order_id} resting at the broker" in left.note
    assert "in flight" not in left.note
    # A sale by hand beside the resting one would sell short: never a bare "close it".
    assert left.note.endswith(CHECK_FIRST)


class AddsResting(bt.Strategy):
    """Buys 10 SBIN with a stop at its first decision and rests a LIMIT add of 5 at 400 at
    its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.buy(5, price=400.0, stop=bt.points(STOP_POINTS))


def test_an_entry_still_working_is_named_with_the_confirmed_stop(world):
    # The cancel of the resting add never reaches the broker: the add still works.
    world.script.at(
        ist(14, 44) - NS // 2, lambda: world.fake.faults.bad_gateway_page("cancelorder")
    )
    world.stop_at(ist(14, 44))
    outcome = world.run(AddsResting, on_stop="keep")

    add = next(o for o in world.orders() if (o.action, o.price_type) == ("BUY", "LIMIT"))
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    (left,) = outcome.open_positions
    assert add.status == "open"
    assert f"entry order {add.order_id} resting at the broker" in left.note
    assert f"stop {stop.order_id} confirmed at the broker" in left.note


def test_an_exit_sent_and_unanswered_is_named_by_side_units_and_intent(world):
    class Holds(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.calls = 0

        def on_bar(self, bar):
            self.calls += 1
            if self.calls == 1:
                self.buy(10, stop=bt.points(STOP_POINTS))

    # The safe stop's exit never reaches the broker and its answer is lost. A second Ctrl+C
    # ends the session with the leg mid exit, named by its line (3.3 step 4), which the
    # session invariants take instead of a confirmed stop.
    world.trace.left_unprotected_by_decision(
        "Holds", "3.3 step 4: a second Ctrl+C with the exit unanswered"
    )
    world.script.at(
        ist(14, 43, 31) + NS // 2, lambda: world.fake.faults.bad_gateway_page("placeorder")
    )
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 34))
    outcome = world.run(Holds)

    (left,) = outcome.open_positions
    assert "exit order SELL 10 (risk_exit, intent 20261116-live-1-Holds-o3)" in left.note
    assert "in flight, sent and unanswered" in left.note
    assert "check the order book before closing it by hand" in left.note


def test_an_exit_accepted_with_no_final_word_is_in_flight_and_never_beside_close_it_by_hand(
    world,
):
    def stream_held_and_reads_failing():
        world.fake.faults.hold_order_updates()
        for _ in range(20):
            world.fake.faults.gateway_error("orderstatus", status=503)
            world.fake.faults.gateway_error("tradebook", status=503)

    # The flatten's book read is compared at 14:43:31, its cancel of the stop answered at
    # 14:43:33 and its exit accepted at 14:43:34; the stream then stops. A second Ctrl+C
    # ends the session with the leg mid exit (3.3 step 4); the exit fills at the broker
    # unheard, and with the books unreadable no comparison names the difference: the
    # leg's line names the exit in flight instead.
    world.trace.waive(BOOKS_AGREE, "3.3 step 4: an exit filled unheard, the books unreadable")
    world.stop_at(ist(14, 43, 30))
    world.script.at(ist(14, 43, 33) + NS // 2, stream_held_and_reads_failing)
    world.stop_at(ist(14, 43, 34) + NS // 2)
    outcome = world.run(StopHold)

    exit_ = next(o for o in world.orders() if (o.action, o.price_type) == ("SELL", "MARKET"))
    (left,) = outcome.open_positions
    assert f"exit order {exit_.order_id} in flight, accepted with no final word" in left.note
    assert "resting at the broker" not in left.note
    assert left.note.endswith(CHECK_FIRST)
    assert "close it by hand" not in left.note.replace(CHECK_FIRST, "")


@pytest.fixture
def option(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return option_world(make_fake_openalgo, monkeypatch, tmp_path)


def test_a_stop_repriced_to_a_limit_that_rests_is_named_as_resting_with_its_price(option):
    # The option trades through its stop at 100.00 at 14:44:05; the best bid then drops
    # to 90.00, so the stop's reprice to a LIMIT at 95.00 rests unfilled at the broker. The
    # leg is left with that LIMIT, no stop, named by its line (4.6), which the session
    # invariants take instead of a confirmed stop.
    option.trace.left_unprotected_by_decision(
        "OptionHold", "4.6: a stop repriced to a resting LIMIT"
    )
    crossed_at = ist(14, 44, 5)
    option.script.at(crossed_at - NS // 2, lambda: setattr(option, "option_price", 100.0))
    option.script.at(
        crossed_at + NS // 2, lambda: option.fake.trade(OPTION, "NFO", 100.0, bid=90.0)
    )
    option.stop_at(crossed_at + 3 * NS)
    outcome = option.run(OptionHold, on_stop="keep")

    stop = option.orders()[1]
    assert (stop.price_type, stop.price, stop.filled, stop.status) == ("LIMIT", 95.0, 0, "open")
    (left,) = outcome.open_positions
    assert f"stop {stop.order_id} repriced to a LIMIT at 95.00, resting at the broker" in left.note
    # A sale by hand beside the resting LIMIT sale of the whole position would sell short.
    assert left.note.endswith(CHECK_FIRST)


def test_a_leg_left_with_its_confirmed_stop_is_printed_and_not_alerted_as_an_error(world, caplog):
    world.stop_at(ist(14, 43, 30))
    world.run(StopHold, on_stop="keep")

    stop = world.orders()[1]
    (line,) = [line for line in world.lines if "StopHold SBIN NSE 10 units" in line]
    assert f"stop {stop.order_id} confirmed at the broker" in line
    assert STOP_PROTECTS in line
    # Design 3.3 step 4: only a leg still without a confirmed stop is the operator's.
    assert not any("SBIN NSE 10 units" in text for text in alerts(caplog, "error"))


class TargetOnly(StopHold):
    """Buys 10 SBIN at its first decision with a target 2 points above and no stop."""

    rules = {"target": bt.points(2)}


def test_a_stops_lines_are_taken_after_its_step_hands_over_the_exit_a_tick_made(world, caplog):
    # SBIN trades at the target (502.00) at the instant of the Ctrl+C: the loop step that
    # sees the stop takes the tick first, so the kernel makes the target's exit in the
    # step whose wait ends at once (no stop to protect, no entry to cancel).
    world.script.at(ist(14, 43, 30), lambda: world.fake.trade("SBIN", "NSE", 502.0, volume=10.0))
    world.stop_at(ist(14, 43, 30))
    world.run(TargetOnly, on_stop="keep")

    (line,) = [line for line in world.lines if line.startswith("TargetOnly SBIN NSE 10 units")]
    assert "exit order SELL 10" in line
    assert line.endswith(CHECK_FIRST)
    assert "close it by hand" not in line.replace(CHECK_FIRST, "")


# ----- a stop not sent, on its own (no server) ----------------------------------------


class _Engine:
    """A kernel session whose leg stop o2 (an SL-M sale of 10 units) works."""

    def order(self, kernel_order):
        return {
            "order_type": "SL-M",
            "side": "SELL",
            "quantity_units": 10,
            "filled_units": 0,
            "limit_rupees": 0.0,
            "trigger_rupees": 497.0,
        }


class _Venue:
    """The venue map of the stop o2: not bound, its terms the broker's (10 units of an SL-M
    at 497.00), none filled."""

    def order_id_of(self, kernel_order):
        return ""

    def terms(self, kernel_order):
        return OrderTerms(10, "SL-M", trigger_price=497.0)

    def filled(self, kernel_order):
        return 0


class _Trader:
    """A strategy holding 10 SBIN whose stop o2 the kernel placed."""

    name, symbol, exchange, finished = "StopHold", "SBIN", "NSE", False
    product = "MIS"

    def __init__(self):
        self.engine, self.venue = _Engine(), _Venue()

    def open_units(self):
        return 10.0

    def working(self):
        return [2]


#: The placement of the stop o2, as the session hands it to the order worker.
STOP_PLACED = PlaceRequest(
    1,
    2,
    OrderIntent(
        intent_id="20261116-live-1-StopHold-o2",
        strategy="StopHold",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        price_type="SL-M",
        product="MIS",
        purpose=Purpose.STOP,
        created_ns=0,
        trigger_price=497.0,
    ),
    Priority.PROTECT,
)


def stop_handed() -> tuple[StopsSupervisor, _Trader]:
    """The stop o2 handed to the order worker, with no answer yet."""
    stops, trader = StopsSupervisor(session=NOTHING_HELD), _Trader()
    stops.leg(trader).handed(STOP_PLACED)
    return stops, trader


def test_a_stop_handed_to_the_worker_and_unanswered_may_not_have_been_sent():
    stops, trader = stop_handed()

    note = left_open(stops, trader)

    assert note.startswith(
        "stop (intent 20261116-live-1-StopHold-o2) handed to the order worker, not "
        "answered: it may not have been sent"
    )
    assert "not confirmed at the broker" not in note
    assert note.endswith(CHECK_FIRST)


def test_a_stop_held_back_while_openalgo_is_silent_is_named_as_not_sent():
    stops, trader = stop_handed()
    stops.leg(trader).not_sent(STOP_PLACED, NotSentReason.SILENT)

    note = left_open(stops, trader)

    assert note.startswith("stop (intent 20261116-live-1-StopHold-o2) held back, not sent")
    assert "not confirmed at the broker" not in note
    assert note.endswith(CHECK_FIRST)


# ----- what works toward a leg, and a stop the kernel is cancelling (no server) ----------


class _Book:
    """A kernel session whose orders are `orders` (kernel order to its terms) and whose
    cancelling list (decided, not yet answered) is `cancelling`."""

    def __init__(self, orders: dict, cancelling: tuple = ()):
        self.orders, self.cancelling = orders, cancelling

    def order(self, kernel_order):
        return self.orders.get(kernel_order)

    def live_state(self):
        return {"desk": {"cancelling": list(self.cancelling)}}


class _Leg:
    """A strategy holding `units` SBIN (10 unless a test changes it) whose working kernel
    orders are `working`."""

    name, symbol, exchange, finished = "StopHold", "SBIN", "NSE", False
    product = "MIS"

    def __init__(self, book: _Book, working: list):
        self.engine, self.venue, self._working = book, _Venue(), working
        self.units = 10

    def open_units(self):
        return float(self.units)

    def working(self):
        return list(self._working)


#: A SELL of 10 SBIN for kernel order `kernel_order`, placed as `purpose` at `price_type`.
def _sale(kernel_order: int, purpose: Purpose, price_type: str) -> PlaceRequest:
    intent = OrderIntent(
        intent_id=f"20261116-live-1-StopHold-o{kernel_order}",
        strategy="StopHold",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        price_type=price_type,
        product="MIS",
        purpose=purpose,
        created_ns=0,
        price=600.0 if price_type == "LIMIT" else 0.0,
        trigger_price=497.0 if price_type == "SL-M" else 0.0,
    )
    return PlaceRequest(kernel_order, kernel_order, intent, Priority.PROTECT)


def _terms(order_type: str) -> dict:
    return {
        "order_type": order_type,
        "side": "SELL",
        "quantity_units": 10,
        "filled_units": 0,
        "limit_rupees": 600.0 if order_type == "LIMIT" else 0.0,
        "trigger_rupees": 497.0 if order_type == "SL-M" else 0.0,
    }


def _bound(stops: StopsSupervisor, trader, request: PlaceRequest, order_id: str) -> None:
    """`request` handed to the worker and accepted as `order_id`."""
    leg = stops.leg(trader)
    leg.handed(request)
    leg.placed(request.kernel_order, Outcome.ACCEPTED, order_id, 0)


def _working_word(order_id: str) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id,
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        filled_quantity=0,
        pending_quantity=10,
        average_price=0.0,
        status=OrderStatus.TRIGGER_PENDING,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=0,
        source="stream",
    )


@pytest.mark.parametrize("unsent", [(), (5,), (5, 6)], ids=["none", "one", "two"])
def test_a_resting_exit_beside_orders_never_sent_comes_before_closing_it_by_hand(unsent):
    # No stop rule; an exit LIMIT rests at the broker; the kernel orders `unsent` were
    # never handed to the order worker.
    stops = StopsSupervisor(session=NOTHING_HELD)
    trader = _Leg(_Book({3: _terms("LIMIT")}), working=[3, *unsent])
    _bound(stops, trader, _sale(3, Purpose.EXIT, "LIMIT"), "261116000000003")

    note = left_open(stops, trader)

    assert note.startswith("exit order 261116000000003 resting at the broker")
    # A sale by hand beside the resting exit would sell short once it fills.
    assert note.endswith(CHECK_FIRST)
    assert "close it by hand" not in note.replace(CHECK_FIRST, "")


#: The session of a supervisor whose reconciler's exit guard holds the leg: its last read
#: showed the broker holding 6 of the engine's 10 units (4 sold by hand, seen once); the
#: strategy alone on its instrument.
BROKER_HOLDS_6 = SimpleNamespace(
    reconcile=SimpleNamespace(
        guard=SimpleNamespace(held_cancels=lambda name: frozenset()),
        broker_units=lambda name: 6,
    ),
    groups=SimpleNamespace(members=lambda trader: (trader,)),
)


def confirmed_stop(cancelling: tuple, session: Any = NOTHING_HELD) -> tuple[StopsSupervisor, _Leg]:
    """A leg whose SL-M stop o2 (order 261116000000002) is confirmed working for its 10
    units, with the kernel's cancelling list `cancelling`, in `session`."""
    stops = StopsSupervisor(session=session)
    trader = _Leg(_Book({2: _terms("SL-M")}, cancelling=cancelling), working=[2])
    _bound(stops, trader, STOP_PLACED, "261116000000002")
    stops.leg(trader).heard(2, _working_word("261116000000002"))
    return stops, trader


def test_a_confirmed_stop_with_nothing_else_working_protects_the_position():
    stops, trader = confirmed_stop(cancelling=())

    assert protected(stops, trader)
    assert left_open(stops, trader).endswith(STOP_PROTECTS)


def test_a_confirmed_stop_larger_than_the_position_is_named_with_its_units_not_as_protecting():
    # 4 of the 10 units were sold by hand and adopted while the stop's cut or cancel
    # waits: a trigger of the stop as the broker holds it would sell 4 beyond the position.
    stops, trader = confirmed_stop(cancelling=())
    trader.units = 6

    note = left_open(stops, trader)

    assert not protected(stops, trader)
    assert note == (
        f"stop 261116000000002 confirmed at the broker for 10 units, more than the 6 held; "
        f"{CHECK_FIRST}"
    )
    assert stop_state(stops, trader) == ("261116000000002", False)


def test_a_stop_the_kernel_is_cancelling_is_named_as_being_cancelled_not_as_protecting():
    # The kernel decided the cancel in this loop step (a target tick, the daily loss):
    # the session hands it to the worker only at the step's end.
    stops, trader = confirmed_stop(cancelling=(2,))

    note = left_open(stops, trader)

    assert not protected(stops, trader)
    assert "stop 261116000000002 being cancelled" in note
    assert STOP_PROTECTS not in note
    assert note.endswith(CHECK_FIRST)
    assert stop_words(stops, trader).endswith("check the order book")
    assert stop_state(stops, trader) == ("261116000000002", False)


def test_a_confirmed_stop_of_a_leg_the_guard_holds_is_named_with_the_books_units():
    # A trigger of the stop for 10 would sell 4 beyond the 6 the broker holds.
    stops, trader = confirmed_stop(cancelling=(), session=BROKER_HOLDS_6)

    assert left_open(stops, trader) == (
        "stop 261116000000002 confirmed at the broker for 10 units; OpenAlgo's books showed 6 "
        f"units at the last read; {CHECK_FIRST}"
    )


def test_a_confirmed_stop_of_a_leg_the_guard_holds_neither_protects_it_nor_is_confirmed():
    stops, trader = confirmed_stop(cancelling=(), session=BROKER_HOLDS_6)

    assert not protected(stops, trader)
    assert stop_state(stops, trader) == ("261116000000002", False)


def test_a_leg_the_guard_holds_with_no_stop_is_sent_to_the_order_book_not_to_close_it():
    # A sale by hand of the engine's 10 would sell 4 beyond the 6 the broker holds.
    stops, trader = StopsSupervisor(session=BROKER_HOLDS_6), _Leg(_Book({}), working=[])

    assert left_open(stops, trader) == (
        f"no stop at the broker; OpenAlgo's books showed 6 units at the last read; {CHECK_FIRST}"
    )


# ----- the note of an alert sent while the session runs (no server) ---------------------

#: What such a note says before any close by hand, and why: the running session may send
#: the exit or the stop again (16-session-design 4.8, 4.10).
END_THE_SESSION = (
    "to close it by hand, end the session first (Ctrl+C, and a second Ctrl+C if it does not end)"
)
WHY = "while the session runs, the engine may send its exit or its stop again"
IF_OPEN = "if its closing lines still show the leg open"


def no_stop() -> tuple[StopsSupervisor, _Leg]:
    """A leg of 10 SBIN that never had a stop, with nothing working."""
    return StopsSupervisor(session=NOTHING_HELD), _Leg(_Book({}), working=[])


@pytest.mark.parametrize(
    ("leg", "note"),
    [
        (
            lambda: confirmed_stop(cancelling=()),
            "stop 261116000000002 confirmed at the broker; "
            f"{END_THE_SESSION}, then, {IF_OPEN}, cancel that stop: {WHY}",
        ),
        (
            lambda: confirmed_stop(cancelling=(2,)),
            "stop 261116000000002 being cancelled, with no final word yet; "
            f"{END_THE_SESSION}, then, {IF_OPEN}, check the order book, cancelling what "
            f"still works: {WHY}",
        ),
        (no_stop, f"no stop at the broker; {END_THE_SESSION}: {WHY}"),
    ],
    ids=["stop confirmed", "stop being cancelled", "no stop"],
)
def test_while_the_session_runs_the_note_says_to_end_it_before_any_close_by_hand(leg, note):
    stops, trader = leg()

    assert open_while_running(stops, trader) == note
