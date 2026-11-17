"""The exit guard's rules on a stand-in session (`_live/_reconcile_guard.py`;
16-session-design 5.3, 5.4 and 3.3 step 5 "As built", unit 12): what it holds back while a
leg's broker holds fewer units than the engine, or while a leg waits for a book read, what
it lets go once the books agree, and the little it lets go once the session has printed
its last lines: what only takes exposure away.

The stand-in session holds one strategy, `S`, long `ENGINE_UNITS` SBIN with a leg stop
(kernel order `STOP`) working; it records what reaches the order worker and what is
refused. Expected values come from the rules.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._reconcile_guard import ExitGuard
from openalgo.backtest._live._types import OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import (
    CancelRequest,
    ModifyRequest,
    OrderTerms,
    PlaceRequest,
    Priority,
)
from openalgo.backtest._live._worker_queue import CancelOrphan, ReadBooks
from openalgo.backtest._live.inbox import NotSent

#: The engine's position, signed units, and the broker's when it holds fewer.
ENGINE_UNITS = 10
BROKER_UNITS = 4
#: The kernel orders: the leg's stop, an exit, an entry.
STOP, EXIT, ENTRY = 2, 3, 4
#: When a read is asked for, and reads begun before and after it, UTC nanoseconds.
ASKED_NS = 1_000
BEFORE_NS, AFTER_NS = 900, 1_100


class Trader:
    """One strategy's engine as the guard sees it; its stop's terms at the broker are
    `stop_units` units of an SL-M at 497.00, none filled; `withdrawn` the cancels its venue
    map was told were withdrawn, never sent."""

    name = "S"

    def __init__(self) -> None:
        self.units = ENGINE_UNITS
        self.stop_units = ENGINE_UNITS
        self.orders = {STOP, EXIT, ENTRY}
        self.locks: list[str] = []
        self.withdrawn: list[CancelRequest] = []
        self.engine = SimpleNamespace(entry_locks=lambda: list(self.locks))
        self.venue = SimpleNamespace(
            changing=lambda kernel_order: False,
            terms=lambda kernel_order: OrderTerms(self.stop_units, "SL-M", trigger_price=497.0),
            filled=lambda kernel_order: 0,
            withdraw_cancel=self.withdraw_cancel,
        )

    def withdraw_cancel(self, request: CancelRequest) -> bool:
        self.withdrawn.append(request)
        return True

    def open_units(self) -> float:
        return float(self.units)

    def working(self) -> list[int]:
        return sorted(self.orders)


class Session:
    """What the guard calls on its session; `handed` what reached the order worker,
    `refused` what was refused as not sent."""

    def __init__(self) -> None:
        self.trader = Trader()
        self.guard = ExitGuard(self)
        self.handed: list[Any] = []
        self.refused: list[NotSent] = []
        self.alerts: list[str] = []
        self.protected: set[str] = set()
        self.stops = SimpleNamespace(
            protection=SimpleNamespace(holds=lambda trader: trader.name in self.protected)
        )
        self.news = SimpleNamespace(take=self.refused.append)

    def trader_named(self, name: str) -> Trader:
        return self.trader

    def event(self, kind: str, **fields: Any) -> None:
        del kind, fields

    def alert(self, kind: Any, text: str) -> None:
        self.alerts.append(text)

    def hand_over(self, requests: list[Any]) -> None:
        self.handed += self.guard.admit(requests)


def intent(purpose: Purpose, action: str, quantity: int) -> OrderIntent:
    return OrderIntent(
        intent_id=f"i-{purpose.value}",
        strategy="S",
        symbol="SBIN",
        exchange="NSE",
        action=action,
        quantity=quantity,
        price_type="MARKET",
        product="MIS",
        purpose=purpose,
        created_ns=0,
    )


def place(kernel_order: int, purpose: Purpose, action: str, quantity: int) -> PlaceRequest:
    return PlaceRequest(
        kernel_order, kernel_order, intent(purpose, action, quantity), Priority.PROTECT
    )


def cancel_of_stop() -> CancelRequest:
    return CancelRequest(
        9, STOP, intent(Purpose.STOP, "SELL", ENGINE_UNITS), "B2", Priority.PROTECT
    )


def stop_modified(units: int) -> ModifyRequest:
    return ModifyRequest(
        8,
        STOP,
        intent(Purpose.STOP, "SELL", ENGINE_UNITS),
        "B2",
        OrderTerms(units, "SL-M", trigger_price=497.0),
        OrderTerms(ENGINE_UNITS, "SL-M", trigger_price=497.0),
        Priority.PROTECT,
    )


@pytest.fixture
def session() -> Session:
    return Session()


@pytest.fixture
def guarded(session: Session) -> Session:
    """The broker's books show 4 of the engine's 10 units."""
    session.guard.open("S", BROKER_UNITS)
    return session


# ----- a guarded leg --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_", "held"),
    [
        (place(EXIT, Purpose.EXIT, "SELL", ENGINE_UNITS), True),
        (place(EXIT, Purpose.SQUARE_OFF, "SELL", ENGINE_UNITS), True),
        (place(EXIT, Purpose.EXIT, "SELL", BROKER_UNITS), False),
        (place(STOP, Purpose.STOP, "SELL", ENGINE_UNITS), True),
        (stop_modified(BROKER_UNITS), False),
        (cancel_of_stop(), True),
        (place(ENTRY, Purpose.ENTRY, "SELL", 20), True),
        (place(ENTRY, Purpose.ADD, "BUY", 5), False),
    ],
    ids=[
        "an exit of the engine's units",
        "a square-off of the engine's units",
        "an exit within what the broker holds",
        "a stop placed for the engine's units",
        "a stop cut down",
        "the cancel of the stop",
        "an entry that reverses the position",
        "an add on the position's side",
    ],
)
def test_while_the_broker_holds_fewer_units_the_guard_holds(guarded, request_, held):
    guarded.hand_over([request_])
    assert (guarded.handed == []) is held


def test_the_cancel_of_the_stop_of_a_flat_engine_goes(guarded):
    # A stop on a flat position could open one: its cancel is never held.
    guarded.trader.units = 0
    guarded.hand_over([cancel_of_stop()])
    assert guarded.handed == [cancel_of_stop()]


def test_the_reconcilers_own_cancel_of_the_stop_goes_a_held_one_at_once(guarded):
    guarded.hand_over([cancel_of_stop()])
    guarded.guard.allow_cancel("S", STOP)
    assert guarded.handed == [cancel_of_stop()]


def test_a_held_cancel_of_the_stop_is_alerted_as_keeping_the_stop(guarded):
    guarded.hand_over([cancel_of_stop()])
    [text] = guarded.alerts
    assert "the cancel of the stop of SBIN NSE (SELL 10) is held back" in text
    assert "the stop stays at the broker" in text


def test_a_held_cancel_of_the_stop_is_alerted_without_promising_the_exit(guarded):
    # Once the books agree neither is sure to go: protect by state keeps the cancel of a
    # stop that still fits and asks for no exit, and after the session's last lines the
    # exit is held.
    guarded.hand_over([cancel_of_stop()])
    [text] = guarded.alerts
    assert "Both go once the books agree" not in text
    assert "go only if the session still asks for them then" in text


# ----- when the books agree -------------------------------------------------------------


def test_once_the_books_agree_the_held_requests_go_in_order(guarded):
    held = [cancel_of_stop(), place(EXIT, Purpose.EXIT, "SELL", ENGINE_UNITS)]
    guarded.hand_over(held)
    guarded.guard.close("S")
    assert guarded.handed == held


def test_a_held_entry_is_refused_once_entries_are_locked(guarded):
    guarded.hand_over([place(ENTRY, Purpose.ENTRY, "SELL", 20)])
    guarded.trader.locks.append("reconciliation")
    guarded.guard.close("S")
    assert guarded.handed == []
    assert [message.request.kernel_order for message in guarded.refused] == [ENTRY]


def test_a_held_exit_beyond_what_the_engine_holds_by_then_is_refused(guarded):
    guarded.hand_over([place(EXIT, Purpose.EXIT, "SELL", ENGINE_UNITS)])
    guarded.trader.units = BROKER_UNITS
    guarded.guard.close("S")
    assert [message.request.kernel_order for message in guarded.refused] == [EXIT]


def test_once_the_session_has_ended_books_that_agree_let_nothing_go(guarded):
    # The session's last lines named the held orders as not sent: they stay so.
    held = [cancel_of_stop(), place(EXIT, Purpose.EXIT, "SELL", ENGINE_UNITS)]
    guarded.hand_over(held)
    guarded.guard.end()
    guarded.guard.close("S")
    assert (guarded.handed, guarded.refused) == ([], [])
    assert guarded.guard.left() == held


def test_the_held_cancel_of_a_leg_protect_by_state_holds_stays_held(guarded):
    # Protect by state asks for no exit of the leg: its stop stays at the broker.
    guarded.hand_over([cancel_of_stop()])
    guarded.protected.add("S")
    guarded.guard.close("S")
    assert guarded.handed == []
    assert guarded.guard.held_cancels("S") == {STOP}


def test_the_held_cancel_of_a_stop_larger_than_the_position_goes_under_protect_by_state(guarded):
    # The engine holds 4 of the stop's 10 units by the time the books agree: a trigger
    # would sell 6 beyond the position, so the cancel goes, and protect by state places a
    # stop for what is left once it is cancelled.
    guarded.hand_over([cancel_of_stop()])
    guarded.protected.add("S")
    guarded.trader.units = BROKER_UNITS
    guarded.guard.close("S")
    assert guarded.handed == [cancel_of_stop()]


# ----- a sale by hand adopted --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ended", "engine_units"),
    [(False, 0), (False, BROKER_UNITS), (True, 0)],
    ids=["before the end, nothing left", "before the end, 4 left", "after the end, nothing left"],
)
def test_an_adoption_sends_the_held_cancel_of_a_stop_larger_than_what_is_left(
    guarded, ended, engine_units
):
    guarded.hand_over([cancel_of_stop()])
    if ended:
        guarded.guard.end()
    guarded.trader.units = engine_units
    assert guarded.guard.withdraw_for_cut("S", engine_units) is False
    sent = guarded.guard.adopted("S")
    assert (sent, guarded.handed) == (True, [cancel_of_stop()])
    assert guarded.guard.held_cancels("S") == frozenset()


def test_after_the_end_an_adoption_that_leaves_units_withdraws_the_held_cancel_for_a_cut(guarded):
    # The cancel is never sent: the venue map tells the kernel it was refused, so the
    # kernel's fit with the adoption cuts the stop to the 4 units left.
    guarded.hand_over([cancel_of_stop()])
    guarded.guard.end()
    assert guarded.guard.withdraw_for_cut("S", BROKER_UNITS) is True
    assert (guarded.handed, guarded.trader.withdrawn) == ([], [cancel_of_stop()])
    assert guarded.guard.held_cancels("S") == frozenset()
    guarded.trader.units = BROKER_UNITS
    assert guarded.guard.adopted("S") is False


def test_after_the_end_the_stops_cut_to_what_is_left_goes(guarded):
    guarded.guard.end()
    guarded.hand_over([stop_modified(BROKER_UNITS)])
    assert guarded.handed == [stop_modified(BROKER_UNITS)]


def test_after_the_end_a_held_cancel_of_a_stop_that_still_fits_is_not_withdrawn(guarded):
    guarded.hand_over([cancel_of_stop()])
    guarded.guard.end()
    guarded.trader.stop_units = BROKER_UNITS
    assert guarded.guard.withdraw_for_cut("S", BROKER_UNITS) is False
    assert (guarded.trader.withdrawn, guarded.guard.held_cancels("S")) == ([], frozenset({STOP}))


def test_an_adoption_keeps_the_held_cancel_of_a_stop_that_still_fits(guarded):
    # The stop covers 4 units and the engine holds 4 once the sale is adopted: it fits,
    # and its cancel waits for the books as before.
    guarded.hand_over([cancel_of_stop()])
    guarded.trader.stop_units = guarded.trader.units = BROKER_UNITS
    assert guarded.guard.adopted("S") is False
    assert guarded.handed == []


# ----- after the session's last lines ---------------------------------------------------


def stop_trailed() -> ModifyRequest:
    """The stop raised to 499.00 for the same units (a trail)."""
    return ModifyRequest(
        7,
        STOP,
        intent(Purpose.STOP, "SELL", ENGINE_UNITS),
        "B2",
        OrderTerms(ENGINE_UNITS, "SL-M", trigger_price=499.0),
        OrderTerms(ENGINE_UNITS, "SL-M", trigger_price=497.0),
        Priority.PROTECT,
    )


@pytest.fixture
def ended(session: Session) -> Session:
    """The session has printed its last lines; the books agree with the engine."""
    session.guard.end()
    return session


def stop_cut_and_trailed() -> ModifyRequest:
    """The stop cut down to 4 units and raised to 499.00."""
    return ModifyRequest(
        6,
        STOP,
        intent(Purpose.STOP, "SELL", ENGINE_UNITS),
        "B2",
        OrderTerms(BROKER_UNITS, "SL-M", trigger_price=499.0),
        OrderTerms(ENGINE_UNITS, "SL-M", trigger_price=497.0),
        Priority.PROTECT,
    )


def cancel_of(kernel_order: int, purpose: Purpose, action: str) -> CancelRequest:
    return CancelRequest(
        kernel_order, kernel_order, intent(purpose, action, 5), f"B{kernel_order}", Priority.ENTER
    )


@pytest.mark.parametrize(
    "request_",
    [
        place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS),
        cancel_of_stop(),
        stop_trailed(),
        stop_cut_and_trailed(),
        place(STOP, Purpose.STOP, "SELL", ENGINE_UNITS),
        place(ENTRY, Purpose.ENTRY, "BUY", 5),
        cancel_of(EXIT, Purpose.EXIT, "SELL"),
    ],
    ids=[
        "a target's exit",
        "the cancel of the stop of a position",
        "a trail of the stop",
        "a stop cut down and moved",
        "a stop placed again",
        "an entry",
        "the cancel of an exit",
    ],
)
def test_after_the_last_lines_a_new_order_request_is_not_sent(ended, request_):
    ended.hand_over([request_])
    assert (ended.handed, ended.refused) == ([], [])


@pytest.mark.parametrize(
    ("request_", "engine_units"),
    [
        (stop_modified(BROKER_UNITS), BROKER_UNITS),
        (cancel_of_stop(), 0),
        (cancel_of(ENTRY, Purpose.ENTRY, "BUY"), ENGINE_UNITS),
        (cancel_of(ENTRY, Purpose.ADD, "BUY"), ENGINE_UNITS),
    ],
    ids=[
        "the stop cut down to what the engine holds",
        "the cancel of the stop of a flat engine",
        "the cancel of an entry",
        "the cancel of an add",
    ],
)
def test_after_the_last_lines_what_only_takes_exposure_away_goes(ended, request_, engine_units):
    # A sale by hand the final read adopts fits the stop to what is left; the safe stop
    # asked before the lines for the cancel of every entry and add.
    ended.trader.units = engine_units
    ended.hand_over([request_])
    assert ended.handed == [request_]


def test_after_the_last_lines_a_request_not_sent_is_left_for_the_close_to_log(ended):
    ended.hand_over([cancel_of_stop()])
    assert ended.guard.left() == [cancel_of_stop()]


def test_after_the_last_lines_the_reconcilers_own_cancel_of_the_stop_goes(ended):
    # The broker holds the other side: a stop on it could add to that position.
    ended.guard.allow_cancel("S", STOP)
    ended.hand_over([cancel_of_stop()])
    assert ended.handed == [cancel_of_stop()]


@pytest.mark.parametrize(
    "request_",
    [CancelOrphan("B7"), ReadBooks(3)],
    ids=["the cancel of an orphan", "a book read"],
)
def test_after_the_last_lines_the_workers_own_requests_go(ended, request_):
    ended.hand_over([request_])
    assert ended.handed == [request_]


# ----- a leg waiting for a book read ------------------------------------------------------


@pytest.fixture
def awaiting(session: Session) -> Session:
    """The safe stop's flatten asked for a read at `ASKED_NS`."""
    session.guard.await_read("S", ASKED_NS)
    return session


@pytest.mark.parametrize(
    ("request_", "held"),
    [
        (place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS), True),
        (cancel_of_stop(), True),
        (place(STOP, Purpose.STOP, "SELL", ENGINE_UNITS), False),
        (place(ENTRY, Purpose.ENTRY, "SELL", 20), True),
    ],
    ids=["an exit", "the cancel of the stop", "a stop placed again", "a reversing entry"],
)
def test_while_a_leg_waits_for_a_read_the_guard_holds(awaiting, request_, held):
    awaiting.hand_over([request_])
    assert (awaiting.handed == []) is held


def test_a_read_begun_before_the_request_lets_nothing_go(awaiting):
    awaiting.hand_over([place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS)])
    awaiting.guard.read("S", BEFORE_NS)
    assert awaiting.handed == []


def test_a_read_begun_after_the_request_lets_the_exits_go(awaiting):
    exit_ = place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS)
    awaiting.hand_over([exit_])
    awaiting.guard.read("S", AFTER_NS)
    assert awaiting.handed == [exit_]


def test_once_the_session_has_ended_the_read_a_leg_awaited_lets_nothing_go(awaiting):
    exit_ = place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS)
    awaiting.hand_over([exit_])
    awaiting.guard.end()
    awaiting.guard.read("S", AFTER_NS)
    assert (awaiting.handed, awaiting.refused) == ([], [])
    assert awaiting.guard.left() == [exit_]


def test_a_read_that_guards_the_leg_keeps_the_exit_held_and_alerts_it(awaiting):
    awaiting.hand_over([place(EXIT, Purpose.RISK_EXIT, "SELL", ENGINE_UNITS)])
    assert awaiting.alerts == []
    awaiting.guard.open("S", BROKER_UNITS)
    awaiting.guard.read("S", AFTER_NS)
    assert awaiting.handed == []
    assert len(awaiting.alerts) == 1
