"""The exit guard's rules on a stand-in session (`_live/_reconcile_guard.py`;
16-session-design 5.3 and 5.4 "As built", unit 12): what it holds back while a leg's
broker holds fewer units than the engine, or while a leg waits for a book read, and what
it lets go once the books agree.

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
    """One strategy's engine as the guard sees it."""

    name = "S"

    def __init__(self) -> None:
        self.units = ENGINE_UNITS
        self.orders = {STOP, EXIT, ENTRY}
        self.locks: list[str] = []
        self.engine = SimpleNamespace(entry_locks=lambda: list(self.locks))
        self.venue = SimpleNamespace(changing=lambda kernel_order: False)

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
