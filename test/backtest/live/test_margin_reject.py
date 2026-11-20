"""A closing order rejected on a shared instrument, the session's part on doubles
(`_live/_margin_reject.py`; 18-multi-strategy 4.14, 11.0a items 1 and 2, 4.14a, unit M6):
which rejections hold a position, the alert's words, and the gate that keeps every
automatic exit and protect of a held position until a book read begun after the
rejection agrees.

The session on the fake OpenAlgo is tested in `test_session_shared_margin.py`. Expected
values are written by hand.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openalgo.backtest._live._margin_reject import (
    CLOSING_PURPOSES,
    HELD_EXIT_WORDS,
    HELD_FOR_READ,
    MARGIN_LOCK,
    MARGIN_PHRASES,
    SQUARE_OFF,
    MarginHolds,
    names_margin,
)
from openalgo.backtest._live._stops_cover import CLOSE_REJECTED, REFUSED_FOR_MARGIN
from openalgo.backtest._live._stops_deadline import Deadlines
from openalgo.backtest._live._stops_orders import Sent
from openalgo.backtest._live._stops_price import PriceRule
from openalgo.backtest._live._types import OrderIntent, Purpose

NS = 10**9
#: The rejection, a read begun before it, and one begun after it (UTC nanoseconds).
REJECTED_AT = 1_000 * NS
SQUARE_OFF_AT = 5_000 * NS


@pytest.mark.parametrize(
    ("reason", "margin"),
    [
        ("RMS: margin exceeds", True),
        ("RMS:Margin Exceeds", True),
        ("Insufficient funds. Required: 50000, Available: 100", True),
        ("insufficient margin for the order", True),
        ("trigger invalid", False),
        ("", False),
    ],
)
def test_the_phrase_table_only_words_the_alert(reason, margin):
    assert names_margin(reason) is margin


def test_each_phrase_says_where_it_comes_from_and_whether_it_was_checked():
    for phrase in MARGIN_PHRASES:
        assert phrase.text == phrase.text.casefold() and phrase.source
        assert isinstance(phrase.verified, bool)


def test_the_closing_orders_are_the_stop_and_every_exit():
    assert {Purpose.STOP, Purpose.EXIT, Purpose.RISK_EXIT, Purpose.SQUARE_OFF} == CLOSING_PURPOSES


class _Engine:
    def __init__(self, trader: _Trader) -> None:
        self._trader = trader

    def apply_margin_reject(self, order: int) -> tuple[int, int, int]:
        self._trader.calls.append(("margin_reject", order))
        return (1, 0, 3)

    def protect(self, instrument: int, product: str) -> str:
        self._trader.calls.append(("protect",))
        return "placed"

    def request_exit(self, instrument: int, product: str, reason: str, limit: float) -> str:
        self._trader.calls.append(("exit", reason, limit))
        return "sent"


class _Trader:
    def __init__(self, name: str) -> None:
        self.name, self.symbol, self.exchange = name, "SBIN", "NSE"
        self.finished = False
        self.square_off_ns = SQUARE_OFF_AT
        self.units = 50.0
        self.calls: list[tuple[Any, ...]] = []
        self.locks: list[str] = []
        self.engine = _Engine(self)

    def request_exit(self, reason: str) -> str:
        self.calls.append(("exit", reason))
        return "sent"

    def open_units(self) -> float:
        return self.units

    def lock(self, reason: str) -> bool:
        self.locks.append(reason)
        return True

    def order_name(self, kernel_order: int) -> str:
        return f"o{kernel_order}"


class _Session:
    """What `MarginHolds` reads of a session: its clock, events, alerts, groups, the
    reconciler's read and the stops supervisor's cover."""

    def __init__(self, traders: list[_Trader], shared: bool) -> None:
        self.traders = traders
        self.clock = REJECTED_AT
        self.events: list[tuple[str, dict]] = []
        self.alerts: list[str] = []
        self.reads: list[list[str]] = []
        shared_ = shared
        self.groups = type("G", (), {"shared": staticmethod(lambda trader: shared_)})()
        self.reconcile = type("R", (), {"read_before_exits": self.reads.append})()
        #: The price rule's LIMIT for an exit at the latest price, rupees.
        self.latest_limit = 495.0
        prices = SimpleNamespace(exit_limit=lambda trader: self.latest_limit)
        self.stops = SimpleNamespace(covered=lambda trader: 0, prices=prices)

    def now(self) -> int:
        return self.clock

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))

    def alert(self, kind: Any, text: str) -> None:
        self.alerts.append(text)


def _intent(purpose: Purpose) -> OrderIntent:
    return OrderIntent(
        intent_id="i1",
        strategy="B",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=50,
        price_type="SL-M",
        price=0.0,
        trigger_price=497.0,
        product="MIS",
        purpose=purpose,
        created_ns=0,
    )


def _held(shared: bool = True) -> tuple[MarginHolds, _Trader, _Session]:
    trader = _Trader("B")
    session = _Session([trader], shared)
    holds = MarginHolds(session)
    return holds, trader, session


def test_a_closing_order_of_a_group_of_one_takes_the_path_of_a_strategy_alone():
    holds, trader, _ = _held(shared=False)
    assert not holds.takes(trader, _intent(Purpose.STOP))


def test_an_entry_never_holds_a_position():
    holds, trader, _ = _held()
    assert not holds.takes(trader, _intent(Purpose.ENTRY))
    assert holds.takes(trader, _intent(Purpose.RISK_EXIT))


def test_a_rejection_locks_the_member_asks_a_read_at_once_and_alerts_once():
    holds, trader, session = _held()
    assert holds.reject(trader, 7, _intent(Purpose.STOP), "RMS: margin exceeds")
    assert holds.reject(trader, 8, _intent(Purpose.STOP), "RMS: margin exceeds")
    assert trader.calls == [("margin_reject", 7), ("margin_reject", 8)]
    assert trader.locks == [MARGIN_LOCK, MARGIN_LOCK]
    assert session.reads == [["B"], ["B"]]
    assert len(session.alerts) == 1 and "refused for margin" in session.alerts[0]
    assert holds.holds(trader) and holds.unprotected_reason(trader) == REFUSED_FOR_MARGIN


def test_a_rejection_in_other_words_is_worded_rejected():
    holds, trader, session = _held()
    holds.reject(trader, 7, _intent(Purpose.EXIT), "trigger invalid")
    assert "was rejected by the broker (trigger invalid)" in session.alerts[0]
    assert holds.unprotected_reason(trader) == CLOSE_REJECTED


def test_an_exit_and_a_protect_wait_and_are_made_once_after_an_agreeing_read():
    holds, trader, session = _held()
    holds.reject(trader, 7, _intent(Purpose.STOP), "RMS: margin exceeds")
    assert holds.request_exit(trader, "operator_stop") == HELD_FOR_READ
    assert holds.request_exit(trader, "kill_switch") == HELD_FOR_READ
    assert holds.protect(trader) == HELD_FOR_READ
    assert trader.calls == [("margin_reject", 7)]

    # A read begun before the rejection cannot let it go; one begun after it can.
    holds.agreed([trader], REJECTED_AT - NS)
    assert trader.calls == [("margin_reject", 7)]
    holds.agreed([trader], REJECTED_AT + NS)
    assert trader.calls == [("margin_reject", 7), ("exit", "operator_stop"), ("protect",)]
    holds.agreed([trader], REJECTED_AT + 2 * NS)
    assert len(trader.calls) == 3
    # From then on the session's own exits go at once.
    assert holds.request_exit(trader, "kill_switch") == "sent"


def test_the_square_off_of_a_held_position_is_asked_once_the_read_agreed():
    holds, trader, session = _held()
    holds.reject(trader, 7, _intent(Purpose.STOP), "RMS: margin exceeds")
    holds.agreed([trader], REJECTED_AT + NS)
    holds.duties(SQUARE_OFF_AT - 1)
    assert ("exit", SQUARE_OFF) not in trader.calls
    holds.duties(SQUARE_OFF_AT)
    holds.duties(SQUARE_OFF_AT + NS)
    assert trader.calls.count(("exit", SQUARE_OFF)) == 1


def test_a_hold_is_forgotten_once_the_position_is_flat():
    holds, trader, _ = _held()
    holds.reject(trader, 7, _intent(Purpose.STOP), "RMS: margin exceeds")
    trader.units = 0.0
    holds.duties(REJECTED_AT + NS)
    assert not holds.holds(trader) and holds.unprotected_reason(trader) is None


def test_a_held_limit_exit_is_priced_again_from_the_latest_price_when_a_read_agrees():
    holds, trader, session = _held()
    holds.reject(trader, 7, _intent(Purpose.STOP), "RMS: margin exceeds")
    # The price rule asked for its exit at 480.00; the price moved before a read agreed.
    assert holds.request_exit(trader, "stop_unfilled", 480.0) == HELD_FOR_READ
    holds.agreed([trader], REJECTED_AT + NS)
    assert trader.calls[-1] == ("exit", "stop_unfilled", session.latest_limit)


def _deadline_world(asked: str) -> tuple[Any, list[str]]:
    """A stops deadline over one leg whose stop was bound at 0 and never confirmed, whose
    exit the margin gate answers `asked`: the deadline and the alerts it sent."""
    trader = SimpleNamespace(name="B", symbol="SBIN", exchange="NSE")
    stop = SimpleNamespace(state=Sent.BOUND, bound_ns=0, kernel_order=2, order_id="S2")
    leg = SimpleNamespace(stop=2, orders={2: stop})
    stops = SimpleNamespace(leg=lambda t: leg, protection=SimpleNamespace(holds=lambda t: False))
    alerts: list[str] = []
    worker = SimpleNamespace(busy_since_ns=None, raw_pending=0)
    session = SimpleNamespace(
        traders=(trader,),
        parts=SimpleNamespace(worker=worker),
        inbox_waiting=0,
        margin=SimpleNamespace(request_exit=lambda t, reason: asked),
        event=lambda kind, **fields: None,
        alert=lambda kind, text: alerts.append(text),
    )
    return Deadlines(session, stops), alerts


def test_the_stops_deadline_says_a_held_positions_exit_waits_rather_than_that_it_went():
    deadlines, alerts = _deadline_world(HELD_FOR_READ)
    deadlines.due(10**12)
    [text] = alerts
    assert HELD_EXIT_WORDS in text and "is exited" not in text


def test_the_stops_deadline_says_the_position_is_exited_when_the_gate_lets_it_go():
    deadlines, alerts = _deadline_world("exit_requested")
    deadlines.due(10**12)
    [text] = alerts
    assert "the position is exited (the stop cancelled first)" in text


def test_the_price_rule_says_a_held_positions_exit_waits_rather_than_that_it_went():
    order = {"order_type": "SL", "side": "SELL", "trigger_rupees": 497.0}
    engine = SimpleNamespace(order=lambda k: order, live_state=lambda: {"desk": {"cancelling": []}})
    instruments = {0: SimpleNamespace(tick_size=0.05)}
    trader = SimpleNamespace(
        name="B",
        symbol="SBIN",
        exchange="NSE",
        finished=False,
        engine=engine,
        working=lambda: [2],
        open_units=lambda: 50.0,
        part=SimpleNamespace(setup=SimpleNamespace(instruments=instruments)),
    )
    alerts: list[str] = []
    session = SimpleNamespace(
        margin=SimpleNamespace(request_exit=lambda t, reason, limit: HELD_FOR_READ),
        event=lambda kind, **fields: None,
        alert=lambda kind, text: alerts.append(text),
    )
    rule = PriceRule(session)
    rule.observe(trader, 2, 496.0, 0)
    rule.refused(trader, 2)
    [text] = alerts
    assert HELD_EXIT_WORDS in text and "LIMIT at" not in text
