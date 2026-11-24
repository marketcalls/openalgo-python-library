"""A leg whose GTT fired or is gone is in flight for the reconciler until the GTT follower
resolves it (`_live/_gtt_flight.py`, `_reconcile_steps.in_flight`, `_session_news`;
20-positional-live 3.2 last paragraph, review H3; the lead's ruling N3a-2): an order of the
child's shape is neither another's order naming the strategy (16-session-design 5.2 step 2)
nor an external trade (5.3 rows 1 and 2), and the position's fall is not a mismatch; once
the follower resolves it, the comparisons apply those rules again.

`SbinHold` holds 10 SBIN CNC bought today (the CNC line of the position book; no holdings).
Its stop GTT fired: the broker placed the child, order "C1", selling the 10. The session's
parts are the in-process doubles of `test_broker_units_callers`. Nothing contacts a server.
"""

from __future__ import annotations

from typing import Any

from test_broker_units_callers import NO_FUNDS, NO_ORDERS, Leg, Loop

from openalgo.backtest._live._gtt_flight import GttFlight
from openalgo.backtest._live._gtt_types import (
    Direction,
    GttAction,
    GttAnswer,
    GttAnswerKind,
    GttEvent,
    GttOutcome,
    GttRequest,
)
from openalgo.backtest._live._session_news import News
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._venue_requests import Priority
from openalgo.backtest._live._worker_queue import ReadBooks
from openalgo.backtest._live.inbox import Books, GttNews
from openalgo.backtest._live.reconcile import Reconciler

NS = 10**9
UNITS = 10
HOLDING_NONE = {"status": "success", "data": {"holdings": []}}
BOUGHT = {
    "status": "success",
    "data": [{"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "quantity": UNITS}],
}
#: The position book once the child sold the 10: the CNC line is flat.
SOLD = {
    "status": "success",
    "data": [{"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "quantity": 0}],
}
#: The child's trade: 10 SBIN CNC sold at 789.50 by order C1, which the session did not send.
CHILD_TRADE = {
    "status": "success",
    "data": [
        {
            "orderid": "C1",
            "symbol": "SBIN",
            "exchange": "NSE",
            "product": "CNC",
            "action": "SELL",
            "quantity": UNITS,
            "average_price": 789.5,
        }
    ],
}
NO_TRADES = {"status": "success", "data": []}
#: The order book with the child resting unfilled, naming the strategy (the sandbox's book
#: names a GTT child's strategy).
CHILD_RESTING = {
    "status": "success",
    "data": {
        "orders": [
            {
                "orderid": "C1",
                "symbol": "SBIN",
                "exchange": "NSE",
                "product": "CNC",
                "action": "SELL",
                "quantity": UNITS,
                "pricetype": "LIMIT",
                "order_status": "open",
                "strategy": "SbinHold",
            }
        ]
    },
}


def found(event: GttEvent, seq: int = 1) -> GttOutcome:
    return GttOutcome(seq, "SbinHold", 1, "T1", "SBIN", "NSE", "CNC", event)


def reconciling() -> tuple[Loop, Reconciler, Leg]:
    leg = Leg("SbinHold", "CNC", UNITS)
    loop = Loop(leg)
    loop.reconcile = Reconciler(loop)
    return loop, loop.reconcile, leg


def read(loop: Loop, reconciler: Reconciler, positions: Any, trades: Any, orders: Any) -> None:
    """Ask for a read of every leg now and answer it with these books."""
    reconciler.ask_of(None)
    request = [r for r in loop.handed if isinstance(r, ReadBooks)][-1]
    loop.clock += NS
    books = Books(
        request.generation,
        positions,
        orders,
        trades,
        NO_FUNDS,
        loop.clock,
        loop.clock,
        holdings=HOLDING_NONE,
    )
    reconciler.books(books)


def compared(loop: Loop) -> list[str]:
    return [fields["found"] for kind, fields in loop.events if kind == "reconcile"]


def acted(reconciler: Reconciler) -> list[tuple[str, str]]:
    return [(row.kind, row.action) for row in reconciler.rows]


# ----- the record ------------------------------------------------------------------------------


def test_a_gtt_fired_or_gone_holds_its_strategys_leg_until_an_outcome_resolves_it():
    flight = GttFlight()

    flight.take(found(GttEvent.IN_FLIGHT))
    held = flight.holds("SbinHold")
    flight.take(found(GttEvent.TRIGGERED))

    assert (held, flight.holds("SbinHold")) == (True, False)


def test_a_leg_with_two_gtts_in_flight_stays_held_until_both_are_resolved():
    flight = GttFlight()
    flight.take(found(GttEvent.IN_FLIGHT, seq=1))
    flight.take(found(GttEvent.IN_FLIGHT, seq=2))

    flight.take(found(GttEvent.CHILD_MISSED, seq=1))

    assert flight.holds("SbinHold")


def test_a_gtt_calls_answer_holds_nothing():
    flight = GttFlight()
    request = GttRequest(
        1,
        GttAction.PLACE,
        Purpose.STOP,
        "SbinHold",
        1,
        "SBIN",
        "NSE",
        "CNC",
        "SELL",
        790.0,
        780.0,
        UNITS,
        Direction.BELOW,
        Priority.PROTECT,
    )

    flight.take(GttAnswer(request, GttAnswerKind.PLACED, 1, "T1"))

    assert not flight.holds("SbinHold")


def test_the_sessions_gtt_news_reaches_the_reconcilers_record():
    loop, reconciler, _ = reconciling()

    News(loop).take(GttNews(found(GttEvent.IN_FLIGHT)))

    assert reconciler.gtts.holds("SbinHold")
    assert ("gtt", {"strategy": "SbinHold", "seq": 1, "found": "in_flight"}) in loop.events


# ----- the reconciler's rules held, then applied -----------------------------------------------


def test_a_leg_in_flight_is_not_compared_so_the_childs_sale_is_no_external_trade():
    loop, reconciler, leg = reconciling()
    reconciler.gtts.take(found(GttEvent.IN_FLIGHT))

    for _ in range(2):
        read(loop, reconciler, SOLD, CHILD_TRADE, NO_ORDERS)

    assert (compared(loop), acted(reconciler), leg.engine.units) == ([], [], UNITS)


def test_once_resolved_the_leg_is_compared_and_the_external_rule_applies_again():
    loop, reconciler, leg = reconciling()
    reconciler.gtts.take(found(GttEvent.IN_FLIGHT))
    read(loop, reconciler, SOLD, CHILD_TRADE, NO_ORDERS)

    reconciler.gtts.take(found(GttEvent.TRIGGERED))
    for _ in range(2):
        read(loop, reconciler, SOLD, CHILD_TRADE, NO_ORDERS)

    # Unit N3b binds the child as the kernel's own exit; until then its trade is another
    # order's, explaining the flat book: noted, then adopted at its price.
    assert acted(reconciler) == [("explained_flat", "noted"), ("explained_flat", "adopted")]
    assert leg.engine.units == 0


def test_the_child_naming_the_strategy_is_no_foreign_order_while_the_leg_is_in_flight():
    loop, reconciler, _ = reconciling()
    reconciler.gtts.take(found(GttEvent.IN_FLIGHT))

    read(loop, reconciler, BOUGHT, NO_TRADES, CHILD_RESTING)

    assert acted(reconciler) == []


def test_once_resolved_an_order_naming_the_strategy_it_did_not_send_is_foreign_again():
    loop, reconciler, _ = reconciling()
    reconciler.gtts.take(found(GttEvent.IN_FLIGHT))
    reconciler.gtts.take(found(GttEvent.MISMATCH))

    read(loop, reconciler, BOUGHT, NO_TRADES, CHILD_RESTING)

    assert acted(reconciler) == [("foreign", "noted")]
