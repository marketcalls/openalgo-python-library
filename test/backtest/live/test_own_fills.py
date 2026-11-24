"""The session's own fills on a leg as the GTT parts read them (`_live/_own_fills.py`;
20-positional-live 3.3, 16-session-design 5.3 and F78): the units its orders filled, and
whether the broker's books can be taken to show them yet.

The leg is SBIN NSE CNC; the clock stands at 1 000 s. An order's word is built as the
order path keeps it; expected values are worked by hand from each row.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._own_fills import POSITION_BOOK_LAG_S, OwnFills
from openalgo.backtest._live._types import Mode, OrderIntent, OrderStatus, OrderUpdate, Purpose

NS = 10**9
NOW_NS = 1_000 * NS
LEG = ("SBIN", "NSE", "CNC")


def intent(action="SELL", *, symbol="SBIN", product="CNC", number=1):
    return OrderIntent(
        intent_id=f"i{number}",
        strategy="Swing",
        symbol=symbol,
        exchange="NSE",
        action=action,
        quantity=5,
        price_type="MARKET",
        product=product,
        purpose=Purpose.EXIT,
        created_ns=0,
    )


def word(status=OrderStatus.COMPLETE, filled=5, heard_s_ago=60.0):
    """An order's latest word, heard `heard_s_ago` seconds before the clock's now."""
    return OrderUpdate(
        order_id="o1",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=5,
        filled_quantity=filled,
        pending_quantity=5 - filled,
        average_price=800.0 if filled else 0.0,
        status=status,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=NOW_NS - round(heard_s_ago * NS),
        source="stream",
    )


class Unbound:
    """A journal whose unbound intents are `intents`."""

    def __init__(self, *intents):
        self.intents = list(intents)

    def unbound(self):
        return self.intents


def own_fills(orders, *unbound):
    return OwnFills(lambda: orders, journal=Unbound(*unbound), now=lambda: NOW_NS)


def test_the_units_are_each_order_on_the_leg_by_its_latest_word_signed():
    orders = [
        (intent("BUY", number=1), word(filled=10)),
        (intent("SELL", number=2), word(filled=4)),
        (intent("SELL", number=3), None),
        (intent("SELL", number=4, product="NRML"), word(filled=7)),
        (intent("SELL", number=5, symbol="TCS"), word(filled=2)),
    ]

    # 10 bought less 4 sold; the order with no word, the NRML leg and TCS count nothing.
    assert own_fills(orders).units(LEG) == 6


@pytest.mark.parametrize(
    ("orders", "unbound", "settled"),
    [
        ([], [], True),
        ([(intent(), word(heard_s_ago=POSITION_BOOK_LAG_S))], [], True),
        ([(intent(), word(heard_s_ago=POSITION_BOOK_LAG_S - 1.0))], [], False),
        ([(intent(), word(OrderStatus.REJECTED, filled=0, heard_s_ago=0.0))], [], True),
        ([(intent(), word(OrderStatus.CANCELLED, filled=2, heard_s_ago=1.0))], [], False),
        ([(intent(), None)], [], False),
        ([(intent(), word(OrderStatus.OPEN, filled=0))], [], False),
        ([(intent(), word(OrderStatus.TRIGGER_PENDING, filled=0))], [], False),
        ([], [intent()], False),
        ([(intent(symbol="TCS"), None)], [intent(product="NRML")], True),
    ],
    ids=[
        "no own order",
        "a fill heard the lag ago",
        "a fill heard within the lag",
        "a rejection with no fill, just heard",
        "a cancel with fills, just heard",
        "an order sent with no word yet",
        "an order working",
        "a stop waiting for its trigger",
        "an intent unbound",
        "orders of other legs only",
    ],
)
def test_the_own_fills_are_settled_only_once_the_books_can_show_them(orders, unbound, settled):
    assert own_fills(orders, *unbound).settled(LEG) is settled
