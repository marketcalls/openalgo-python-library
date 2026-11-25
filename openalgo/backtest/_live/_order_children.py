"""The child orders of fired GTTs on the order path (20-positional-live 3.2; the lead's
ruling N3b-2), split from `orders` so each module keeps its size: once the session binds a
GTT's child (`inbox`'s `BindGttChild`), the order path follows it as an order of its own
(`follow_child`), so its words from the stream and its polls come as its news and its
cancel can be sent, and a word of it that came before the binding, kept aside, is handed
over then: never dropped, never applied twice (the fill ledger takes cumulative words).

A child has no intent in the journal: its intent is made from the journal's GTT
(`child_intent`), a LIMIT at the GTT's child limit for its units, a leg's stop's child an
EXIT and a strategy GTT order's an ENTRY. Times are UTC nanoseconds. Used by the thread
that owns the order path.
"""

from __future__ import annotations

from ._journal_gtt_rows import CHILD_PRICE_TYPE, GttBinding
from ._order_follow import OrderFollower
from ._types import Now, OrderIntent, OrderUpdate, Purpose


def child_intent(gtt: GttBinding, order_id: str, created_ns: int) -> OrderIntent:
    """The intent the order path follows broker order `order_id`, the child of the
    journal's GTT `gtt`, by (module docstring)."""
    return OrderIntent(
        intent_id=f"gtt{gtt.seq}-child-{order_id}",
        strategy=gtt.strategy,
        symbol=gtt.symbol,
        exchange=gtt.exchange,
        action=gtt.terms.side,
        quantity=gtt.terms.quantity,
        price_type=CHILD_PRICE_TYPE,
        product=gtt.product,
        purpose=Purpose.EXIT if gtt.purpose is Purpose.STOP else Purpose.ENTRY,
        created_ns=created_ns,
        price=gtt.terms.limit_price,
    )


class ChildOrders:
    """The order path's part for GTT children (module docstring), mixed into
    `orders.OrderPath`, which holds the follower and the clock."""

    _follower: OrderFollower
    _now: Now

    def follow_child(self, order_id: str, intent: OrderIntent) -> OrderUpdate | None:
        """Follow broker order `order_id`, a fired GTT's child, by `intent`
        (`child_intent`); the news of its latest word that came before, or None."""
        self._follower.follow(order_id, intent, sent_ns=self._now())
        return self._follower.claim(order_id)


__all__ = ["ChildOrders", "child_intent"]
