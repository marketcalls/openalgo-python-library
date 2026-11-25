"""The child orders of fired GTTs in a strategy's venue map (20-positional-live 3.2 and 3.3;
the lead's ruling N3b-2), split from `venue_map` so each module keeps one job.

When a GTT fires the kernel follows its child as an order of its own (the binding's
`apply_gtt_triggered`); the session then binds that kernel order here to the broker's
order ID of the child the GTT follower found (`bind_child`), so the child's fills, its
cancel (the kernel's when it stays unfilled through a base bar) and its end go through
the venue map and the order's fill ledger as any order's do. A child the follower could
not find, whose fill the position's move stands for, is bound to an ID of the session's
own (`missed_child_id`), never sent to the broker: its one fill is given by the session.

The child's intent is the venue map's own record of it: a LIMIT at the GTT's child limit
for its units, a leg's stop's child an EXIT and a strategy GTT order's an ENTRY.
Quantities are whole units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..errors import LiveError
from ._fill_ledger import FillLedger
from ._journal_gtt_rows import CHILD_PRICE_TYPE, GttTerms
from ._journal_reads import JournalSession
from ._plan_catch_up import INSTRUMENT
from ._types import AlertsLike, Now, OrderIntent, Purpose
from ._venue_requests import OrderTerms, intent_id
from ._venue_saved import VenueOrder


def missed_child_id(trigger_id: str) -> str:
    """The order ID the session gives a fired GTT's child it could not find (module
    docstring): never a broker's, so no word of the broker's is ever taken for it."""
    return f"gtt-{trigger_id}-missed"


class GttChildren:
    """The child orders of `venue_map.VenueMap` (module docstring), mixed into it, which
    holds the kernel session, the strategy, its product, its instruments and ticks, the
    journal session, the alerts, the clock, and its orders by kernel order, broker order
    ID and intent."""

    _engine: Any
    _strategy: str
    _version_id: str
    _product: str
    _markets: Sequence[tuple[str, str]]
    _ticks: Sequence[float]
    _session: JournalSession
    _alerts: AlertsLike
    _now: Now
    _orders: dict[int, VenueOrder]
    _by_order_id: dict[str, int]
    _by_intent: dict[str, int]

    def bind_child(
        self, kernel_order: int, order_id: str, terms: GttTerms, purpose: Purpose
    ) -> None:
        """Bind kernel order `kernel_order`, the child of a fired GTT of `terms` held for
        `purpose` (a leg's STOP, or a strategy's ENTRY), to broker order `order_id`.
        Raises LiveError for a kernel order or a broker order bound already."""
        if kernel_order in self._orders or order_id in self._by_order_id:
            raise LiveError(
                f"the child order {order_id} of {self._strategy} (kernel o{kernel_order}) "
                "is bound already"
            )
        symbol, exchange = self._markets[INSTRUMENT]
        intent = OrderIntent(
            intent_id=intent_id(self._session, self._strategy, kernel_order),
            strategy=self._strategy,
            symbol=symbol,
            exchange=exchange,
            action=terms.side,
            quantity=terms.quantity,
            price_type=CHILD_PRICE_TYPE,
            product=self._product,
            purpose=Purpose.EXIT if purpose is Purpose.STOP else Purpose.ENTRY,
            created_ns=self._now(),
            price=terms.limit_price,
            version_id=self._version_id,
        )
        ledger = FillLedger(
            self._engine, kernel_order, tick=self._ticks[INSTRUMENT], alerts=self._alerts
        )
        order_terms = OrderTerms(terms.quantity, CHILD_PRICE_TYPE, terms.limit_price, 0.0)
        self._orders[kernel_order] = VenueOrder(intent, order_terms, ledger, order_id=order_id)
        self._by_order_id[order_id] = kernel_order
        self._by_intent[intent.intent_id] = kernel_order


__all__ = ["GttChildren", "missed_child_id"]
