"""Following the session's orders once they are bound, for the order path (`orders`).

The engine is handed each order's news once (`_order_news`), every fill with its price,
and nothing of orders the session did not send. The order-updates stream is the source;
`orderstatus` is polled only as a fallback: every second while an order has no word at
all, and every 30 seconds as a safety net while it rests. Polling runs in rounds of at
most one a second, each polling a few orders and reading the tradebook at most once.

Times are UTC nanoseconds, quantities units, prices rupees.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._order_answers import server_answer
from ._order_news import is_newer, is_news, is_priceless
from ._order_updates import fills_given, normalize_update, number, units
from ._types import LOGGER_NAME, GuardLike, JournalLike, Now, OrderUpdate

logger = logging.getLogger(LOGGER_NAME)

#: An order with no word 2 seconds after it was sent is polled (08-live-trading 6.1) ...
POLL_AFTER_NS = 2 * NANOS_PER_SECOND
#: ... then once a second until a word of it arrives; polling rounds are this far apart.
POLL_EVERY_NS = 1 * NANOS_PER_SECOND
#: A resting order (open, or a stop waiting for its trigger) is polled this often as a
#: safety net for a stream that falls silent; the stream is its source.
SAFETY_POLL_EVERY_NS = 30 * NANOS_PER_SECOND
#: OpenAlgo answers `orderstatus` 10 times a second per client IP (`API_RATE_LIMIT`,
#: restx_api/orderstatus.py on the server); a round polls at most 3, well under it.
MAX_POLLS_PER_ROUND = 3
#: The stream carries every update of the account (manual orders, the server's own
#: strategy runs); the latest of this many orders not yet bound are kept, in case an
#: unbound intent turns out to be one of them.
MAX_UNCLAIMED_UPDATES = 256

#: A polled order's word: its update, or the `orderstatus` answer whose fills are to be
#: read from the tradebook.
_Polled = OrderUpdate | Mapping[str, Any]


@dataclass
class _Followed:
    """An order of the session: its strategy, when it was sent, the last update handed to
    the engine, a fill waiting for its price, and when it was last polled."""

    strategy: str
    sent_ns: int
    seen: OrderUpdate | None = None
    held: OrderUpdate | None = None
    polled_ns: int | None = None

    def poll_due_ns(self) -> int | None:
        """When the order is next polled; None once it is terminal."""
        if self.seen is not None and self.seen.status.terminal:
            return None
        last = self.seen or self.held
        if last is None:
            if self.polled_ns is None:
                return self.sent_ns + POLL_AFTER_NS
            return self.polled_ns + POLL_EVERY_NS
        return max(self.polled_ns or 0, last.ts_recv_ns) + SAFETY_POLL_EVERY_NS


class OrderFollower:
    """The session's orders once bound, their updates and the polling fallback.

    `client` is an `openalgo.api`; `journal` and `guard` the session's; `now` gives UTC
    nanoseconds. Every `orderstatus` and `tradebook` answer goes to the guard as the
    server wrote it, and no poll is made while the guard is locked: the analyzer switch
    routes the reads too, so they would answer from the other book.
    """

    def __init__(self, client: Any, *, journal: JournalLike, guard: GuardLike, now: Now) -> None:
        self._client = client
        self._journal = journal
        self._guard = guard
        self._now = now
        self._orders: dict[str, _Followed] = {}
        self._unclaimed: collections.OrderedDict[str, OrderUpdate] = collections.OrderedDict()
        self._round_ns: int | None = None

    def order_ids(self) -> set[str]:
        """The IDs of the orders followed, terminal ones included."""
        return set(self._orders)

    def strategy_of(self, order_id: str) -> str | None:
        """The strategy of a followed order, or None for an order not followed."""
        order = self._orders.get(order_id)
        return None if order is None else order.strategy

    def follow(self, order_id: str, strategy: str, *, sent_ns: int) -> None:
        """Follow an order just bound in the journal, and note it with the guard."""
        self._guard.note_order_sent(order_id)
        self._orders[order_id] = _Followed(strategy, sent_ns)

    def claim(self, order_id: str) -> OrderUpdate | None:
        """Hand over the latest update of a newly followed order that arrived before it
        was bound, or None (none arrived, or its fill has no price yet)."""
        update = self._unclaimed.pop(order_id, None)
        return None if update is None else self._take(self._orders[order_id], update)

    def restore(self) -> None:
        """Follow the journal's bound orders again after a restart, from their last
        updates: they can be modified, cancelled and polled."""
        for bound in self._journal.bound_orders():
            self.follow(bound.order_id, bound.intent.strategy, sent_ns=bound.intent.created_ns)
            self._orders[bound.order_id].seen = bound.last_update

    def on_update(self, update: OrderUpdate) -> OrderUpdate | None:
        """The update if it is news of a followed order with its price, journaled and
        checked by the guard; None otherwise (kept aside for an order not bound)."""
        order = self._orders.get(update.order_id)
        if order is None:
            self._keep_unclaimed(update)
            return None
        return self._take(order, update)

    def due_polls(self) -> list[OrderUpdate]:
        """Run a polling round when one is due and return its news (`source` "poll").

        A round polls the most overdue orders (at most `MAX_POLLS_PER_ROUND`) and reads
        the tradebook once when a fill lacks its quantity or price. A failed or
        unreadable answer is logged and the order is polled again later.
        """
        now = self._now()
        if self._guard.locked or (
            self._round_ns is not None and now - self._round_ns < POLL_EVERY_NS
        ):
            return []
        due = sorted(
            (due_ns, order_id)
            for order_id, order in self._orders.items()
            if (due_ns := order.poll_due_ns()) is not None and due_ns <= now
        )[:MAX_POLLS_PER_ROUND]
        held = [order_id for order_id, order in self._orders.items() if order.held is not None]
        if not due and not held:
            return []
        self._round_ns = now
        polled = self._poll([order_id for _, order_id in due], now)
        if polled is None:
            return []
        wanted = {order_id for order_id, update in polled.items() if _lacks_fills(update)}
        trades = self._trades(wanted | set(held)) if wanted or held else {}
        if trades is None:
            return []
        new = []
        for order_id in dict.fromkeys([*polled, *held]):
            order = self._orders[order_id]
            word = polled.get(order_id) or order.held
            update = None if word is None else _with_trades(word, trades.get(order_id), now)
            taken = None if update is None else self._take(order, update)
            if taken is not None:
                new.append(taken)
        return new

    def _take(self, order: _Followed, update: OrderUpdate) -> OrderUpdate | None:
        if not is_news(order.seen, update):
            logger.debug("dropped an update that is not news: %s", update)
            return None
        if is_priceless(update):
            seen = order.seen
            if seen is None or seen.filled_quantity != update.filled_quantity:
                if order.held is None or is_newer(order.held, update):
                    order.held = update
                logger.info("order %s has a fill with no price yet: it is held", update.order_id)
                return None
            update = dataclasses.replace(update, average_price=seen.average_price)
        self._journal.record_update(update)
        order.seen = update
        if order.held is not None and (
            update.status.terminal or update.filled_quantity >= order.held.filled_quantity
        ):
            order.held = None
        self._guard.check_update(update)
        return update

    def _keep_unclaimed(self, update: OrderUpdate) -> None:
        kept = self._unclaimed.get(update.order_id)
        if kept is not None and not is_newer(kept, update):
            return
        self._unclaimed.pop(update.order_id, None)
        self._unclaimed[update.order_id] = update
        while len(self._unclaimed) > MAX_UNCLAIMED_UPDATES:
            self._unclaimed.popitem(last=False)

    def _poll(self, order_ids: list[str], now: int) -> dict[str, _Polled] | None:
        """The `orderstatus` answers of a round, keyed by order: an `OrderUpdate`, or the
        answer itself when its fills are to be read from the tradebook. None when an
        answer locked the guard: the round is then dropped."""
        polled: dict[str, _Polled] = {}
        for order_id in order_ids:
            order = self._orders[order_id]
            order.polled_ns = now
            answer = server_answer(
                self._client.orderstatus(order_id=order_id, strategy=order.strategy)
            )
            self._guard.check_response(answer)
            if self._guard.locked:
                return None
            data = answer.get("data")
            if answer.get("status") != "success" or not isinstance(data, Mapping):
                logger.warning("orderstatus of %s failed: %s", order_id, answer.get("message"))
                continue
            try:
                if fills_given(data, order_id):
                    polled[order_id] = normalize_update(answer, ts_recv_ns=now, source="poll")
                else:
                    polled[order_id] = answer
            except LiveError as error:
                logger.error("orderstatus of %s cannot be read: %s", order_id, error)
        return polled

    def _trades(self, order_ids: set[str]) -> dict[str, tuple[int, float]] | None:
        """Each order's filled quantity and average price in the tradebook. None when it
        cannot be read, or when its answer locked the guard."""
        answer = server_answer(self._client.tradebook())
        self._guard.check_response(answer)
        trades = answer.get("data")
        if self._guard.locked:
            return None
        if answer.get("status") != "success" or not isinstance(trades, list):
            logger.warning("the tradebook could not be read: %s", answer.get("message"))
            return {}
        fills = {}
        for order_id in order_ids:
            own = [trade for trade in trades if str(trade.get("orderid")) == order_id]
            try:
                filled = sum(units(trade, "quantity", order_id) for trade in own)
                value = sum(
                    units(trade, "quantity", order_id) * number(trade, "average_price", order_id)
                    for trade in own
                )
            except LiveError as error:
                logger.error("the trades of order %s cannot be read: %s", order_id, error)
                continue
            fills[order_id] = (filled, value / filled if filled else 0.0)
        return fills


def _lacks_fills(update: _Polled) -> bool:
    """Whether a polled answer needs the tradebook: no fills given, or no price."""
    return not isinstance(update, OrderUpdate) or is_priceless(update)


def _with_trades(update: _Polled, fills: tuple[int, float] | None, now: int) -> OrderUpdate | None:
    """An update completed from the order's trades: a polled answer without its fills
    takes theirs, and a fill without its price takes theirs when they add up to it."""
    if not isinstance(update, OrderUpdate):
        if fills is None:
            return None
        data = {**update["data"], "filled_quantity": fills[0], "average_price": fills[1]}
        try:
            return normalize_update({**update, "data": data}, ts_recv_ns=now, source="poll")
        except LiveError as error:
            logger.error("an orderstatus answer cannot be read: %s", error)
            return None
    if is_priceless(update) and fills is not None and fills[0] == update.filled_quantity:
        return dataclasses.replace(update, average_price=fills[1])
    return update


__all__ = [
    "MAX_POLLS_PER_ROUND",
    "MAX_UNCLAIMED_UPDATES",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "OrderFollower",
]
