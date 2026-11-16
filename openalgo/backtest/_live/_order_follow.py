"""Following the session's orders once they are bound, for the order path (`orders`).

The engine is handed each order's news once (`_order_news`), with its status as the
broker reported it, every fill with its price, and nothing of orders the session did not
send. A fill found by polling, a fill with no price and a cancel that reports no fill
wait for the tradebook (`_order_trades`), whose reads back off as the polls do; a wait
longer than 10 seconds is alerted, and the session may release the fill at a price.

The order-updates stream is the source; `orderstatus` is polled as a fallback
(`_order_state`): once a second while an order has no word yet (from 2 s after the send)
or is a market order still working; for a resting order (any other order open, or one
waiting for a trigger), every 30 seconds as a safety net once the stream has spoken for
it, else backing off from 1 s to 2 s to 5 s (2 s for an entry or an add). Polling runs
in rounds of at most one a second, each polling at most 3 orders and reading the
tradebook at most once. A gap in the stream, and a targeted read whose row gives no
fills, ask for polls out of turn, which the next rounds make.

Times are UTC nanoseconds, quantities units, prices rupees.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Mapping
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError, ModeChangedError
from ._order_answers import ServerWatch
from ._order_check import OrderCheck, check_of, polled_word, read_status
from ._order_news import is_newer, is_news, is_priceless, repeats, with_known_price
from ._order_state import (
    POLL_EVERY_NS,
    Followed,
    HeldFill,
    released_cancel,
    released_fill,
    with_floor,
)
from ._order_trades import Fills, Polled, needs_trades, read_tradebook, wants_trades, with_fills
from ._types import (
    LOGGER_NAME,
    AlertKind,
    AlertsLike,
    BoundOrder,
    GuardLike,
    JournalLike,
    Now,
    OrderIntent,
    OrderUpdate,
)

logger = logging.getLogger(LOGGER_NAME)

#: OpenAlgo answers `orderstatus` 10 times a second per client IP (`API_RATE_LIMIT`,
#: restx_api/orderstatus.py on the server); a round polls at most 3, well under it.
MAX_POLLS_PER_ROUND = 3
#: The stream carries every update of the account (manual orders, the server's own
#: strategy runs, the session's other instruments), and only an unbound intent's order
#: is ever claimed from them, within seconds of its send: the latest words of the 256
#: orders heard of most recently cover that wait on a busy account, and bound the memory.
MAX_UNCLAIMED_UPDATES = 256
#: A fill held this long for its price (or a cancel for the tradebook's word) is alerted.
HELD_ALERT_AFTER_NS = 10 * NANOS_PER_SECOND


class OrderFollower:
    """The session's orders once bound, their updates and the polling fallback.

    `client` is an `openalgo.api`; `journal`, `guard` and `alerts` the session's; `now`
    gives UTC nanoseconds. Every `orderstatus` and `tradebook` answer goes to the guard
    as the server wrote it, and no poll is made while the guard is locked: the analyzer
    switch routes the reads too, so they would answer from the other book.
    """

    def __init__(
        self,
        client: Any,
        *,
        journal: JournalLike,
        guard: GuardLike,
        alerts: AlertsLike,
        watch: ServerWatch,
        now: Now,
    ) -> None:
        self._client = client
        self._journal = journal
        self._guard = guard
        self._alerts = alerts
        self._watch = watch
        self._now = now
        self._orders: dict[str, Followed] = {}
        self._unclaimed: collections.OrderedDict[str, OrderUpdate] = collections.OrderedDict()
        self._round_ns: int | None = None

    def order_ids(self) -> set[str]:
        """The IDs of the orders followed, terminal ones included."""
        return set(self._orders)

    def intent_of(self, order_id: str) -> OrderIntent | None:
        """The intent of a followed order, or None for an order not followed."""
        order = self._orders.get(order_id)
        return None if order is None else order.intent

    def follow(self, order_id: str, intent: OrderIntent, *, sent_ns: int) -> None:
        """Follow an order just bound in the journal, and note it with the guard."""
        self._guard.note_order_sent(order_id)
        self._orders[order_id] = Followed.of(intent, sent_ns=sent_ns)

    def changed_price_type(self, order_id: str, price_type: str) -> None:
        """Keep the price type of a followed order that a modify changed."""
        self._orders[order_id].price_type = price_type

    def claim(self, order_id: str) -> OrderUpdate | None:
        """Hand over the latest update of a newly followed order that arrived before it
        was bound, or None (none arrived, or it waits for the tradebook)."""
        update = self._unclaimed.pop(order_id, None)
        return None if update is None else self._take(self._orders[order_id], update)

    def restore(self, bound_orders: list[BoundOrder]) -> None:
        """Follow the journal's bound orders again after a restart, from their last
        updates. Raises LiveError (and follows none) before the guard has started."""
        for bound in bound_orders:
            self._guard.note_order_sent(bound.order_id)
        for bound in bound_orders:
            last = bound.last_update
            order = Followed.of(bound.intent, sent_ns=bound.intent.created_ns)
            order.seen = last
            if last is not None and last.source == "stream":
                order.heard_from_stream(last.ts_recv_ns)
            self._orders[bound.order_id] = order

    def on_update(self, update: OrderUpdate) -> OrderUpdate | None:
        """The update if it is news of a followed order and needs no tradebook word,
        journaled and checked by the guard; None otherwise (kept aside for an order not
        bound, held for one that waits for the tradebook)."""
        order = self._orders.get(update.order_id)
        if order is None:
            self._keep_unclaimed(update)
            return None
        return self._take(order, update)

    def last_known(self, order_id: str) -> OrderUpdate | None:
        """The latest word of a followed order (`Followed.last_word`): held for the
        tradebook, or handed over. None before any word, or for an order not followed."""
        order = self._orders.get(order_id)
        return None if order is None else order.last_word

    def stream_gap(self, since_ns: int) -> None:
        """Ask for one poll of every followed order still working that the stream has not
        spoken for since `since_ns` (UTC nanoseconds): the next polling rounds make them."""
        now = self._now()
        missed = [order for order in self._orders.values() if order.missed_since(since_ns)]
        for order in missed:
            order.ask_poll(now)
        logger.info("the order stream may have missed words: %d orders to poll", len(missed))

    def check(self, order_id: str) -> OrderCheck:
        """One `orderstatus` read of a followed order, its news taken as a poll's; see
        `OrderPath.check_order`."""
        order = self._orders.get(order_id)
        if order is None:
            raise LiveError(f"order {order_id} was not sent by this session: it reads its own")
        if self._guard.locked:
            raise ModeChangedError(
                f"order {order_id} is not read: the mode guard is locked, so the session "
                "reads nothing more (the analyzer switch routes the reads to the other book)"
            )
        now = self._now()
        answer = self._read_status(order_id, now)
        if self._guard.locked:
            raise ModeChangedError(
                f"the orderstatus of order {order_id} was answered from the other mode's "
                "book, which says nothing of this session's order: the guard is locked"
            )
        word = polled_word(answer, order_id, now)
        news = None
        if isinstance(word, OrderUpdate):
            news = self._take(order, word)
        elif word is not None:
            # Its fills are in the tradebook, which the next round reads with a poll.
            order.ask_poll(now)
        return check_of(answer, order_id, news)

    def held_fills(self) -> list[HeldFill]:
        """The updates waiting for their price or the tradebook's confirmation."""
        return [
            HeldFill(
                order_id, order.held.filled_quantity, order.held_since_ns, order.held.average_price
            )
            for order_id, order in self._orders.items()
            if order.held is not None
        ]

    def release_fill(self, order_id: str, average_price: float) -> OrderUpdate | None:
        """Hand over a held update at `average_price` (rupees; the feed's last traded
        price, say), or as reported when it has no fill; None when it is no longer news.
        Raises LiveError when the order holds nothing, or the price is not positive."""
        order = self._orders.get(order_id)
        held = None if order is None else order.held
        if order is None or held is None:
            raise LiveError(f"order {order_id} holds no fill to release")
        return self._release(order, released_fill(held, average_price))

    def release_cancel(
        self, order_id: str, filled_quantity: int, average_price: float
    ) -> OrderUpdate | None:
        """Hand over a held cancel with `filled_quantity` units filled at `average_price`
        (rupees), or None when it is no longer news; see `OrderPath.release_cancel`."""
        order = self._orders.get(order_id)
        held = None if order is None else order.held
        if order is None or held is None:
            raise LiveError(f"order {order_id} holds no cancel to release")
        released = released_cancel(held, filled_quantity, average_price)
        return self._release(order, with_known_price(order.seen, released))

    def _release(self, order: Followed, released: OrderUpdate) -> OrderUpdate | None:
        if not is_news(order.seen, released):
            order.release()
            return None
        return self._deliver(order, released)

    def due_polls(self) -> list[OrderUpdate]:
        """Run a polling round when one is due and return its news.

        A round polls the most overdue orders (at most `MAX_POLLS_PER_ROUND`), reads the
        tradebook once for the words that wait for it, and alerts the holds past their
        deadline. A failed or unreadable answer is logged and asked again later.
        """
        now = self._now()
        self._alert_overdue(now)
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
        held_due = [order_id for order_id in held if self._orders[order_id].trades_due_ns() <= now]
        if not due and not held_due:
            return []
        self._round_ns = now
        polled = self._poll([order_id for _, order_id in due], now)
        if polled is None:
            return []
        wanted = {order_id for order_id, word in polled.items() if wants_trades(word)}
        if not wanted and not held_due:
            return self._complete_all(polled, [], {}, now)
        fills = read_tradebook(
            self._client,
            guard=self._guard,
            watch=self._watch,
            order_ids=wanted | set(held),
            now=self._now(),
        )
        if fills is None:
            return []
        for order_id in held:
            order = self._orders[order_id]
            order.trades_read_ns, order.trade_reads = now, order.trade_reads + 1
        return self._complete_all(polled, held, fills, now)

    def _complete_all(
        self, polled: dict[str, Polled], held: list[str], fills: dict[str, Fills], now: int
    ) -> list[OrderUpdate]:
        """The news of a round: each polled word, then each held one, with their fills."""
        new = [self._complete(order_id, word, fills, now) for order_id, word in polled.items()]
        for order_id in held:
            word = self._orders[order_id].held
            if word is not None:
                new.append(self._complete(order_id, word, fills, now))
        return [update for update in new if update is not None]

    def _complete(
        self, order_id: str, word: Polled, fills: dict[str, Fills], now: int
    ) -> OrderUpdate | None:
        update, checked = with_fills(word, fills.get(order_id), now)
        return (
            None if update is None else self._take(self._orders[order_id], update, checked=checked)
        )

    def _take(
        self, order: Followed, update: OrderUpdate, *, checked: bool = False
    ) -> OrderUpdate | None:
        if update.source == "stream":
            order.heard_from_stream(update.ts_recv_ns)
        floored = with_floor(order, update)
        priced = with_known_price(order.seen, floored)
        if not is_news(order.seen, priced):
            if update.source == "stream" and repeats(order.seen, floored):
                # Kept all the same: after a restart it tells that the stream spoke for the
                # order. A word that only loses information is not kept, since the
                # journal's last word is the latest to arrive.
                self._journal.record_update(floored)
                self._guard.check_update(floored)
            logger.debug("an update that is not news: %s", floored)
            return None
        # Whether the tradebook must confirm it is judged on the word as it came: a known
        # price stands for the engine's fills, but confirms no new word about them.
        if (
            floored is not update
            or (not checked and needs_trades(floored))
            or is_priceless(floored)
        ):
            self._hold(order, floored)
            return None
        return self._deliver(order, priced)

    def _hold(self, order: Followed, update: OrderUpdate) -> None:
        held = order.held
        if held is not None and not is_newer(held, update):
            return
        order.hold(update, self._now())
        logger.debug("order %s waits for the tradebook: %s", update.order_id, update)

    def _deliver(self, order: Followed, update: OrderUpdate) -> OrderUpdate:
        self._journal.record_update(update)
        order.seen, order.quiet_polls = update, 0
        held = order.held
        # A terminal held word (a cancel the tradebook must confirm) outlives a working one.
        if (
            held is not None
            and update.filled_quantity >= held.filled_quantity
            and (update.status.terminal or not held.status.terminal)
        ):
            order.release()
        self._guard.check_update(update)
        return update

    def _alert_overdue(self, now: int) -> None:
        for order_id, order in self._orders.items():
            held = order.held
            if held is None or order.alerted or now - order.held_since_ns < HELD_ALERT_AFTER_NS:
                continue
            order.alerted = True
            seconds = (now - order.held_since_ns) // NANOS_PER_SECOND
            what = f"order {order_id} ({held.action} {held.symbol} {held.exchange})"
            if held.filled_quantity:
                text = (
                    f"The price of {held.filled_quantity} units filled on {what} is not known "
                    f"after {seconds} seconds: neither the stream nor the tradebook gives it. "
                    "The engine holds the fill; the session can release it at the last price."
                )
            else:
                text = (
                    f"OpenAlgo reported {what} {held.status.value}, and the tradebook has not "
                    f"confirmed its fills after {seconds} seconds: the engine holds it."
                )
            logger.error(text)
            self._alerts.send(AlertKind.ERROR, text)

    def _keep_unclaimed(self, update: OrderUpdate) -> None:
        kept = self._unclaimed.get(update.order_id)
        if kept is not None and not is_newer(kept, update):
            return
        self._unclaimed.pop(update.order_id, None)
        self._unclaimed[update.order_id] = update
        while len(self._unclaimed) > MAX_UNCLAIMED_UPDATES:
            self._unclaimed.popitem(last=False)

    def _poll(self, order_ids: list[str], now: int) -> dict[str, Polled] | None:
        """The round's `orderstatus` words by order; None when an answer locked the
        guard, which drops the round."""
        polled: dict[str, Polled] = {}
        for order_id in order_ids:
            answer = self._read_status(order_id, now)
            if self._guard.locked:
                return None
            word = polled_word(answer, order_id, now)
            if word is not None:
                polled[order_id] = word
        return polled

    def _read_status(self, order_id: str, now: int) -> Mapping[str, Any]:
        """One `orderstatus` read of a followed order at `now`, counted as its poll."""
        order = self._orders[order_id]
        order.polled(now)
        return read_status(
            self._client,
            guard=self._guard,
            watch=self._watch,
            order_id=order_id,
            strategy=order.strategy,
            now=now,
        )


__all__ = [
    "HELD_ALERT_AFTER_NS",
    "MAX_POLLS_PER_ROUND",
    "MAX_UNCLAIMED_UPDATES",
    "OrderFollower",
]
