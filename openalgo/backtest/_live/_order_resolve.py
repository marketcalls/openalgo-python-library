"""Settling unbound intents against OpenAlgo's order book, for the order path (`orders`).

An intent is unbound when the answer to its `placeorder` was lost: its order may or may
not be at the broker, and nothing more is sent until the order book settles it. The one
order of its shape placed at or after it is its order (`_order_book`); none, read twice
and late enough for the order to have shown, means it was never placed. What the path
cannot settle alone is alerted once per intent. Times are UTC nanoseconds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._order_answers import server_answer
from ._order_book import BookMatch, Resolution, Resolved, book_rows, match_unbound
from ._order_follow import OrderFollower
from ._types import (
    LOGGER_NAME,
    AlertKind,
    AlertsLike,
    GuardLike,
    JournalLike,
    Now,
    OrderIntent,
    QueuedIntent,
)

logger = logging.getLogger(LOGGER_NAME)

#: An unbound intent is declared not placed only after this many order-book reads miss
#: it, once misses count: a broker's book can show a new order a few seconds late.
NOT_PLACED_MISSES = 2
#: When the server has finished with the request (it refused the connection, or answered
#: 500 or 502 in its own JSON), misses count from this long after its answer.
NOT_PLACED_AFTER_NS = 5 * NANOS_PER_SECOND
#: Otherwise the server may still be placing the order. It waits this long for each
#: phase of the broker's answer (`REQUEST_TIMEOUT_SECONDS`, 120, utils/httpx_client.py on
#: the server) ...
SERVER_BROKER_TIMEOUT_NS = 120 * NANOS_PER_SECOND
#: ... after the broker's pacing queue, which has no ceiling under eventlet
#: (utils/broker_backpressure.py): this margin covers an order queued behind others.
BROKER_QUEUE_MARGIN_NS = 30 * NANOS_PER_SECOND
#: So misses of such an intent count only from its send plus this window.
BROKER_WINDOW_NS = SERVER_BROKER_TIMEOUT_NS + BROKER_QUEUE_MARGIN_NS
#: The order book is read at most this often while an intent is unbound (the session
#: asks on every turn of its loop; OpenAlgo answers 10 reads a second).
BOOK_READ_EVERY_NS = 1 * NANOS_PER_SECOND


class UnboundResolver:
    """Settles the journal's unbound intents; see the module docstring. Every order-book
    answer goes to the guard, and nothing is read while it is locked."""

    def __init__(
        self,
        client: Any,
        *,
        journal: JournalLike,
        guard: GuardLike,
        alerts: AlertsLike,
        follower: OrderFollower,
        now: Now,
    ) -> None:
        self._client = client
        self._journal = journal
        self._guard = guard
        self._alerts = alerts
        self._follower = follower
        self._now = now
        self._misses_count_from_ns: dict[str, int] = {}
        self._misses: dict[str, int] = {}
        self._alerted: set[str] = set()
        self._book_read_ns: int | None = None

    def note_doubt(self, intent_id: str, *, misses_count_from_ns: int) -> None:
        """Count the order-book misses of an unbound intent from `misses_count_from_ns`
        on. An intent the path has no note of (left unbound before a restart) waits the
        whole `BROKER_WINDOW_NS` from its creation."""
        self._misses_count_from_ns[intent_id] = misses_count_from_ns

    def alert_queued(self, queued: QueuedIntent) -> None:
        """Alert an intent semi-auto queued before a restart, naming its pending order:
        approved, it would become a broker order the session does not follow."""
        pending = queued.pending_order_id or "(no pending order ID was given)"
        self._alert_once(
            queued.intent,
            f"was queued for approval in OpenAlgo's Action Center (semi-auto) as pending "
            f"order {pending} before the restart: approved, it becomes a broker order this "
            f"session does not follow, so reject pending order {pending} in the Action Center",
        )

    def resolve(self) -> list[Resolution]:
        """Read the order book, at most once a second, and settle what it can; see
        `OrderPath.resolve_unbound`."""
        intents = self._journal.unbound()
        now = self._now()
        if not intents or self._guard.locked:
            return []
        if self._book_read_ns is not None and now - self._book_read_ns < BOOK_READ_EVERY_NS:
            return []
        self._book_read_ns = now
        answer = server_answer(self._client.orderbook())
        self._guard.check_response(answer)
        if self._guard.locked:
            logger.error("the order book answered in the other mode: it settles no intent")
            return []
        rows = _rows_or_none(answer)
        if rows is None:
            for intent in intents:
                self._alert_stuck(
                    intent, f"the order book cannot be read ({answer.get('message')})"
                )
            return []
        taken = self._follower.order_ids() | {
            bound.order_id for bound in self._journal.bound_orders()
        }
        resolutions = []
        for intent in intents:
            resolution = self._settle(intent, match_unbound(intent, rows, taken), now)
            if resolution is None:
                continue
            if resolution.kind is Resolved.BOUND:
                taken.add(resolution.order_id)
            resolutions.append(resolution)
        return resolutions

    def _settle(self, intent: OrderIntent, match: BookMatch, now: int) -> Resolution | None:
        """The resolution one book read gives an unbound intent; None while it waits."""
        if match.unsure or len(match.orders) > 1:
            candidates = match.orders + match.unsure
            self._alert_stuck(intent, f"orders {', '.join(candidates)} may be its order")
            return Resolution(intent, Resolved.AMBIGUOUS, candidates=candidates)
        intent_id = intent.intent_id
        if match.orders:
            order_id = match.orders[0]
            self._journal.bind(intent_id, order_id)
            self._follower.follow(order_id, intent, sent_ns=intent.created_ns)
            logger.info("intent %s is order %s, found in the order book", intent_id, order_id)
            self._forget(intent_id)
            update = self._follower.claim(order_id)
            return Resolution(intent, Resolved.BOUND, order_id=order_id, update=update)
        counts_from_ns = self._misses_count_from_ns.get(
            intent_id, intent.created_ns + BROKER_WINDOW_NS
        )
        if now < counts_from_ns:
            return None
        self._misses[intent_id] = self._misses.get(intent_id, 0) + 1
        if self._misses[intent_id] < NOT_PLACED_MISSES:
            return None
        reason = "not in OpenAlgo's order book, read twice: the order never reached the broker"
        self._journal.mark_not_placed(intent_id, reason)
        logger.info("intent %s was not placed", intent_id)
        self._forget(intent_id)
        return Resolution(intent, Resolved.NOT_PLACED)

    def _forget(self, intent_id: str) -> None:
        """Drop what was kept of an intent that is settled."""
        self._misses_count_from_ns.pop(intent_id, None)
        self._misses.pop(intent_id, None)
        self._alerted.discard(intent_id)

    def _alert_stuck(self, intent: OrderIntent, why: str) -> None:
        """Alert an intent that stays unbound for a reason the path cannot settle."""
        self._alert_once(
            intent,
            f"may or may not be at the broker: {why}. Nothing more is sent until it is "
            "resolved; check the broker's order book, and stop the session with Ctrl+C if "
            "it cannot be settled",
        )

    def _alert_once(self, intent: OrderIntent, says: str) -> None:
        """One ERROR alert per intent, naming it, then `says`."""
        if intent.intent_id in self._alerted:
            return
        self._alerted.add(intent.intent_id)
        text = (
            f"Intent {intent.intent_id} ({intent.action} {intent.quantity} {intent.symbol} "
            f"{intent.exchange}) {says}."
        )
        logger.error(text)
        self._alerts.send(AlertKind.ERROR, text)


def _rows_or_none(answer: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """The rows of a successful order-book answer, or None when it cannot be read."""
    if answer.get("status") != "success":
        return None
    try:
        return book_rows(answer)
    except LiveError as error:
        logger.error("%s", error)
        return None


__all__ = [
    "BOOK_READ_EVERY_NS",
    "BROKER_QUEUE_MARGIN_NS",
    "BROKER_WINDOW_NS",
    "NOT_PLACED_AFTER_NS",
    "NOT_PLACED_MISSES",
    "SERVER_BROKER_TIMEOUT_NS",
    "UnboundResolver",
]
