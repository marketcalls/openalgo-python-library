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
from ._types import LOGGER_NAME, AlertKind, AlertsLike, GuardLike, JournalLike, Now, OrderIntent

logger = logging.getLogger(LOGGER_NAME)

#: An unbound intent is declared not placed only after this many order-book reads miss
#: it, the first of them at least `NOT_PLACED_AFTER_NS` after it was sent: a broker's
#: book can show a new order a few seconds late.
NOT_PLACED_MISSES = 2
NOT_PLACED_AFTER_NS = 5 * NANOS_PER_SECOND
#: After the client's own timeout the server may still be placing the order: it waits
#: this long for the broker (`REQUEST_TIMEOUT_SECONDS`, 120, utils/httpx_client.py on the
#: server), so misses count only from then. A 502 or a refused connection keeps 5 s.
SERVER_BROKER_TIMEOUT_NS = 120 * NANOS_PER_SECOND
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
        self._doubt_since_ns: dict[str, int] = {}
        self._misses: dict[str, int] = {}
        self._alerted: set[str] = set()
        self._book_read_ns: int | None = None

    def note_doubt(self, intent_id: str, *, since_ns: int) -> None:
        """Count the order-book misses of an unbound intent from `since_ns` on, plus
        `NOT_PLACED_AFTER_NS`."""
        self._doubt_since_ns[intent_id] = since_ns

    def alert_queued(self, intent: OrderIntent) -> None:
        """Alert an intent semi-auto queued before a restart: approved, it would become a
        broker order the session does not follow."""
        self._alert_once(
            intent,
            "was queued for approval in OpenAlgo's Action Center (semi-auto) before the "
            "restart: approved, it becomes a broker order this session does not follow, "
            "so reject it in the Action Center",
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
        # An intent left unbound before a restart may have timed out: wait as for one.
        since_ns = self._doubt_since_ns.get(intent_id, intent.created_ns + SERVER_BROKER_TIMEOUT_NS)
        if now - since_ns < NOT_PLACED_AFTER_NS:
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
        self._doubt_since_ns.pop(intent_id, None)
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
    "NOT_PLACED_AFTER_NS",
    "NOT_PLACED_MISSES",
    "SERVER_BROKER_TIMEOUT_NS",
    "UnboundResolver",
]
