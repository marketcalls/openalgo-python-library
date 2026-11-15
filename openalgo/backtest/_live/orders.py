"""The order path of a live session (08-live-trading section 6.1, the rate budget of
section 8): journal first, send once, resolve from the stream or by polling.

Every placement is written to the journal before its HTTP call and sent exactly once;
place, modify and cancel are never retried. An answer that leaves the outcome unknown (a
502, a timeout, a lost connection) leaves the intent unbound, and nothing more is sent
until `resolve_unbound` has found it in the order book or, after two reads that miss it,
marked it not placed. Order status comes from the order-updates stream, with polling as
the fallback (`_order_follow`).

Reading answers, updates and the order book lives in `_order_answers`, `_order_updates`,
`_order_news` and `_order_book`; their public names are exported here. Times are UTC
nanoseconds, prices rupees, quantities units. One loop owns an `OrderPath`: it is not
thread-safe, and the feed's thread only queues the updates it receives.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError, ModeChangedError
from ._order_answers import Outcome, SendOutcome, outcome_of, server_answer
from ._order_book import (
    CLOCK_SKEW_LIMIT_NS,
    BookMatch,
    Resolution,
    Resolved,
    book_rows,
    match_unbound,
)
from ._order_follow import (
    MAX_POLLS_PER_ROUND,
    POLL_AFTER_NS,
    POLL_EVERY_NS,
    SAFETY_POLL_EVERY_NS,
    OrderFollower,
)
from ._order_rate import MAX_ORDERS_PER_SECOND, OPENALGO_ORDERS_PER_SECOND, RateBudget
from ._order_updates import normalize_update
from ._types import (
    LOGGER_NAME,
    AlertKind,
    AlertsLike,
    GuardLike,
    JournalLike,
    Now,
    OrderIntent,
    OrderUpdate,
    Purpose,
    Sleep,
)

logger = logging.getLogger(LOGGER_NAME)

#: An unbound intent is declared not placed only after this many order-book reads miss
#: it, the first of them at least `NOT_PLACED_AFTER_NS` after it was sent: a broker's
#: book can show a new order a few seconds late.
NOT_PLACED_MISSES = 2
NOT_PLACED_AFTER_NS = 5 * NANOS_PER_SECOND
#: The order book is read at most this often while an intent is unbound (the session
#: asks on every turn of its loop; OpenAlgo answers 10 reads a second).
BOOK_READ_EVERY_NS = 1 * NANOS_PER_SECOND

#: The purposes `guard.before_entry` reads the mode for, just before they are sent.
_ENTRY_PURPOSES = frozenset({Purpose.ENTRY, Purpose.ADD})
#: The keyword arguments `modifyorder` takes besides the order ID and strategy.
_MODIFY_REQUIRED = frozenset(
    {"symbol", "action", "exchange", "price_type", "product", "quantity", "price", "trigger_price"}
)
_MODIFY_OPTIONAL = frozenset({"disclosed_quantity"})


class OrderPath:
    """Sends the session's orders through an OpenAlgo client, once each, and follows
    them until they are terminal.

    `client` is an `openalgo.api`; `journal`, `guard` and `alerts` the session's; `now`
    gives UTC nanoseconds and `sleep` waits seconds (the clock must move while it waits);
    `rate` is the account's `RateBudget`. The analyzer switch routes the order calls and
    the books alike, so the server's answers to `placeorder`, `modifyorder`,
    `cancelorder`, `orderstatus`, `orderbook` and `tradebook`, success or error, go to the
    guard (`server_answer` recovers them from the SDK's errors); market data carries no
    mode and is never read here.
    """

    def __init__(
        self,
        client: Any,
        *,
        journal: JournalLike,
        guard: GuardLike,
        alerts: AlertsLike,
        now: Now,
        sleep: Sleep,
        rate: RateBudget,
    ) -> None:
        self._client = client
        self._journal = journal
        self._guard = guard
        self._alerts = alerts
        self._now = now
        self._sleep = sleep
        self._rate = rate
        self._follower = OrderFollower(client, journal=journal, guard=guard, now=now)
        self._doubt_since_ns: dict[str, int] = {}
        self._misses: dict[str, int] = {}
        self._queued: set[str] = set()
        self._alerted: set[str] = set()
        self._book_read_ns: int | None = None

    def restore(self) -> None:
        """Take back the journal's bound orders after a restart: they can then be
        modified, cancelled and polled. Call it once the journal's day has started."""
        self._follower.restore()

    def send(self, intent: OrderIntent) -> SendOutcome:
        """Place `intent`'s order once: journal first, then one `placeorder` call.

        ACCEPTED binds the order ID in the journal; REJECTED marks the intent not
        placed; UNANSWERED and QUEUED leave it unbound and note it in doubt with the
        guard. An entry or add has its mode read after the rate-budget wait, just before
        the call.

        Raises ModeChangedError when the guard is locked or (for an entry or add)
        `guard.before_entry` finds the mode changed; LiveError when the mode cannot be
        read before an entry, or an intent is unbound; JournalError when the journal
        cannot be written. Nothing is sent then.
        """
        self._check_can_send(f"intent {intent.intent_id}")
        self._wait_for_budget()
        if intent.purpose in _ENTRY_PURPOSES:
            self._guard.before_entry()
        self._rate.take()
        self._journal.record_intent(intent)
        sent = self._client.placeorder(
            strategy=intent.strategy,
            symbol=intent.symbol,
            action=intent.action,
            exchange=intent.exchange,
            price_type=intent.price_type,
            product=intent.product,
            quantity=intent.quantity,
            price=intent.price,
            trigger_price=intent.trigger_price,
        )
        answer = server_answer(sent)
        outcome = outcome_of(answer)
        # The journal hears first, so the order's fate is kept even if a later step fails.
        if outcome.kind is Outcome.ACCEPTED:
            self._journal.bind(intent.intent_id, outcome.order_id)
        elif outcome.kind is Outcome.REJECTED:
            self._journal.mark_not_placed(intent.intent_id, outcome.reason)
        else:
            self._journal.mark_unbound(intent.intent_id, outcome.reason)
        self._guard.check_response(answer)
        if outcome.kind is Outcome.ACCEPTED:
            self._follower.follow(outcome.order_id, intent.strategy, sent_ns=self._now())
            logger.info("intent %s is order %s", intent.intent_id, outcome.order_id)
            return outcome
        if outcome.kind is not Outcome.REJECTED:
            self._doubt_since_ns[intent.intent_id] = self._now()
            if outcome.kind is Outcome.QUEUED:
                self._queued.add(intent.intent_id)
            self._guard.note_order_in_doubt(intent.intent_id)
        logger.warning("intent %s is %s: %s", intent.intent_id, outcome.kind.value, outcome)
        return outcome

    def modify(self, order_id: str, **fields: Any) -> SendOutcome:
        """Change an order of this session once, with `modifyorder`'s keyword arguments
        (symbol, action, exchange, price_type, product, quantity, price, trigger_price,
        and optionally disclosed_quantity). Raises as `send` does (no mode read), and
        LiveError for an order this session did not send or a missing or unknown field."""
        missing = _MODIFY_REQUIRED - fields.keys()
        unknown = fields.keys() - _MODIFY_REQUIRED - _MODIFY_OPTIONAL
        if missing or unknown:
            raise LiveError(
                f"modify takes {sorted(_MODIFY_REQUIRED)}: missing {sorted(missing)}, "
                f"unknown {sorted(unknown)}"
            )
        strategy = self._strategy(order_id)
        self._check_can_send(f"the change to order {order_id}")
        self._wait_for_budget()
        self._rate.take()
        sent = self._client.modifyorder(order_id=order_id, strategy=strategy, **fields)
        return self._settle_request(order_id, server_answer(sent))

    def cancel(self, order_id: str) -> SendOutcome:
        """Cancel an order of this session once. Raises as `modify` does."""
        strategy = self._strategy(order_id)
        self._check_can_send(f"the cancel of order {order_id}")
        self._wait_for_budget()
        self._rate.take()
        sent = self._client.cancelorder(order_id=order_id, strategy=strategy)
        return self._settle_request(order_id, server_answer(sent))

    def on_update(self, update: OrderUpdate) -> OrderUpdate | None:
        """Take an order update from the stream: returned, journaled and checked by the
        guard when it is news of an order of this session with its fills priced
        (`_order_news`); None for a duplicate, a stale update, a fill whose price is not
        known yet (held until it is), or an order this session has not bound (kept
        aside in case an unbound intent turns out to be it; otherwise ignored)."""
        return self._follower.on_update(update)

    def due_polls(self) -> list[OrderUpdate]:
        """The polling fallback (`_order_follow`): at most one round a second, polling
        `orderstatus` every second for an order with no word 2 seconds after it was sent
        and every 30 seconds for a resting one, with fills priced from the tradebook.
        Returns the news (`source` "poll"); nothing while the guard is locked."""
        return self._follower.due_polls()

    def resolve_unbound(self) -> list[Resolution]:
        """Look for the unbound intents in OpenAlgo's order book, at most once a second.

        An intent is bound to the one order of its shape placed at or after it (the
        Resolution carries the order's latest update when one arrived before), and never
        to an order another intent owns. It is marked not placed after two reads miss it,
        the first at least 5 seconds after it was sent; never when semi-auto queued it.
        Orders it cannot be told from (AMBIGUOUS), or a book that cannot be read, keep
        it unbound and send one ERROR alert per intent. Nothing is read while the guard
        is locked, and a book that locks it settles nothing.
        """
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

    def _check_can_send(self, what: str) -> None:
        if self._guard.locked:
            raise ModeChangedError(
                f"{what} is not sent: the mode guard is locked (OpenAlgo's analyzer switch "
                "changed, or semi-auto appeared), so the session sends nothing more; "
                "start it again in the mode you want"
            )
        unbound = self._journal.unbound()
        if unbound:
            raise LiveError(
                f"{what} is not sent: intent {unbound[0].intent_id} is unbound (its order "
                "may be at the broker), so call resolve_unbound() first"
            )

    def _wait_for_budget(self) -> None:
        while (wait := self._rate.wait_seconds()) > 0:
            self._sleep(wait)

    def _strategy(self, order_id: str) -> str:
        strategy = self._follower.strategy_of(order_id)
        if strategy is None:
            raise LiveError(
                f"order {order_id} was not sent by this session: it changes its own orders "
                "only (after a restart, call restore() first)"
            )
        return strategy

    def _settle_request(self, order_id: str, answer: Mapping[str, Any]) -> SendOutcome:
        outcome = outcome_of(answer, order_id)
        self._guard.check_response(answer)
        if outcome.kind is not Outcome.ACCEPTED:
            logger.warning("a request on order %s is %s: %s", order_id, outcome.kind.value, outcome)
        return outcome

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
            self._follower.follow(order_id, intent.strategy, sent_ns=intent.created_ns)
            logger.info("intent %s is order %s, found in the order book", intent_id, order_id)
            self._forget(intent_id)
            update = self._follower.claim(order_id)
            return Resolution(intent, Resolved.BOUND, order_id=order_id, update=update)
        since_ns = self._doubt_since_ns.get(intent_id, intent.created_ns)
        if intent_id in self._queued or now - since_ns < NOT_PLACED_AFTER_NS:
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
        """One ERROR alert per intent that stays unbound for a reason the path cannot
        settle by itself."""
        if intent.intent_id in self._alerted:
            return
        self._alerted.add(intent.intent_id)
        text = (
            f"Intent {intent.intent_id} ({intent.action} {intent.quantity} {intent.symbol} "
            f"{intent.exchange}) may or may not be at the broker: {why}. Nothing more is "
            "sent until it is resolved; check the broker's order book, and stop the "
            "session with Ctrl+C if it cannot be settled."
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
    "CLOCK_SKEW_LIMIT_NS",
    "MAX_ORDERS_PER_SECOND",
    "MAX_POLLS_PER_ROUND",
    "NOT_PLACED_AFTER_NS",
    "NOT_PLACED_MISSES",
    "OPENALGO_ORDERS_PER_SECOND",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "OrderPath",
    "Outcome",
    "RateBudget",
    "Resolution",
    "Resolved",
    "SendOutcome",
    "normalize_update",
]
