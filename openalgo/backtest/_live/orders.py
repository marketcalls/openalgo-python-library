"""The order path of a live session (08-live-trading section 6.1, the rate budget of
section 8): journal first, send once, resolve from the stream or by polling.

Every placement is written to the journal before its HTTP call and sent exactly once;
place, modify and cancel are never retried. An answer that leaves the outcome unknown (a
502, a timeout, a lost connection) leaves the intent unbound, and nothing more is sent
until `resolve_unbound` has found it in the order book or, after two reads that miss it,
marked it not placed (`_order_resolve`). Order status comes from the order-updates
stream, with polling as the fallback and fills confirmed from the tradebook
(`_order_follow`, `_order_state`, `_order_trades`).

Reading answers, updates and the order book lives in `_order_answers`, `_order_updates`,
`_order_news` and `_order_book`, the budget in `_order_rate`; their public names are
exported here. Times are UTC nanoseconds, prices rupees, quantities units. One loop owns
an `OrderPath`: it is not thread-safe, and the feed's thread only queues the updates it
receives.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..errors import LiveError, ModeChangedError
from ._order_answers import (
    Outcome,
    SendOutcome,
    outcome_of,
    server_answer,
    server_has_finished,
)
from ._order_book import CLOCK_SKEW_LIMIT_NS, Resolution, Resolved
from ._order_follow import HELD_ALERT_AFTER_NS, MAX_POLLS_PER_ROUND, OrderFollower
from ._order_rate import MAX_ORDERS_PER_SECOND, OPENALGO_ORDERS_PER_SECOND, RateBudget
from ._order_resolve import (
    BOOK_READ_EVERY_NS,
    BROKER_QUEUE_MARGIN_NS,
    BROKER_WINDOW_NS,
    NOT_PLACED_AFTER_NS,
    NOT_PLACED_MISSES,
    SERVER_BROKER_TIMEOUT_NS,
    UnboundResolver,
)
from ._order_state import POLL_AFTER_NS, POLL_EVERY_NS, SAFETY_POLL_EVERY_NS, HeldFill
from ._order_trades import UNSUPPORTED_BROKERS
from ._order_updates import normalize_update
from ._types import (
    LOGGER_NAME,
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
        self._follower = OrderFollower(client, journal=journal, guard=guard, alerts=alerts, now=now)
        self._resolver = UnboundResolver(
            client, journal=journal, guard=guard, alerts=alerts, follower=self._follower, now=now
        )

    def restore(self) -> None:
        """Take back the journal's bound orders after a restart: they can then be
        modified, cancelled and polled. An intent semi-auto queued before the restart
        sends one ERROR alert: approved, it would become an order nobody follows.

        Call it after `guard.start()` and `journal.start_session(...)`, in that order,
        even with nothing to take back. Raises LiveError (and takes back nothing) before
        the guard has started, and JournalError before the journal's session has.
        """
        try:
            # GuardLike tells nothing of having started: an answer that carries nothing
            # asks it, and the guard raises LiveError before start() (guard.py).
            self._guard.check_response({})
        except LiveError as error:
            raise LiveError(
                "restore() takes back orders the mode guard must judge: call guard.start(), "
                f"then journal.start_session(...), then restore() ({error})"
            ) from error
        self._follower.restore(self._journal.bound_orders())
        for queued in self._journal.queued():
            self._resolver.alert_queued(queued)

    def send(self, intent: OrderIntent) -> SendOutcome:
        """Place `intent`'s order once: journal first, then one `placeorder` call.

        ACCEPTED binds the order ID in the journal; REJECTED marks the intent not
        placed; UNANSWERED leaves it unbound and QUEUED marks it queued, both noted in
        doubt with the guard. An entry or add has its mode read after the rate-budget
        wait, just before the call.

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
        sent_ns = self._now()
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
        elif outcome.kind is Outcome.QUEUED:
            self._journal.mark_queued(intent.intent_id, outcome.pending_order_id)
        else:
            self._journal.mark_unbound(intent.intent_id, outcome.reason)
        self._guard.check_response(answer)
        if outcome.kind is Outcome.ACCEPTED:
            self._follower.follow(outcome.order_id, intent, sent_ns=self._now())
            logger.info("intent %s is order %s", intent.intent_id, outcome.order_id)
            return outcome
        if outcome.kind is not Outcome.REJECTED:
            if server_has_finished(answer):
                counts_from_ns = self._now() + NOT_PLACED_AFTER_NS
            else:
                counts_from_ns = sent_ns + BROKER_WINDOW_NS
            self._resolver.note_doubt(intent.intent_id, misses_count_from_ns=counts_from_ns)
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
        outcome = self._settle_request(order_id, server_answer(sent))
        if outcome.kind is Outcome.ACCEPTED:
            self._follower.changed_price_type(order_id, str(fields["price_type"]))
        return outcome

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
        or working but not resting, and every 30 seconds for a resting one the stream
        has spoken for, with fills priced and cancels confirmed from the tradebook. A
        hold past 10 seconds sends one ERROR alert. Returns the news; nothing while the
        guard is locked."""
        return self._follower.due_polls()

    def held_fills(self) -> list[HeldFill]:
        """The order updates held for their price or the tradebook's confirmation, with
        their units and since when (UTC nanoseconds)."""
        return self._follower.held_fills()

    def release_fill(self, order_id: str, average_price: float) -> OrderUpdate | None:
        """Hand over a held fill at `average_price` (rupees, such as the feed's last
        traded price), journaled and checked by the guard; None when it is no longer
        news. Raises LiveError when the order holds nothing or the price is not positive."""
        return self._follower.release_fill(order_id, average_price)

    def resolve_unbound(self) -> list[Resolution]:
        """Look for the unbound intents in OpenAlgo's order book, at most once a second.

        An intent is bound to the one order of its shape placed at or after it (the
        Resolution carries the order's latest update when one arrived before), and never
        to an order another intent owns. It is marked not placed after two reads miss it,
        the first at least 5 seconds after it was sent (after a client timeout, or for an
        intent left unbound before a restart, 5 seconds after the server's own broker
        timeout). A queued intent (semi-auto) is not unbound and is never touched.
        Orders it cannot be told from (AMBIGUOUS), or a book that cannot be read, keep
        it unbound and send one ERROR alert per intent. Nothing is read while the guard
        is locked, and a book that locks it settles nothing.
        """
        return self._resolver.resolve()

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
        if outcome.kind in (Outcome.ACCEPTED, Outcome.UNANSWERED):
            # The broker may report the change itself as open (a stop being modified or
            # cancelled): until the stream says the stop waits again, that is no trigger.
            self._follower.changed_by_session(order_id)
        if outcome.kind is not Outcome.ACCEPTED:
            logger.warning("a request on order %s is %s: %s", order_id, outcome.kind.value, outcome)
        return outcome


__all__ = [
    "BOOK_READ_EVERY_NS",
    "BROKER_QUEUE_MARGIN_NS",
    "BROKER_WINDOW_NS",
    "CLOCK_SKEW_LIMIT_NS",
    "HELD_ALERT_AFTER_NS",
    "MAX_ORDERS_PER_SECOND",
    "MAX_POLLS_PER_ROUND",
    "NOT_PLACED_AFTER_NS",
    "NOT_PLACED_MISSES",
    "OPENALGO_ORDERS_PER_SECOND",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "SERVER_BROKER_TIMEOUT_NS",
    "UNSUPPORTED_BROKERS",
    "HeldFill",
    "OrderPath",
    "Outcome",
    "RateBudget",
    "Resolution",
    "Resolved",
    "SendOutcome",
    "normalize_update",
]
