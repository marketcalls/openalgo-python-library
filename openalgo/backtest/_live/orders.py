"""The order path of a live session (08-live-trading section 6.1, the rate budget of
section 8): journal first, send once, resolve from the stream or by polling.

Every placement is written to the journal before its HTTP call and sent exactly once;
place, modify and cancel are never retried. An answer that leaves the outcome unknown (a
502, a timeout, a lost connection) leaves the intent unbound, and nothing more is sent
for its symbol, exchange and product until `resolve_unbound` has found it in the order
book or, after two reads that miss it, marked it not placed (`_order_resolve`). After a
client timeout or a dropped connection nothing is sent for any instrument until
OpenAlgo answers a call again (`ServerWatch`). Order status comes from the order-updates
stream, with polling as the fallback and fills confirmed from the tradebook
(`_order_follow`, `_order_state`, `_order_trades`).

Reading answers, updates, the order book and one order's row lives in `_order_answers`,
`_order_updates`, `_order_news`, `_order_book` and `_order_check`, the budget in
`_order_rate`; their public names are exported here. Times are UTC nanoseconds, prices
rupees, quantities units. One loop owns an `OrderPath`: it is not thread-safe, and the
feed's thread only queues the updates it receives.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..errors import LiveError, ModeChangedError
from ._answers import server_answer
from ._order_answers import (
    Outcome,
    SendOutcome,
    ServerWatch,
    outcome_of,
    server_has_finished,
)
from ._order_book import Resolution, Resolved
from ._order_check import OrderCheck
from ._order_follow import OrderFollower
from ._order_rate import RateBudget
from ._order_resolve import BROKER_WINDOW_NS, NOT_PLACED_AFTER_NS, UnboundResolver, same_leg
from ._order_state import ENTRY_PURPOSES, HeldFill
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
    Sleep,
)

logger = logging.getLogger(LOGGER_NAME)

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
        self._watch = ServerWatch()
        self._follower = OrderFollower(
            client, journal=journal, guard=guard, alerts=alerts, watch=self._watch, now=now
        )
        self._resolver = UnboundResolver(
            client,
            journal=journal,
            guard=guard,
            alerts=alerts,
            follower=self._follower,
            watch=self._watch,
            now=now,
        )

    def restore(self) -> None:
        """Take back the journal's bound orders after a restart: they can then be
        modified, cancelled and polled. An intent semi-auto queued before the restart
        sends one ERROR alert: approved, it would become an order nobody follows.

        Call it after `guard.start()` and `journal.start_session(...)`, in that order,
        even with nothing to take back. Raises LiveError (and takes back nothing) before
        the guard has started, and JournalError before the journal's session has.
        """
        if not self._guard.started:
            raise LiveError(
                "restore() takes back orders the mode guard must judge, and the guard has "
                "not read OpenAlgo's mode yet: call guard.start(), then "
                "journal.start_session(...), then restore()"
            )
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
        read before an entry, an intent of the same symbol, exchange and product is
        unbound, or OpenAlgo has not answered since a client timeout or a dropped
        connection; JournalError when the journal cannot be written. Nothing is sent then.
        """
        self._check_can_send(f"intent {intent.intent_id}", intent)
        self._wait_for_budget()
        if intent.purpose in ENTRY_PURPOSES:
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
        self._watch.note(answer, self._now())
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
        and optionally disclosed_quantity). Raises as `send` does (no mode read, and
        the hold is the order's instrument's), and LiveError for an order this session
        did not send or a missing or unknown field."""
        missing = _MODIFY_REQUIRED - fields.keys()
        unknown = fields.keys() - _MODIFY_REQUIRED - _MODIFY_OPTIONAL
        if missing or unknown:
            raise LiveError(
                f"modify takes {sorted(_MODIFY_REQUIRED)}: missing {sorted(missing)}, "
                f"unknown {sorted(unknown)}"
            )
        sent_for = self._intent_of(order_id)
        self._check_can_send(f"the change to order {order_id}", sent_for)
        self._wait_for_budget()
        self._rate.take()
        sent = self._client.modifyorder(order_id=order_id, strategy=sent_for.strategy, **fields)
        outcome = self._settle_request(order_id, server_answer(sent))
        if outcome.kind is Outcome.ACCEPTED:
            self._follower.changed_price_type(order_id, str(fields["price_type"]))
        return outcome

    def cancel(self, order_id: str) -> SendOutcome:
        """Cancel an order of this session once. Raises as `modify` does."""
        sent_for = self._intent_of(order_id)
        self._check_can_send(f"the cancel of order {order_id}", sent_for)
        self._wait_for_budget()
        self._rate.take()
        sent = self._client.cancelorder(order_id=order_id, strategy=sent_for.strategy)
        return self._settle_request(order_id, server_answer(sent))

    def on_update(self, update: OrderUpdate) -> OrderUpdate | None:
        """Take an order update from the stream: returned with its status as reported,
        journaled and checked by the guard, when it is news of an order of this session
        with its fills priced (`_order_news`); None for a duplicate (a word that repeats
        exactly what is known is journaled, so a restart knows the stream spoke), a stale
        update, a fill whose price or a cancel whose fills the tradebook has not
        confirmed yet (held until it does), or an order this session has not bound (kept
        aside in case an unbound intent turns out to be it; otherwise ignored)."""
        return self._follower.on_update(update)

    def due_polls(self) -> list[OrderUpdate]:
        """The polling fallback (`_order_follow`), in rounds of at most one a second and
        three orders: `orderstatus` every second for an order with no word from 2 seconds
        after it was sent, or a market order still working; a resting order every 30
        seconds once the stream has spoken for it, else backing off from 1 s to 2 s to
        5 s (2 s for an entry or an add). Fills are priced, and cancels confirmed, from
        the tradebook, read at most once a round with its own back-off; a hold past 10
        seconds sends one ERROR alert. Returns the news; nothing while the guard is
        locked."""
        return self._follower.due_polls()

    def held_fills(self) -> list[HeldFill]:
        """The order updates held for their price or the tradebook's confirmation, with
        their units and since when (UTC nanoseconds)."""
        return self._follower.held_fills()

    def release_fill(self, order_id: str, average_price: float) -> OrderUpdate | None:
        """Hand over a held fill at `average_price` (rupees, such as the feed's last
        traded price; a price already known for its units stands), journaled and checked
        by the guard; None when it is no longer news. Raises LiveError when the order
        holds nothing or the price is not positive."""
        return self._follower.release_fill(order_id, average_price)

    def release_cancel(
        self, order_id: str, filled_quantity: int, average_price: float
    ) -> OrderUpdate | None:
        """Hand over a held cancel (CANCELLED, its fills not confirmed by the tradebook)
        as having filled `filled_quantity` units in all before it ended (cumulative, as
        every update counts them: those already handed over included), at `average_price`
        rupees, the average of all of them (an estimate, unused for 0 units; a price
        already known for that many units stands), journaled and checked by the guard;
        None when it is no longer news. The session finds the units from the position
        book. Raises LiveError when the order holds no cancel, the units are not whole,
        fewer than those already reported filled or more than the order's, or a fill's
        price is not positive."""
        return self._follower.release_cancel(order_id, filled_quantity, average_price)

    def last_known(self, order_id: str) -> OrderUpdate | None:
        """The latest word of an order of this session, as reported: the one held for the
        tradebook when there is one, else the last handed over. None before any word, and
        for an order this session did not send. A terminal one means a modify or a cancel
        of it has nothing left to change."""
        return self._follower.last_known(order_id)

    def check_order(self, order_id: str) -> OrderCheck:
        """Read one order of this session with one `orderstatus` call, through the guard
        and the server watch (an answer in OpenAlgo's own JSON lifts the hold of a silent
        server), whatever its polling schedule: the row's status, quantity, filled
        quantity when given, price, trigger price and price type (`OrderCheck`), or why
        it could not be read. Its news is taken as a poll's and returned in the check: a
        fill or a cancel waits for the tradebook, which the next polling round reads. The
        row's price type becomes the order's for its polling (a modify whose answer was
        lost may have changed it).

        Raises ModeChangedError when the guard is locked, before the read or by its
        answer (a row of the other mode's book says nothing of this order), and LiveError
        for an order this session did not send.
        """
        return self._follower.check(order_id)

    def stream_gap(self, since_ns: int) -> None:
        """The order-updates stream may have missed words since `since_ns` (UTC
        nanoseconds: a feed outage's start, or the journal snapshot's time at a restart):
        every followed order still working that the stream has not spoken for since then
        is polled once, in the next rounds of `due_polls` (at most three orders a round,
        one round a second). Nothing is read here."""
        self._follower.stream_gap(since_ns)

    @property
    def silent_since_ns(self) -> int | None:
        """Since when (UTC nanoseconds) OpenAlgo has not answered a call after a client
        timeout or a dropped connection, during which nothing is sent; None while it
        answers."""
        return self._watch.silent_since_ns

    def probe(self) -> bool:
        """Ask OpenAlgo whether it answers again with one `orderbook` read, its answer
        checked by the guard: an answer in OpenAlgo's own JSON (a success, or its own
        error) lifts the hold on sending of a silent server; a timeout, a dropped
        connection or a page from a gateway in front of it does not. Nothing is read while
        the guard is locked. Returns True when OpenAlgo is not silent afterwards."""
        if not self._guard.locked:
            answer = server_answer(self._client.orderbook())
            self._watch.note(answer, self._now())
            self._guard.check_response(answer)
        return not self._watch.silent

    def resolve_unbound(self) -> list[Resolution]:
        """Look for the unbound intents in OpenAlgo's order book, at most once a second.

        An intent is bound to the one order of its shape placed at or after it (the
        Resolution carries the order's latest update when one arrived before), and never
        to an order another intent owns. It is marked not placed after two reads miss it,
        counted from 5 seconds after the answer when the server had finished with the
        request (`server_has_finished`), else from its send plus `BROKER_WINDOW_NS` (150
        seconds; from its creation for an intent left unbound before a restart). A queued
        intent (semi-auto) is not unbound and is never touched. While an intent is
        unbound nothing is sent for its symbol, exchange and product. Orders it cannot be
        told from (AMBIGUOUS), or a book that cannot be read, keep it unbound and send one
        ERROR alert per intent. Nothing is read while the guard is locked, and a book
        that locks it settles nothing.
        """
        return self._resolver.resolve()

    def _check_can_send(self, what: str, sent_for: OrderIntent) -> None:
        if self._guard.locked:
            raise ModeChangedError(
                f"{what} is not sent: the mode guard is locked (OpenAlgo's analyzer switch "
                "changed, or semi-auto appeared), so the session sends nothing more; "
                "start it again in the mode you want"
            )
        if self._watch.silent:
            raise LiveError(
                f"{what} is not sent: OpenAlgo has not answered since a client timeout or a "
                "dropped connection, so it may be hung and still placing what it was sent; "
                "nothing is sent for any instrument until it answers a call again (the polls "
                "and the order-book reads ask it)"
            )
        held = [intent for intent in self._journal.unbound() if same_leg(intent, sent_for)]
        if held:
            raise LiveError(
                f"{what} is not sent: intent {held[0].intent_id} of {sent_for.symbol} on "
                f"{sent_for.exchange} ({sent_for.product}) is unbound (its order may be at "
                "the broker), so nothing is sent for that instrument until resolve_unbound() "
                "settles it"
            )

    def _wait_for_budget(self) -> None:
        while (wait := self._rate.wait_seconds()) > 0:
            self._sleep(wait)

    def _intent_of(self, order_id: str) -> OrderIntent:
        intent = self._follower.intent_of(order_id)
        if intent is None:
            raise LiveError(
                f"order {order_id} was not sent by this session: it changes its own orders "
                "only (after a restart, call restore() first)"
            )
        return intent

    def _settle_request(self, order_id: str, answer: Mapping[str, Any]) -> SendOutcome:
        self._watch.note(answer, self._now())
        outcome = outcome_of(answer, order_id)
        self._guard.check_response(answer)
        if outcome.kind is not Outcome.ACCEPTED:
            logger.warning("a request on order %s is %s: %s", order_id, outcome.kind.value, outcome)
        return outcome


__all__ = [
    "BROKER_WINDOW_NS",
    "UNSUPPORTED_BROKERS",
    "HeldFill",
    "OrderCheck",
    "OrderPath",
    "Outcome",
    "RateBudget",
    "Resolution",
    "Resolved",
    "SendOutcome",
    "normalize_update",
]
