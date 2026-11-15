"""The order path of a live session (08-live-trading section 6.1, the rate budget of
section 8): journal first, send once, resolve from the stream or by polling.

Every placement is written to the journal before its HTTP call and sent exactly once;
place, modify and cancel are never retried. An answer that leaves the outcome unknown (a
502, a timeout, a lost connection) leaves the intent unbound, and nothing more is sent
until `resolve_unbound` has found it in the order book or marked it not placed. Order
status comes from the order-updates stream; an order with no terminal update 2 seconds
after it was sent is polled with `orderstatus` every second.

Reading answers, order updates and the order book lives in `_order_answers`,
`_order_updates` and `_order_book`; their public names are exported here. Times are UTC
nanoseconds, prices rupees, quantities units. One loop owns an `OrderPath`: it is not
thread-safe, and the feed's thread only queues the updates it receives.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError, ModeChangedError
from ._order_answers import Outcome, SendOutcome, outcome_of, server_answer
from ._order_book import (
    CLOCK_SKEW_LIMIT_NS,
    Resolution,
    Resolved,
    book_rows,
    match_unbound,
)
from ._order_updates import fills_given, normalize_update, number, units
from ._types import (
    LOGGER_NAME,
    GuardLike,
    JournalLike,
    Now,
    OrderIntent,
    OrderStatus,
    OrderUpdate,
    Purpose,
    Sleep,
)

logger = logging.getLogger(LOGGER_NAME)

#: OpenAlgo's order budget per client IP (`ORDER_RATE_LIMIT`, "10 per second", in the
#: server's restx_api/place_order.py); the engine counts place, modify and cancel together.
OPENALGO_ORDERS_PER_SECOND = 10
#: The engine uses at most half of it, leaving room for manual use (08 section 8).
MAX_ORDERS_PER_SECOND = OPENALGO_ORDERS_PER_SECOND // 2
#: An order with no terminal update this long after it was sent is polled (08 section 6.1).
POLL_AFTER_NS = 2 * NANOS_PER_SECOND
#: Then it is polled this often until a terminal update arrives (08 section 6.1).
POLL_EVERY_NS = 1 * NANOS_PER_SECOND

#: The purposes `guard.before_entry` reads the mode for, just before they are sent.
_ENTRY_PURPOSES = frozenset({Purpose.ENTRY, Purpose.ADD})
#: The keyword arguments `modifyorder` takes besides the order ID and strategy.
_MODIFY_REQUIRED = frozenset(
    {"symbol", "action", "exchange", "price_type", "product", "quantity", "price", "trigger_price"}
)
_MODIFY_OPTIONAL = frozenset({"disclosed_quantity"})


class RateBudget:
    """The order budget of the engine's process: at most `per_second` (1 to 5) place,
    modify and cancel requests in any one second, half of OpenAlgo's 10.

    `now` gives UTC nanoseconds. Raises LiveError for a `per_second` outside 1 to 5.
    """

    def __init__(self, now: Now, *, per_second: int = MAX_ORDERS_PER_SECOND) -> None:
        if not 1 <= per_second <= MAX_ORDERS_PER_SECOND:
            raise LiveError(
                f"per_second is {per_second!r}: the engine uses 1 to {MAX_ORDERS_PER_SECOND} "
                f"orders a second, at most half of OpenAlgo's {OPENALGO_ORDERS_PER_SECOND}"
            )
        self._now = now
        self._per_second = per_second
        self._taken_ns: collections.deque[int] = collections.deque()

    def wait_seconds(self) -> float:
        """Seconds until a request may be sent; 0.0 when one may be sent now."""
        now = self._now()
        while self._taken_ns and now - self._taken_ns[0] >= NANOS_PER_SECOND:
            self._taken_ns.popleft()
        if len(self._taken_ns) < self._per_second:
            return 0.0
        return (self._taken_ns[0] + NANOS_PER_SECOND - now) / NANOS_PER_SECOND

    def take(self) -> None:
        """Count a request sent now. Raises LiveError when the budget is spent: wait
        `wait_seconds()` first."""
        wait = self.wait_seconds()
        if wait > 0:
            raise LiveError(f"the order rate budget is spent for {wait:.3f} s more")
        self._taken_ns.append(self._now())


@dataclass
class _Working:
    """An order sent and not yet terminal: its strategy, when it was sent and polled."""

    strategy: str
    sent_ns: int
    polled_ns: int | None = None


class OrderPath:
    """Sends the session's orders through an OpenAlgo client, once each, and follows
    them until they are terminal.

    `client` is an `openalgo.api`; `journal` and `guard` the session's journal and mode
    guard; `now` gives UTC nanoseconds and `sleep` waits seconds (the clock must move
    while it waits); `rate` is the account's `RateBudget`. The server's answers to
    `placeorder`, `modifyorder`, `cancelorder` and `orderstatus`, success or error, go to
    the guard (`server_answer` recovers them from the SDK's errors), and no other answer
    does: a market-data answer carries no mode, which the guard would read as live.
    """

    def __init__(
        self,
        client: Any,
        *,
        journal: JournalLike,
        guard: GuardLike,
        now: Now,
        sleep: Sleep,
        rate: RateBudget,
    ) -> None:
        self._client = client
        self._journal = journal
        self._guard = guard
        self._now = now
        self._sleep = sleep
        self._rate = rate
        self._strategy_of: dict[str, str] = {}
        self._working: dict[str, _Working] = {}
        self._seen: dict[str, tuple[OrderStatus, int]] = {}

    def send(self, intent: OrderIntent) -> SendOutcome:
        """Place `intent`'s order once: journal first, then one `placeorder` call.

        ACCEPTED binds the order ID in the journal; REJECTED marks the intent not
        placed; UNANSWERED and QUEUED leave it unbound and note it in doubt with the
        guard. The guard sees the server's own answer (`server_answer`).

        Raises ModeChangedError when the guard is locked or (for an entry or add)
        `guard.before_entry` finds the mode changed; LiveError when the mode cannot be
        read before an entry, or an intent is unbound; JournalError when the journal
        cannot be written. Nothing is sent then.
        """
        self._check_can_send(f"intent {intent.intent_id}")
        if intent.purpose in _ENTRY_PURPOSES:
            self._guard.before_entry()
        self._wait_for_budget()
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
            self._follow(intent, outcome.order_id, sent_ns=self._now())
            return outcome
        if outcome.kind is not Outcome.REJECTED:
            self._guard.note_order_in_doubt(intent.intent_id)
        logger.warning("intent %s is %s: %s", intent.intent_id, outcome.kind.value, outcome)
        return outcome

    def modify(self, order_id: str, **fields: Any) -> SendOutcome:
        """Change an order of this path once, with `modifyorder`'s keyword arguments
        (symbol, action, exchange, price_type, product, quantity, price, trigger_price,
        and optionally disclosed_quantity). Raises as `send` does (no mode read), and
        LiveError for an order this path did not send or a missing or unknown field."""
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
        sent = self._client.modifyorder(order_id=order_id, strategy=strategy, **fields)
        return self._settle_request(order_id, server_answer(sent))

    def cancel(self, order_id: str) -> SendOutcome:
        """Cancel an order of this path once. Raises as `modify` does."""
        strategy = self._strategy(order_id)
        self._check_can_send(f"the cancel of order {order_id}")
        self._wait_for_budget()
        sent = self._client.cancelorder(order_id=order_id, strategy=strategy)
        return self._settle_request(order_id, server_answer(sent))

    def on_update(self, update: OrderUpdate) -> OrderUpdate | None:
        """Take an order update from the stream or a poll: None for a duplicate (the
        same order, status and filled quantity) or a stale one (a lower filled quantity
        than seen, or any change after a terminal status); otherwise it is written to
        the journal, its mode checked by the guard, and returned."""
        seen = self._seen.get(update.order_id)
        if seen is not None:
            status, filled = seen
            if (
                status.terminal
                or update.filled_quantity < filled
                or (update.status is status and update.filled_quantity == filled)
            ):
                logger.debug("dropped a duplicate or stale update: %s", update)
                return None
        self._journal.record_update(update)
        self._seen[update.order_id] = (update.status, update.filled_quantity)
        if update.status.terminal:
            self._working.pop(update.order_id, None)
        self._guard.check_update(update)
        return update

    def due_polls(self) -> list[OrderUpdate]:
        """Poll `orderstatus` for each order with no terminal update 2 seconds after it
        was sent, once a second, and return the new updates (`source` "poll"): the
        fallback when the stream is slow or disabled. An open or cancelled order whose
        status omits its fills has them read from the tradebook. A failed read is logged
        and made again a second later. Raises LiveError for an answer that cannot be
        read (`normalize_update`)."""
        now = self._now()
        new: list[OrderUpdate] = []
        for order_id, working in list(self._working.items()):
            if now - working.sent_ns < POLL_AFTER_NS:
                continue
            if working.polled_ns is not None and now - working.polled_ns < POLL_EVERY_NS:
                continue
            working.polled_ns = now
            polled = self._poll(order_id, working.strategy, now)
            update = None if polled is None else self.on_update(polled)
            if update is not None:
                new.append(update)
        return new

    def resolve_unbound(self) -> list[Resolution]:
        """Look for each unbound intent in OpenAlgo's order book (`_order_book`): bind it
        to the one order of its shape placed at or after it, or mark it not placed when
        there is none; several matches leave it unbound (AMBIGUOUS). An order book that
        cannot be read is logged and resolves nothing. Raises LiveError for a matching
        row whose quantity or time cannot be read, and then changes nothing."""
        intents = self._journal.unbound()
        if not intents:
            return []
        answer = self._client.orderbook()
        if answer.get("status") != "success":
            logger.warning("the order book could not be read: %s", answer.get("message"))
            return []
        rows = book_rows(answer)
        taken = set(self._strategy_of)
        resolutions = []
        for intent in intents:
            resolution = match_unbound(intent, rows, taken)
            if resolution.kind is Resolved.BOUND:
                taken.add(resolution.order_id)
            resolutions.append(resolution)
        for resolution in resolutions:
            self._apply(resolution)
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
        self._rate.take()

    def _strategy(self, order_id: str) -> str:
        strategy = self._strategy_of.get(order_id)
        if strategy is None:
            raise LiveError(
                f"order {order_id} was not sent by this session: it changes its own orders only"
            )
        return strategy

    def _follow(self, intent: OrderIntent, order_id: str, *, sent_ns: int) -> None:
        """Follow an order bound in the journal: noted with the guard, and polled while it
        is not terminal."""
        self._guard.note_order_sent(order_id)
        self._strategy_of[order_id] = intent.strategy
        seen = self._seen.get(order_id)
        if seen is None or not seen[0].terminal:
            self._working[order_id] = _Working(intent.strategy, sent_ns)
        logger.info("intent %s is order %s", intent.intent_id, order_id)

    def _settle_request(self, order_id: str, answer: Mapping[str, Any]) -> SendOutcome:
        outcome = outcome_of(answer, order_id)
        self._guard.check_response(answer)
        if outcome.kind is not Outcome.ACCEPTED:
            logger.warning("a request on order %s is %s: %s", order_id, outcome.kind.value, outcome)
        return outcome

    def _poll(self, order_id: str, strategy: str, now: int) -> OrderUpdate | None:
        answer = server_answer(self._client.orderstatus(order_id=order_id, strategy=strategy))
        self._guard.check_response(answer)
        data = answer.get("data")
        if answer.get("status") != "success" or not isinstance(data, Mapping):
            logger.warning("orderstatus of %s failed: %s", order_id, answer.get("message"))
            return None
        if not fills_given(data, order_id):
            fills = self._fills(order_id)
            if fills is None:
                return None
            answer = {**answer, "data": {**data, **fills}}
        return normalize_update(answer, ts_recv_ns=now, source="poll")

    def _fills(self, order_id: str) -> dict[str, int | float] | None:
        """An order's filled quantity and average price from the tradebook, for the
        brokers whose order book omits them."""
        answer = self._client.tradebook()
        trades = answer.get("data")
        if answer.get("status") != "success" or not isinstance(trades, list):
            logger.warning("the tradebook could not be read: %s", answer.get("message"))
            return None
        own = [trade for trade in trades if str(trade.get("orderid")) == order_id]
        filled = sum(units(trade, "quantity", order_id) for trade in own)
        value = sum(
            units(trade, "quantity", order_id) * number(trade, "average_price", order_id)
            for trade in own
        )
        return {"filled_quantity": filled, "average_price": value / filled if filled else 0.0}

    def _apply(self, resolution: Resolution) -> None:
        intent = resolution.intent
        if resolution.kind is Resolved.BOUND:
            self._journal.bind(intent.intent_id, resolution.order_id)
            self._follow(intent, resolution.order_id, sent_ns=intent.created_ns)
        elif resolution.kind is Resolved.NOT_PLACED:
            reason = "not in OpenAlgo's order book: the order never reached the broker"
            self._journal.mark_not_placed(intent.intent_id, reason)
            logger.info("intent %s was not placed", intent.intent_id)
        else:
            logger.error(
                "intent %s matches orders %s in the order book: it stays unbound until the "
                "operator settles it",
                intent.intent_id,
                ", ".join(resolution.candidates),
            )


__all__ = [
    "CLOCK_SKEW_LIMIT_NS",
    "MAX_ORDERS_PER_SECOND",
    "OPENALGO_ORDERS_PER_SECOND",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "OrderPath",
    "Outcome",
    "RateBudget",
    "Resolution",
    "Resolved",
    "SendOutcome",
    "normalize_update",
]
