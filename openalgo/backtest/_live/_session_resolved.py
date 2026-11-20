"""The order worker's resolutions of unbound placements, as the session loop takes them
(16-session-design 2.5 A.6; unit 8), split from `_session_news`: bound (with the
order's update, if any), not placed (a REJECT alert, an exit asked for again on the
schedule), or ambiguous (an ERROR alert once per intent, and the entries locked of the
strategy and of every other strategy of the session on its instrument: nothing more is
sent for the instrument, 18-multi-strategy 4.4); and how an alert names an order
(`order_words`).

Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._order_book import Resolved
from ._session_trader import Trader
from ._types import AlertKind, OrderIntent
from ._venue_requests import Applied
from .inbox import Resolutions

#: The entry lock of an order the order book cannot tell apart from another (6.2).
AMBIGUOUS_LOCK = "ambiguous_order"


def order_words(intent: OrderIntent) -> str:
    """How an alert names an order: "ema_cross: BUY 10 SBIN NSE (entry)"."""
    return (
        f"{intent.strategy}: {intent.action} {intent.quantity} {intent.symbol} "
        f"{intent.exchange} ({intent.purpose.value})"
    )


class ResolutionNews:
    """The handler of `Resolutions` of `_session_news.News`, which holds the session
    (`_session`), the intents alerted ambiguous (`_ambiguous_seen`) and `apply_update`."""

    _session: Any
    _ambiguous_seen: set[str]
    apply_update: Any

    def _resolutions(self, message: Resolutions) -> None:
        session = self._session
        for resolution in message.resolutions:
            intent = resolution.intent
            trader = session.trader_named(intent.strategy)
            handled = trader.venue.resolved(resolution)
            session.stops.resolved(trader, resolution)
            session.event(
                "resolved",
                strategy=trader.name,
                intent_id=intent.intent_id,
                resolution=resolution.kind.value,
                order_id=resolution.order_id,
                applied=handled.applied.value,
            )
            session.hand_over(list(handled.requests))
            if handled.applied is Applied.BOUND and resolution.update is not None:
                self.apply_update(trader, resolution.update)
            elif handled.applied is Applied.REJECTED:
                session.rejected += 1
                session.alert(
                    AlertKind.REJECT,
                    f"{order_words(intent)} order was not placed: the order book does not "
                    f"hold it (intent {intent.intent_id})",
                )
                kernel_order = trader.venue.kernel_order_of_intent(intent.intent_id)
                session.exits.rejected(trader, kernel_order, intent, "not placed")
            elif resolution.kind is Resolved.AMBIGUOUS:
                self._ambiguous(trader, intent, resolution.candidates)

    def _ambiguous(self, trader: Trader, intent: OrderIntent, candidates: tuple[str, ...]) -> None:
        """Lock the entries of the strategy and of the other strategies of its instrument,
        and alert, once per intent: the resolver reports the same intent at each book read
        while it stays unsettled."""
        if intent.intent_id in self._ambiguous_seen:
            return
        self._ambiguous_seen.add(intent.intent_id)
        session = self._session
        members = session.groups.members(trader)
        for member in members:
            if member.lock(AMBIGUOUS_LOCK):
                session.event("lock", strategy=member.name, reason=AMBIGUOUS_LOCK)
        others = [m.name for m in members if m is not trader]
        also = f" and those of {', '.join(others)}" if others else ""
        session.alert(
            AlertKind.ERROR,
            f"{order_words(intent)} order (intent {intent.intent_id}) cannot be told apart "
            f"in the order book from {', '.join(candidates) or 'another order'}: nothing more "
            f"is sent for {intent.symbol} and its entries{also} are locked. Check the order "
            "book in OpenAlgo",
        )


__all__ = ["AMBIGUOUS_LOCK", "ResolutionNews", "order_words"]
