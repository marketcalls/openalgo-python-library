"""One order update as the session loop applies it (16-session-design 2.5 A.1; units 8 and
10), split from `_session_news` so each module keeps one job: through the order's venue map
and fill ledger to the kernel, then the stops supervisor; ENTRY, EXIT and REJECT alerts
with the order IDs, an EXIT alert with the order's exit reason (`_exit_reasons`), an ENTRY
or EXIT alert of a session of several strategies with the strategy's name after its kind;
a fill the kernel refuses is a reconciliation mismatch: the strategy's entries are locked
and a RECONCILE_MISMATCH alert sent.

A closing order rejected on an instrument several strategies trade, by its REJECTED word,
is taken by `_margin_reject` first (its word's fills before it): the venue map then takes
the word, whose own rejection the kernel refuses as the race of an order ended already.
Times are UTC nanoseconds, prices rupees, quantities units. Used by the loop only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ._exit_reasons import exit_reason, exit_words
from ._fill_ledger import FillOutcome, FillResult, Increment
from ._order_state import ENTRY_PURPOSES
from ._session_records import fill_record
from ._session_resolved import ResolutionNews, order_words
from ._session_trader import Trader
from ._types import AlertKind, OrderIntent, OrderStatus, OrderUpdate

#: The entry lock a fill the kernel refused sets on its strategy (16-session-design 6.2,
#: decision 2: lock and alert, nothing sent to fix it).
RECONCILE_LOCK = "reconciliation"


class FillNews(ResolutionNews):
    """The handling of one order update (module docstring) of `_session_news.News`, which
    holds the session (`_session`)."""

    _session: Any

    def apply_update(
        self, trader: Trader, update: OrderUpdate, flags: tuple[str, ...] = ()
    ) -> FillResult:
        """Give one order update to the kernel through `trader`'s venue map, its fills with
        the venue's `flags` (a GTT child's fill missed, `_session_gtts`), then send the
        alerts it calls for and note it in the event stream."""
        session = self._session
        kernel_order = trader.venue.kernel_order_of(update.order_id)
        if update.status is OrderStatus.REJECTED and kernel_order is not None:
            self._closing_rejected(trader, kernel_order, update)
        before = trader.open_units()
        result = trader.venue.news(update, flags)
        if result.increment is not None:
            trader.records.filled(before, trader.open_units())
        session.stops.updated(trader, update, result)
        intent = trader.venue.intent_of(result.kernel_order)
        session.event(
            "news",
            strategy=trader.name,
            order_id=update.order_id,
            kernel_order=kernel_order,
            status=update.status.value,
            filled=update.filled_quantity,
            average_price=update.average_price,
            source=update.source,
            outcome=result.outcome.value,
        )
        if result.increment is not None:
            session.reconcile.filled(trader, update.ts_recv_ns)
            reason = exit_reason(trader, result.kernel_order, intent.purpose)
            words = None if reason is None else exit_words(trader, result.kernel_order, reason)
            self._fill_alert(intent, update, result.increment, words, flags)
        if result.ended is OrderStatus.REJECTED:
            why = update.rejection_reason or "no reason given"
            session.rejected += 1
            session.alert(
                AlertKind.REJECT, f"{order_words(intent)} order {update.order_id} rejected: {why}"
            )
            session.exits.rejected(trader, result.kernel_order, intent, why)
        if result.outcome is FillOutcome.REFUSED or result.mismatch:
            self._mismatch(trader, update)
        return result

    def _closing_rejected(self, trader: Trader, kernel_order: int, update: OrderUpdate) -> None:
        """A REJECTED word of `trader`'s working order: a closing order on a shared
        instrument is held (`_margin_reject`), the word's fills given first."""
        intent, margin = trader.venue.intent_of(kernel_order), self._session.margin
        if not margin.takes(trader, intent) or kernel_order not in trader.working():
            return
        if update.filled_quantity > trader.venue.filled(kernel_order):
            self.apply_update(trader, dataclasses.replace(update, status=OrderStatus.OPEN))
        margin.reject(trader, kernel_order, intent, update.rejection_reason or "no reason given")

    def _fill_alert(
        self,
        intent: OrderIntent,
        update: OrderUpdate,
        increment: Increment,
        words: str | None,
        flags: tuple[str, ...],
    ) -> None:
        """The fill's record, its event and its ENTRY or EXIT alert, which names in `words`
        the exit reason of an order that closes a position (`_exit_reasons`), the fill's
        `flags` when it has any, and in a session of several strategies, first, the
        strategy whose order filled."""
        kind = AlertKind.ENTRY if intent.purpose in ENTRY_PURPOSES else AlertKind.EXIT
        why = intent.purpose.value
        if words is not None:
            why = f"{why}: {words}"
        if flags:
            why = f"{why}; flagged {', '.join(flags)}"
        session = self._session
        # The alert's prefix lists every strategy of the session, so with several the line
        # says whose order it was (2.4.0 item P5-1); a session of one keeps its old line.
        named = f"{intent.strategy} " if len(session.traders) > 1 else ""
        estimated = session.stops.held.estimated(update.order_id, update.filled_quantity)
        record = fill_record(intent, update, increment, estimated=estimated, flags=flags)
        session.fills.append(record)
        session.event(
            "fill",
            strategy=intent.strategy,
            order_id=update.order_id,
            units=increment.units,
            price=increment.price,
            source=update.source,
            derived_from_average=increment.derived_from_average,
            estimated=record.estimated,
            flags=list(flags),
        )
        self._session.alert(
            kind,
            f"{named}{intent.action} {increment.units} {intent.symbol} {intent.exchange} at "
            f"{increment.price:.2f} ({why}, order {update.order_id})",
        )

    def _mismatch(self, trader: Trader, update: OrderUpdate) -> None:
        session = self._session
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        session.alert(
            AlertKind.RECONCILE_MISMATCH,
            f"{trader.name}: the engine and OpenAlgo differ on order {update.order_id} "
            f"({update.symbol} {update.exchange}, {update.filled_quantity} filled): entries "
            "of the strategy are locked for the rest of the session and nothing is sent to "
            "fix it. Check the position and the order in OpenAlgo",
        )


__all__ = ["RECONCILE_LOCK", "FillNews"]
