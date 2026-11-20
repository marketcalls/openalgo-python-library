"""What the reconciler (`reconcile`) does about a mismatch (16-session-design 5.3): the
safe corrections a finding seen twice in a row allows. What a comparison judges a leg by
(the in-flight rule, the session's own orders, the steps on the trade and order books) is
`_reconcile_steps`'.

**The safe corrections** (`Corrections.act`, decided 2026-11-16): the strategy's entries
locked for the rest of the session ("reconciliation") and a RECONCILE_MISMATCH alert
naming the leg, both figures and the orders, then only what can never open or add
exposure: an explained difference adopted (`apply_external_fill` at the other orders'
price, on a shared instrument each member's share of it, `_reconcile_allocate`; the
kernel fitting the leg's stop to the adopted units in the same call, and a
cancel of it the exit guard held sent when the stop is larger than what is left, or,
after the session's last lines with units left, withdrawn first so that the fit cuts the
stop instead: 16-session-design 3.3 step 5 "As built", round 7; held again when the kernel
refuses the adoption, which the alert then says, round 8), the leg's own stop
cancelled when the broker holds the other side, the session's own orphan cancelled.
Never an exit, never the kill switch. A finding on a leg in flight past the in-flight
limit is alerted, with nothing done; so is a restart's difference whose journal units
the restored engine no longer holds (`stale`), with entries locked.

Quantities are signed units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._engine import call
from ..errors import VenueRefusalError
from ._group_rules import GroupFigure, and_join
from ._plan_catch_up import INSTRUMENT
from ._reconcile_allocate import allocation_words
from ._reconcile_rules import FEWER_STOP_CANCELLED, RULES, Kind, LegBooks, adoption_price
from ._reconcile_steps import Finding
from ._session_news import RECONCILE_LOCK
from ._session_records import adopted_fill
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import LIVE_PRODUCT, CancelRequest
from ._worker_queue import CancelOrphan

logger = logging.getLogger(LOGGER_NAME)

#: The exit reason the kernel records for units adopted from other orders' trades.
ADOPTED_REASON = "manual"
#: What an adoption did when it also sent the cancel of the leg's stop that the exit guard
#: held (`ExitGuard.adopted`): the stop is cancelled rather than cut down.
ADOPTED_STOP_CANCELLED = "adopted, its held stop cancel sent"
#: What an adoption the kernel refused did, and what its RECONCILE_MISMATCH alert says:
#: nothing was adopted, and the stop stays as it was (a cancel withdrawn is held again).
NOT_ADOPTED = "not adopted"
NOT_ADOPTED_WORDS = (
    "trades of other orders explain it, but the engine could not take them as its own: "
    "nothing is adopted or sent for it, and its stop stays as it was. Check the position in "
    "OpenAlgo"
)


class Corrections:
    """The safe corrections of one session (module docstring); `session` is the
    `_session_loop.Session`, `adopted` the reconciler's signed units adopted by
    strategy, `orphans` the order IDs it asked to cancel, `figures` each member of a
    shared instrument's group as the comparison judged it (the words name it; none when
    no strategy shares its instrument)."""

    def __init__(
        self,
        session: Any,
        adopted: dict[str, int],
        orphans: set[str],
        figures: dict[str, GroupFigure] | None = None,
    ) -> None:
        self._session = session
        self._adopted = adopted
        self._orphans = orphans
        self.figures: dict[str, GroupFigure] = {} if figures is None else figures

    def act(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        """Act on a finding seen twice in a row; what was done."""
        session, kind = self._session, finding[0]
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        action = "locked"
        if kind == "orphan":
            self._orphans.add(finding[1])
            session.hand_over([CancelOrphan(finding[1])])
            action = "orphan cancelled"
            words = f"order {finding[1]} works at the broker, the engine does not hold it: "
            words += "it is cancelled"
        elif kind == "foreign":
            words = f"order {finding[1]} names the strategy and the session did not send it: "
            words += "nothing is sent for it. Check it in OpenAlgo"
        elif kind == "stop_terms":
            words = f"its stop {finding[1]} shows {finding[2]} units at {finding[3]:.2f} at the "
            words += "broker, not the engine's terms. Check it in OpenAlgo"
        elif kind is Kind.EXPLAINED_SHARED:
            action, words = self._share_adopted(trader, books, own)
        else:
            rule = RULES[kind]
            words = rule.words
            if rule.adopt:
                action = self._adopt(trader, finding[1], finding[2], books, own)
            if action == ADOPTED_STOP_CANCELLED and kind is Kind.EXPLAINED_FEWER:
                words = FEWER_STOP_CANCELLED
            if action.startswith(NOT_ADOPTED):
                words = NOT_ADOPTED_WORDS
            if rule.cancel_stop:
                action = self._cancel_stop(trader)
        session.alert(AlertKind.RECONCILE_MISMATCH, self._words(trader, finding, books, own, words))
        if kind is Kind.OTHER_SIDE:
            session.alert(
                AlertKind.ERROR,
                f"{trader.name}: OpenAlgo's books show {trader.symbol} {trader.exchange} on "
                f"the other side of the engine's position: close it by hand in OpenAlgo",
            )
        return action

    def alert_only(
        self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]
    ) -> str:
        """A finding on a leg in flight past the in-flight limit: alerted, nothing done."""
        words = (
            "an order of the leg has been in flight too long, so nothing is done. Check the "
            "account in OpenAlgo"
        )
        text = self._words(trader, finding, books, own, words, locked=False)
        self._session.alert(AlertKind.ERROR, text)
        return "alerted"

    def stale(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        """A restart's difference judged on the journal's units when the restored engine
        holds other units: entries locked and alerted, nothing adopted or cancelled, since
        the books read at the start no longer describe the leg."""
        session = self._session
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        words = (
            f"the engine holds {round(trader.open_units())} units once restored, since a fill "
            "reached it after these books were read, so nothing is done until a fresh read "
            "of the books judges the leg"
        )
        session.alert(AlertKind.RECONCILE_MISMATCH, self._words(trader, finding, books, own, words))
        return "locked"

    def _share_adopted(
        self, trader: Trader, books: LegBooks, own: frozenset[str]
    ) -> tuple[str, str]:
        """A member's share of a difference other orders' trades explain on its shared
        instrument (`_reconcile_allocate`), adopted as a leg's is: what was done, and the
        alert's words naming the allocation."""
        figure = self.figures.get(trader.name, GroupFigure((trader.name,), 0))
        share, words = figure.share(trader.name), allocation_words(figure, trader.name)
        if share is None or not share.taken:
            return "locked", words
        action = self._adopt(trader, share.units, share.left, books, own, share.price)
        return action, (NOT_ADOPTED_WORDS if action.startswith(NOT_ADOPTED) else words)

    def _adopt(
        self,
        trader: Trader,
        engine: int,
        broker: int,
        books: LegBooks,
        own: frozenset[str],
        price: float | None = None,
    ) -> str:
        """Adopt the difference of the leg's `engine` units from `broker` units (signed) at
        `price`, by default the other orders' newest trades' (module docstring)."""
        units, side = abs(engine - broker), "SELL" if broker < engine else "BUY"
        session, guard = self._session, self._session.reconcile.guard
        withdrawn: list[CancelRequest] = []
        try:
            if price is None:
                price = adoption_price(books.external(own), units, side)
            # After the last lines a held cancel of the stop is withdrawn before the fill,
            # so the kernel's fit with it cuts the stop to what is left (module docstring).
            withdrawn = guard.withdraw_for_cut(trader.name, broker)
            before = trader.open_units()
            call(
                trader.engine.apply_external_fill,
                INSTRUMENT,
                LIVE_PRODUCT,
                side,
                float(units),
                price,
                session.now(),
                ADOPTED_REASON,
            )
        except (ValueError, VenueRefusalError) as error:
            # No fit comes: a cancel withdrawn for it is held again, as it was.
            guard.put_back(trader.name, withdrawn)
            logger.error("%s: the broker's units not adopted: %s", trader.name, error)
            return f"{NOT_ADOPTED} ({error})"
        signed = units if side == "BUY" else -units
        self._adopted[trader.name] = self._adopted.get(trader.name, 0) + signed
        trader.records.filled(before, trader.open_units())
        session.fills.append(adopted_fill(trader, side, units, price, session.now()))
        session.event("adopted", strategy=trader.name, side=side, units=units, price=price)
        # An exit's cancel of the stop the exit guard held: the kernel waits for it and
        # fits nothing more, so the stop would stay larger than what is left.
        if guard.adopted(trader.name):
            return ADOPTED_STOP_CANCELLED
        return "adopted"

    def _cancel_stop(self, trader: Trader) -> str:
        stop = self._session.stops.leg(trader).stop
        if stop is None or stop not in trader.working():
            return "no stop to cancel"
        # The exit guard holds the stop's cancel of a leg whose broker holds fewer units:
        # this one it lets go (a stop on the other side's position could add to it).
        self._session.reconcile.guard.allow_cancel(trader.name, stop)
        trader.cancel(stop)
        return "stop cancelled"

    def _words(
        self,
        trader: Trader,
        finding: Finding,
        books: LegBooks,
        own: frozenset[str],
        what: str,
        *,
        locked: bool = True,
    ) -> str:
        """The alert's words: the leg, the engine's units (as compared), the broker's, the
        other orders on the instrument, what is done, and the lock. A member of a shared
        instrument's words name its group's sum, which the broker's net was judged
        against (18-multi-strategy 4.6)."""
        units = finding[1] if isinstance(finding[0], Kind) else round(trader.open_units())
        others = sorted({trade.order_id for trade in books.external(own)})
        orders = f"; other orders {', '.join(others)}" if others else ""
        lock = " Entries of the strategy are locked for the rest of the session" if locked else ""
        held, shown = f"the engine holds {units} units", f"OpenAlgo's books show {books.units}"
        group = self.figures.get(trader.name)
        if group is not None:
            held += f", {group.units} with {and_join(group.others(trader.name))}'s"
            shown += " for them together"
        leg = f"{trader.name} {trader.symbol} {trader.exchange}"
        return f"{leg}: {held}, {shown}{orders}: {what}.{lock}"


__all__ = [
    "ADOPTED_REASON",
    "ADOPTED_STOP_CANCELLED",
    "NOT_ADOPTED",
    "NOT_ADOPTED_WORDS",
    "Corrections",
]
