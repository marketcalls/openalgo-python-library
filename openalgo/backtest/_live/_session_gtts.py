"""What the session loop does with the order worker's GTT news (20-positional-live 3; the
lead's ruling N3b-2): each answer and finding applied to its strategy's kernel through its
GTT map (`_venue_gtts`), then what the session must do around it.

- **Placed, modified, cancelled** and their refusals: the kernel call alone; the modifies
  and cancels that waited for a place's answer go to the order worker. A trailing modify
  confirmed is a STOP_MOVED alert, as an order's.
- **Refused at placement** (3.5, decision 11.7): for a leg's stop GTT the kernel exits
  the leg at market and locks the strategy's entries for the day (`GTT_REFUSED_LOCK`,
  copied to the order worker's record); an ERROR alert quotes the broker's message, and
  says when the refusal is kept across starts. A strategy's GTT order ends rejected, with
  a REJECT alert quoting the message: nothing is exited or locked.
- **Unresolved, a mismatch, other terms**: no kernel call; the strategy's entries are
  locked (`GTT_UNRESOLVED_LOCK`, for the session: the user resolves it by hand and starts
  again) with an ERROR alert naming the GTT and why, never an exit (16-session-design 5.3).
  A result call the kernel refuses is the same.
- **Fired**: the child the follower found is bound in the venue map and, through the order
  worker, in the journal (`BindGttChild`), so its fills and its end come as an order's;
  with its child missed, the fill the position's move stands for is given at once through
  the venue map's fill path, at the trade book's price or the child's limit, with the
  finding's flags (`price_unverified`, and `missed_day` for a GTT that fired on a day no
  session ran: 20-positional-live 3.3, 4.3), and a WARN alert when unverified naming the
  GTT, the price used and that the contract note holds the true one.
- **Ended outside the session**, expired or rejected after placement: the kernel places
  the stop again (or exits flagged `stop_crossed`), the session asks `protect` for the leg
  too (it answers that the stop works when the kernel placed it), and a REJECT alert
  says what ended. A strategy's GTT order that ended so ends in the kernel too, with a
  REJECT alert and no protect. A leg's stop GTT found gone with no child while the read
  that judged it shows the broker holding other units than the leg (one-sided for CNC,
  `_broker_units.strategy_units`) may have fired, its fall hidden by a reference that was
  not the leg's: nothing is applied, so no stop is placed for units the account may not
  hold; the strategy's entries are locked with an ERROR alert, as for an unresolved GTT,
  and the close names the GTT not confirmed (review round 1 of unit N5a).

Quantities are whole units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._broker_units import strategy_units
from ._gtt_types import MISSED_DAY, PRICE_UNVERIFIED, GttAction, GttAnswer, GttEvent, GttOutcome
from ._journal_gtt_rows import GttTerms
from ._session_carry import FINDING_WORDS
from ._session_trader import Trader
from ._stops_kernel import protect
from ._types import LOGGER_NAME, AlertKind, OrderStatus, OrderUpdate, Purpose
from ._venue_children import missed_child_id
from ._venue_gtts import GttApplied, GttHandled

logger = logging.getLogger(LOGGER_NAME)

#: The kernel's entry lock of a stop GTT refused at placement (the binding's
#: `GTT_REFUSED_LOCK`, rust/oa_bt/src/engine: the day's).
GTT_REFUSED_LOCK = "gtt_refused"
#: The session's entry lock of a GTT whose fate it cannot tell (module docstring).
GTT_UNRESOLVED_LOCK = "gtt_unresolved"
#: Where the fill of a missed child came from, as its word names it.
MISSED_SOURCE = "gtt_missed"


class GttNewsHandler:
    """The loop's handling of GTT news for `session`, a `_session_loop.Session` (module
    docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        # The (journal GTT, finding) of each GTT named as no strategy's of the session.
        self._strangers: set[tuple[int, GttEvent]] = set()

    def take(self, news: GttAnswer | GttOutcome) -> None:
        """Apply one GTT answer or finding (module docstring)."""
        strategy = news.request.strategy if isinstance(news, GttAnswer) else news.strategy
        try:
            trader = self._session.trader_named(strategy)
        except LiveError:
            logger.error("GTT news of %s, which runs in no strategy of the session", strategy)
            if isinstance(news, GttOutcome):
                self._stranger(news, f"and {strategy} does not run in this session")
            return
        if trader.gtts is not None and isinstance(news, GttOutcome):
            news = trader.gtts.local(news)
            known = trader.gtts.gtt_found(news) is not None
            if not trader.finished and not known:
                self._stranger(news, "and the session neither placed it nor continued it")
        if trader.gtts is None or trader.finished:
            logger.warning("%s: GTT news after its kernel session ended: %r", strategy, news)
            if trader.gtts is not None and isinstance(news, GttOutcome):
                # Found once the kernel's session closed: the leg's stop is not confirmed.
                self._session.news.carry.after_close(trader, news)
            return
        if isinstance(news, GttAnswer):
            handled = trader.gtts.answered(news)
            self._answered(trader, news, handled)
        elif self._unheld(trader, news):
            return
        else:
            handled = trader.gtts.found(news)
            self._found(trader, news, handled)
        if handled.refused:
            self._unresolved(trader, handled.refused)
        self._session.hand_over(list(handled.requests))

    def _answered(self, trader: Trader, answer: GttAnswer, handled: GttHandled) -> None:
        request = answer.request
        what = _named(answer.trigger_id or "", request.purpose)
        if handled.applied is GttApplied.UNRESOLVED:
            self._unresolved(trader, f"{what} {request.action.value}: {answer.message}")
        elif handled.applied is GttApplied.REFUSED and request.purpose is not Purpose.STOP:
            # A strategy's GTT order ends rejected: nothing was exited, nothing locked.
            self._session.alert(
                AlertKind.REJECT,
                f"{trader.name}: the {what} of {trader.symbol} {trader.exchange} was not "
                f'placed: "{answer.message}". The order is rejected',
            )
        elif handled.applied is GttApplied.REFUSED:
            trader.kernel_locked(GTT_REFUSED_LOCK)
            kept = (
                ". It is kept: the strategy is refused at every later start until its "
                'product or positional_stops="rearm_daily" is changed'
                if answer.kept
                else ""
            )
            self._session.alert(
                AlertKind.ERROR,
                f"{trader.name}: the broker refused the {what} of {trader.symbol} "
                f'{trader.exchange}: "{answer.message}". The position is exited at market '
                f"and entries are locked for the day{kept}",
            )
        elif handled.applied is GttApplied.MODIFIED and request.action is GttAction.MODIFY:
            self._session.alert(
                AlertKind.STOP_MOVED,
                f"{trader.name}: {what} of {trader.symbol} {trader.exchange} now triggers "
                f"at {request.trigger_price:.2f} for {request.quantity} units",
            )

    def _found(self, trader: Trader, outcome: GttOutcome, handled: GttHandled) -> None:
        applied = handled.applied
        known = trader.gtts.gtt_found(outcome) if trader.gtts is not None else None
        purpose = Purpose.STOP if known is None else known.purpose
        what = _named(outcome.trigger_id, purpose)
        if applied is GttApplied.UNRESOLVED:
            self._unresolved(trader, f"{what} ({outcome.event.value}): {outcome.message}")
        elif applied is GttApplied.TRIGGERED and handled.child is not None:
            trader.venue.bind_child(handled.child, outcome.order_id, *self._child(trader, handled))
        elif applied is GttApplied.CHILD_MISSED and handled.child is not None:
            self._missed(trader, outcome, handled.child)
        elif (
            applied in (GttApplied.ENDED_OUTSIDE, GttApplied.ENDED) and purpose is not Purpose.STOP
        ):
            self._session.alert(
                AlertKind.REJECT,
                f"{trader.name}: {what} of {trader.symbol} {trader.exchange} ended at the "
                f"broker with no request from the session ({outcome.event.value}): the order "
                "ends there",
            )
        elif applied in (GttApplied.ENDED_OUTSIDE, GttApplied.ENDED):
            outcome_of = protect(trader)
            self._session.event("protect", strategy=trader.name, outcome=outcome_of)
            self._session.alert(
                AlertKind.REJECT,
                f"{trader.name}: {what} of {trader.symbol} {trader.exchange} ended at the "
                f"broker with no request from the session ({outcome.event.value}): the "
                "engine places it again, or exits if the price is past the stop",
            )

    def _unheld(self, trader: Trader, outcome: GttOutcome) -> bool:
        """Whether `outcome` is a leg's stop GTT ended outside the session while the broker
        held other units than the leg (module docstring): then it is not applied, and the
        strategy's entries are locked with an ERROR alert."""
        if outcome.event is not GttEvent.ENDED_OUTSIDE or outcome.held is None:
            return False
        known = trader.gtts.gtt_found(outcome) if trader.gtts is not None else None
        if known is None or known.unsettled or known.purpose is not Purpose.STOP:
            return False
        units = round(trader.open_units())
        if strategy_units(trader.product, units, outcome.held) == units:
            return False
        known.finding = outcome.event
        self._unresolved(
            trader,
            f"{_named(outcome.trigger_id, Purpose.STOP)} ended at the broker with no child "
            f"found while OpenAlgo's books show {outcome.held} of the leg's {units} units: it "
            "may have fired, so it is not placed again",
        )
        return True

    def _child(self, trader: Trader, handled: GttHandled) -> tuple[GttTerms, Purpose]:
        """The terms and the purpose of a fired GTT's child, as the kernel follows it."""
        order = call(trader.engine.order, handled.child)
        known = trader.gtts.gtt_of_child(handled.child) if trader.gtts is not None else None
        purpose = Purpose.STOP if known is None else known.purpose
        terms = GttTerms(
            str(order["side"]),
            float(order["limit_rupees"] or 0.0),
            float(order["limit_rupees"] or 0.0),
            round(float(order["quantity_units"])),
        )
        return terms, purpose

    def _missed(self, trader: Trader, outcome: GttOutcome, child: int) -> None:
        """Bind a missed child to the session's own ID and give its one fill (module
        docstring)."""
        session = self._session
        terms, purpose = self._child(trader, GttHandled(GttApplied.CHILD_MISSED, child=child))
        order_id = missed_child_id(outcome.trigger_id)
        trader.venue.bind_child(child, order_id, terms, purpose)
        update = OrderUpdate(
            order_id=order_id,
            symbol=outcome.symbol,
            exchange=outcome.exchange,
            action=terms.side,
            quantity=outcome.units,
            filled_quantity=outcome.units,
            pending_quantity=0,
            average_price=outcome.price,
            status=OrderStatus.COMPLETE,
            rejection_reason="",
            mode=session.parts.mode,
            ts_recv_ns=session.now(),
            source=MISSED_SOURCE,
        )
        session.news.apply_update(trader, update, outcome.flags)
        if PRICE_UNVERIFIED in outcome.flags:
            when = "on a trading day no session ran" if MISSED_DAY in outcome.flags else ""
            when = when or "and its child order was not found"
            done = "bought" if terms.side == "BUY" else "sold"
            session.alert(
                AlertKind.WARN,
                f"{trader.name}: {_named(outcome.trigger_id, purpose)} of {trader.symbol} "
                f"{trader.exchange} fired {when}: its {outcome.units} units are recorded "
                f"{done} at the child's limit, {outcome.price:.2f}, flagged "
                f"{', '.join(outcome.flags)}; the contract note holds the true price",
            )

    def _unresolved(self, trader: Trader, why: str) -> None:
        lock_unresolved(self._session, trader, why)

    def _stranger(self, outcome: GttOutcome, why: str) -> None:
        """A finding about a GTT of the journal no strategy of the session holds (a
        strategy that does not run in it, or a GTT it did not continue): an ERROR alert
        once per GTT and finding, never only a log line (the lead's ruling N5a-4); a GTT
        that rests is named by nothing."""
        key = (outcome.seq, outcome.event)
        if outcome.event is GttEvent.RESTING or key in self._strangers:
            return
        self._strangers.add(key)
        self._session.alert(
            AlertKind.ERROR,
            f"{outcome.strategy}: GTT {outcome.trigger_id} of {outcome.symbol} "
            f"{outcome.exchange} {FINDING_WORDS[outcome.event]}, {why}: no engine follows "
            "it, so check the GTT and order books in OpenAlgo",
        )


def lock_unresolved(session: Any, trader: Trader, why: str) -> None:
    """Lock `trader`'s entries for the session and alert why (module docstring "Unresolved");
    `session` is the `_session_loop.Session`."""
    trader.lock(GTT_UNRESOLVED_LOCK)
    session.event("gtt_unresolved", strategy=trader.name, why=why)
    session.alert(
        AlertKind.ERROR,
        f"{trader.name}: {why}. Its entries are locked and nothing is exited: check the "
        f"GTT and order books in OpenAlgo for {trader.symbol} {trader.exchange}, settle "
        "it by hand, then start the session again",
    )


def _named(trigger_id: str, purpose: Purpose) -> str:
    """A GTT in an alert's words."""
    kind = "stop GTT" if purpose is Purpose.STOP else "GTT order"
    return f"{kind} {trigger_id}" if trigger_id else kind


__all__ = [
    "GTT_REFUSED_LOCK",
    "GTT_UNRESOLVED_LOCK",
    "MISSED_SOURCE",
    "GttNewsHandler",
    "lock_unresolved",
]
