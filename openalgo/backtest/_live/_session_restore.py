"""A restart restored (16-session-design 3.1 steps 11 and 12, 3.6, 5.5, 6.2; unit 12): each
strategy's kernel session given its saved state once the plan's catch-up is done, the
venue map rebuilt from the kernel's (saved id, new id) pairs, every restored order checked
against the journal and the start's books, the day-long locks set again, the start's
differences acted on, and then, before the banner, the protective start phase. A strategy
with nothing saved today that carries a position or a GTT from an earlier day is given
its day-close record instead (`_carry_restore`, 20-positional-live 4.2), and its start's
difference is acted on as any (`_restart_carried` judged it).

**The restored-order check** (`_check`, 3.6), for each order the saved state worked:

- Saved as not sent (a strategy exit that waited for its cancels): nothing; the kernel
  places it when the cancels end.
- Bound in the journal: bound in the venue map and taken back by the stops supervisor
  (confirmed when the order book shows it working); its journal's last word and the
  trade book's fills applied as news, with its ending when the journal or its row has
  one; a modify or cancel that waited for its result settled by reading the order
  (`Settle`).
- Queued (semi-auto): `apply_reject`, with an ERROR alert.
- Unbound: cannot be, the start resolved it (step 9).
- Marked not placed, or never recorded (the crash came between the save and the send):
  an entry or exit `apply_reject` ("not sent before the restart"; an exit is asked for
  again by the retry schedule), a leg stop `apply_cancel`, which places it again (R3).

Then: the saved entry locks that last the day (reconciliation, exit failed, the
account's daily loss, a stop GTT refused at placement: `_session_saved.DAY_LOCKS`) set
again, since the kernel restores none (K4); a leg the start judged different from the
broker locked from the start and, when other orders' trades explain the difference,
adopted; the exit guard opened for a leg whose broker holds fewer units (a leg whose
restored engine holds other units than the start judged waits for a fresh read instead:
`Reconciler.differs_at_start`); a CNC or NRML leg's GTTs taken back by the journal
(`_session_restore_gtts`, for a saved state of today: a carry's are taken back by
`_carry_restore`, and a GTT its kernel places since, as a stop again, goes out once with
the loop's commands); each leg with a position given its stop again (`protect`, R7:
nothing when one works), or, for an MIS leg past its square-off, exited
(`request_exit("square_off")`: a CNC or NRML leg is never squared off, it is carried at
the close, 20-positional-live 7.3); the reconciler told what was adopted and which orders
are the session's own; every restored order polled once (`StreamGap`). The kernel's
events not delivered before the crash are lost (K4): the strategy's `on_fill`, `on_order`
and `on_position` never run for them.

**The protective start phase** (`protect_restored`, 3.1 step 12): the loop turns with no
feed and no strategy called until every resumed leg is flat, has no stop rule, or its
confirmed stop covers it, for at most `RESTART_PROTECT_WAIT_S`, before the banner and the
countdown.

Times are UTC nanoseconds, quantities signed units. Run by the main thread at the start.
"""

from __future__ import annotations

import logging
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._carry_restore import report_settled, restore_carried
from ._fill_ledger import VENUE_REJECT_CODE
from ._plan_release import CANDLE_LAST_READ_S
from ._reconcile_rules import Kind, LegBooks
from ._restart_books import LegStart, Restart
from ._session_records import restored
from ._session_restore_gtts import restore_gtts
from ._session_saved import Saved
from ._session_trader import Trader
from ._stops_gtt import overnight
from ._stops_kernel import protect
from ._stops_report import stop_words
from ._types import (
    LOGGER_NAME,
    AlertKind,
    OrderIntent,
    OrderStatus,
    OrderUpdate,
    Purpose,
    Sleep,
)
from ._venue_requests import PlaceRequest, place_priority
from ._venue_saved import SavedOrder
from ._worker_queue import Settle, StreamGap
from .inbox import NotSent, NotSentReason

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the protective start phase waits for the resumed legs' stops (3.1 step 12): the
#: unprotected deadline of a stop never confirmed (`_stops_deadline`).
RESTART_PROTECT_WAIT_S = 10.0
#: Seconds a restart waits for the catch-up to reach the slice a state was saved at: a
#: slice's last candle read (`_plan_release.CANDLE_LAST_READ_S`) and a read's margin; and
#: seconds between two catch-ups meanwhile.
RESTORE_CATCH_UP_WAIT_S = CANDLE_LAST_READ_S + 3.0
RESTORE_CATCH_UP_EVERY_S = 1.0
#: The kernel's reason for a restart's exit past the square-off (3.4).
SQUARE_OFF = "square_off"
#: Why a restored order that never reached the broker ends.
NOT_SENT = "not sent before the restart"


def restore(session: Any, restart: Restart, sleep: Sleep) -> None:
    """Step 11's restoration (module docstring) of every strategy of `session` (whose
    traders exist and whose catch-up is done); `sleep` waits seconds while the catch-up
    reaches the slice a state was saved at. Raises LiveError for a saved state the kernel
    refuses, or an intent the start left unbound."""
    adopted = {
        name: leg.saved.adopted for name, leg in restart.legs.items() if leg.saved is not None
    }
    session.reconcile.start(adopted, restart.own)
    carry = session.parts.carry
    for trader in session.traders:
        leg = restart.legs[trader.name]
        carried = None if carry is None else carry.of(trader.name)
        if leg.saved is not None:
            _catch_up_to(session, trader, int(leg.saved.live_state["slice"]), sleep)
            _restore(session, trader, leg.saved, restart)
        elif carried is not None:
            restore_carried(session, trader, carried)
        elif carry is not None and trader.name in carry.expired:
            report_settled(session, carry.expired[trader.name])
        _after(session, trader, leg, restart)
    if adopted:
        # Every order taken back is polled once: the stream may have missed its words.
        session.hand_over([StreamGap(restart.read_ns)])


def _catch_up_to(session: Any, trader: Trader, saved_slice: int, sleep: Sleep) -> None:
    """Catch the plan up until `trader`'s kernel session stands at the slice its state was
    saved at (a restart soon after the save may find the slice's candle not settled yet),
    for at most `RESTORE_CATCH_UP_WAIT_S`; the restore refuses a session still behind."""
    parts = session.parts
    until_ns = session.now() + round(RESTORE_CATCH_UP_WAIT_S * NANOS_PER_SECOND)
    while call(trader.engine.live_state)["slice"] < saved_slice and session.now() < until_ns:
        sleep(RESTORE_CATCH_UP_EVERY_S)
        parts.plan.catch_up(parts.clients.data, session.now)


def _restore(session: Any, trader: Trader, saved: Saved, restart: Restart) -> None:
    try:
        pairs = call(trader.engine.restore_live_state, saved.live_state)
    except Exception as error:
        raise LiveError(
            f"{trader.name}: the engine refused its saved state ({error}): the restart "
            "cannot resume it. Check its positions in OpenAlgo"
        ) from error
    pairs = {int(old): int(new) for old, new in pairs.items()}
    trader.records.restored = restored(trader)
    session.event("restored", strategy=trader.name, orders=len(pairs))
    logger.info(
        "%s restored: the events its kernel had not delivered before the restart are lost",
        trader.name,
    )
    rebuilt = trader.venue.restore(saved.orders, pairs)
    for old, new in sorted(pairs.items(), key=lambda pair: pair[1]):
        if saved.sent.get(old, False):
            record = rebuilt.get(new)
            if record is None:
                _not_placed(session, trader, new, None, leg_stop=saved.leg_stop(old))
            else:
                _check(session, trader, new, record, restart)
    for reason in sorted(saved.locks):
        if trader.lock(reason):
            session.event("lock", strategy=trader.name, reason=reason)


def _check(
    session: Any, trader: Trader, kernel_order: int, saved: SavedOrder, restart: Restart
) -> None:
    """The restored-order check of one order the venue map has a record of (module
    docstring)."""
    journal, day = session.parts.journal, session.parts.spec.day
    intent = saved.intent
    order_id = saved.order_id or journal.order_id_of(intent.intent_id) or ""
    if order_id:
        _bound(session, trader, kernel_order, saved, order_id, restart.legs[trader.name].books)
    elif any(queued.intent.intent_id == intent.intent_id for queued in journal.queued(day=day)):
        trader.venue.not_sent(NotSent(_place(kernel_order, intent), NotSentReason.REFUSED))
        session.alert(
            AlertKind.ERROR,
            f"{intent.strategy}: intent {intent.intent_id} waits in OpenAlgo's Action Center "
            "(semi-auto) since before the restart: reject it there; the engine takes it as "
            "rejected",
        )
    elif any(unbound.intent_id == intent.intent_id for unbound in journal.unbound(day=day)):
        raise LiveError(f"intent {intent.intent_id} is still unbound after the start's wait")
    else:
        _not_placed(session, trader, kernel_order, saved, leg_stop=False)


def _bound(
    session: Any,
    trader: Trader,
    kernel_order: int,
    saved: SavedOrder,
    order_id: str,
    books: LegBooks,
) -> None:
    """A restored order the journal binds: taken back, its news since the save applied,
    and its waiting modify or cancel settled by a read."""
    venue, now_ns = trader.venue, session.now()
    venue.bind_restored(kernel_order, order_id)
    row = books.row(order_id)
    working = row is not None and row.working
    session.stops.leg(trader).restored(kernel_order, saved.intent, order_id, working, now_ns)
    word = _word(session, saved, order_id, books)
    if word is not None:
        session.news.apply_update(trader, word)
    if kernel_order not in trader.working():
        return
    changes = venue.restored_changes(kernel_order, saved)
    session.stops.handed(changes)
    session.hand_over([Settle(change) for change in changes])


def _word(session: Any, saved: SavedOrder, order_id: str, books: LegBooks) -> OrderUpdate | None:
    """One cumulative word of a restored order from the journal's last update, the trade
    book's fills and its order-book row, or None when they tell nothing new: its fills,
    and its ending when the journal or its row has one (a COMPLETE only once the trade
    book has every unit; else the polls tell)."""
    journal, day = session.parts.journal, session.parts.spec.day
    last = journal.last_update(order_id, day=day)
    traded, average = books.fills(order_id)
    filled = max(traded, 0 if last is None else last.filled_quantity)
    if last is not None and last.filled_quantity >= traded:
        average = last.average_price
    row, quantity = books.row(order_id), saved.terms.quantity
    status = OrderStatus.OPEN
    if last is not None and last.status.terminal:
        status = last.status
    elif row is not None and row.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
        status = row.status
    elif filled >= quantity:
        status = OrderStatus.COMPLETE
    if filled <= saved.filled_units and not status.terminal:
        return None
    return OrderUpdate(
        order_id=order_id,
        symbol=saved.intent.symbol,
        exchange=saved.intent.exchange,
        action=saved.intent.action,
        quantity=quantity,
        filled_quantity=filled,
        pending_quantity=max(0, quantity - filled),
        average_price=average,
        status=status,
        rejection_reason=(last.rejection_reason if last is not None else "") or "",
        mode=session.parts.mode,
        ts_recv_ns=session.now(),
        source="reconcile",
    )


def _not_placed(
    session: Any, trader: Trader, kernel_order: int, saved: SavedOrder | None, *, leg_stop: bool
) -> None:
    """A restored order that never reached the broker ends: a leg stop is placed again
    (R3), any other order rejected; a rejected exit is asked for again on the schedule.
    With no venue record (`saved` None) the saved state's role tells a leg stop
    (`leg_stop`): the kernel's order dict has none."""
    if saved is None:
        if leg_stop:
            call(trader.engine.apply_cancel, kernel_order)
        else:
            call(trader.engine.apply_reject, kernel_order, VENUE_REJECT_CODE)
        session.event("not_sent_before_restart", strategy=trader.name, kernel_order=kernel_order)
        return
    request = _place(kernel_order, saved.intent)
    trader.venue.not_sent(NotSent(request, NotSentReason.REFUSED, NOT_SENT))
    session.event("not_sent_before_restart", strategy=trader.name, kernel_order=kernel_order)
    if saved.intent.purpose not in (Purpose.STOP, Purpose.ENTRY, Purpose.ADD):
        session.exits.rejected(trader, kernel_order, saved.intent, NOT_SENT)


def _place(kernel_order: int, intent: OrderIntent) -> PlaceRequest:
    """A placement as the venue map's answers take it: request 0, never handed over."""
    return PlaceRequest(0, kernel_order, intent, place_priority(intent))


def _after(session: Any, trader: Trader, leg: LegStart, restart: Restart) -> None:
    """The start's difference acted on, then the leg's stop or, for an MIS leg past its
    square-off, its square-off (a CNC or NRML leg is carried, never squared off)."""
    if leg.kind not in (Kind.AGREE, Kind.UNREADABLE):
        reconcile = session.reconcile
        reconcile.differs_at_start(trader, leg.kind, leg.journal_units, leg.books, leg.group)
    if leg.saved is not None:
        # A carry's GTTs were all taken back by `_carry_restore`; any other GTT of its
        # kernel is one the kernel made since, whose place the loop sends (`commands`).
        restore_gtts(session, trader, leg.saved.bindings)
    units = trader.open_units()
    if units == 0:
        return
    if session.now() >= trader.square_off_ns and not overnight(trader):
        outcome = trader.request_exit(SQUARE_OFF)
        session.event("restart_square_off", strategy=trader.name, outcome=outcome)
    elif not session.reconcile.holds(trader):
        outcome = protect(trader)
        session.event("protect", strategy=trader.name, outcome=outcome, restart=True)


def protect_restored(session: Any, wait_s: float) -> None:
    """The protective start phase (module docstring): one loop step at least, so the
    commands of the restoration reach the order worker."""
    until_ns = session.now() + round(wait_s * NANOS_PER_SECOND)
    session.step(until_ns, protecting=True)
    while session.now() < until_ns and not all(_covered(session, t) for t in session.traders):
        session.step(until_ns, protecting=True)


def _covered(session: Any, trader: Trader) -> bool:
    """Whether the leg needs nothing more of the start: flat, held by the exit guard (no
    stop beyond the broker's units), with no stop rule, or covered by its confirmed stop."""
    units = round(abs(trader.open_units()))
    stops = session.stops
    if units == 0 or session.reconcile.holds(trader) or not stops.leg(trader).guarded:
        return True
    return stops.covered(trader) >= units


def restart_lines(session: Any) -> list[str]:
    """The banner's lines of a restart (3.1 step 13): the start's number, and each resumed
    leg with its stop."""
    starts = session.parts.journal_session.starts
    if starts <= 1:
        return []
    lines = [f"Restarted: start {starts} of the day"]
    for trader in session.traders:
        units = trader.open_units()
        if units != 0:
            lines.append(
                f"{trader.name}: resumed {units:g} {trader.symbol} {trader.exchange}, "
                f"{stop_words(session.stops, trader)}"
            )
    return lines


__all__ = [
    "NOT_SENT",
    "RESTART_PROTECT_WAIT_S",
    "SQUARE_OFF",
    "protect_restored",
    "restart_lines",
    "restore",
]
