"""The GTT follower of the order worker (20-positional-live 3.1 to 3.3; the lead's ruling
N3a-2): every GTT of the journal the broker holds, judged at each read of the GTT book,
and the child order of one that fired found in the order book. Each finding is a typed
outcome posted to the loop (`_gtt_types.GttOutcome`, as `inbox.GttNews`), and the journal
notes each state.

**When it reads** (3.1): after every GTT call (the calls' confirmations, `_gtt_send`, whose
reads it judges too: `observe`); every `GTT_BOOK_EVERY_S` while the market is open and a
GTT it follows rests; at once when the order stream shows an order the journal does not
know on a symbol with a GTT (`order_seen`); when the session asks (the start's read and
the close's, `inbox`'s `ReadGtts`: `ask`). Nothing is read while OpenAlgo is silent
(`busy`): a read could hold the worker for the client's whole timeout. A read that fails
changes nothing.

**What a read says** (`_gtt_book.Seen`, the table of 3.1): resting, the GTT's leg's units
read just before it become its baseline (`_gtt_reads`); an unknown status is taken as
resting, with an ERROR alert once per GTT; ended without firing (cancelled, deleted,
disabled), expired or rejected after placement, the GTT ends in the journal and its
outcome is posted (the session places it again, 3.4); fired or gone, its leg is in flight
(IN_FLIGHT posted) while its child is looked for. A GTT with a call of the order worker
pending is the call's to judge (`hold`).

**The child** (3.2, 3.3; `_gtt_child`): the sandbox's book names it; otherwise the order
book is read every `CHILD_READ_EVERY_S` for `CHILD_SEARCH_S`: one candidate is the child
(TRIGGERED), several a MISMATCH naming them, never a guess. None by then: the leg's units
read now against its baseline say whether it fired with the child missed (CHILD_MISSED,
priced from the trade book or at the GTT's limit, flagged), ended outside the session
(ENDED_OUTSIDE), or cannot be told (MISMATCH). A GTT first seen gone with no baseline
(no read showed it resting with its position) cannot be told either. The journal keeps a
fired GTT TRIGGERED until the session binds its child (`bind_child`), so a restart before
that looks for the child again.

Times are UTC nanoseconds, prices rupees, quantities signed whole units. Used by the
worker's thread only.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._gtt_book import GttRow, Seen
from ._gtt_child import Move, child_candidates, missed_price, move_of
from ._gtt_reads import BookRead, GttReader, LegKey
from ._gtt_types import PRICE_UNVERIFIED, GttEvent, GttOutcome
from ._journal_gtt_rows import GttBinding, GttState
from ._types import LOGGER_NAME, AlertKind, AlertsLike, Now, OrderUpdate
from .inbox import GttNews, Message
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)

#: Seconds between two reads of the GTT book while the market is open and a GTT rests
#: (20-positional-live 3.1).
GTT_BOOK_EVERY_S = 10.0
#: Seconds between two order-book reads while a fired or gone GTT's child is looked for,
#: and how long it is looked for before the leg's units decide (3.2 rule 4).
CHILD_READ_EVERY_S = 2.0
CHILD_SEARCH_S = 60.0
#: How each ending the book shows is noted, and the outcome it is.
_ENDINGS = {
    Seen.ENDED: (GttEvent.ENDED_OUTSIDE, "ended outside the session: the book shows {status}"),
    Seen.EXPIRED: (GttEvent.EXPIRED, "ended by the broker's expiry"),
    Seen.REJECTED: (GttEvent.REJECTED, "rejected by the broker after its placement"),
}
_NS = NANOS_PER_SECOND


@dataclass
class _Followed:
    """A GTT followed: its journal binding, its leg's units before the last read that
    showed it resting and when that read began, and once it fired or is gone, since when
    and when its child is looked for next (the sandbox's child, when its row names one)."""

    gtt: GttBinding
    units_before: int | None = None
    active_ns: int | None = None
    since_ns: int | None = None
    next_ns: int = 0
    child: str | None = None
    warned: bool = False

    @property
    def leg(self) -> LegKey:
        return self.gtt.symbol, self.gtt.exchange, self.gtt.product


class GttFollower:
    """The follower (module docstring). `reader` makes the reads, `journal` is handle W,
    `post` puts a message on the loop's inbox, `alerts` the session's; `now` gives UTC
    nanoseconds; `market_open(now_ns)` says whether the market is open (None: never, so
    no read is made on the clock); `busy()` is True while OpenAlgo is silent."""

    def __init__(
        self,
        reader: GttReader,
        *,
        journal: Journal,
        post: Callable[[Message], None],
        alerts: AlertsLike,
        now: Now,
        market_open: Callable[[int], bool] | None,
        busy: Callable[[], bool],
    ) -> None:
        self._reader = reader
        self._journal = journal
        self._post = post
        self._alerts = alerts
        self._now = now
        self._market_open = market_open
        self._busy = busy
        self._followed: dict[int, _Followed] = {}
        self._held: set[int] = set()
        self._children: set[str] = set()
        self._asked = False
        self._next_read_ns = 0

    # ----- what the GTT calls and the worker tell it --------------------------------------

    def restore(self) -> None:
        """Follow every GTT of the journal the broker holds that has not ended, and read
        the book at the first step when there is any (20-positional-live 4.2 step 5). Its
        child is looked for from its last change in the journal (it was placed or modified
        then), and its baseline is the first read that shows it resting."""
        for gtt in self._journal.open_gtts():
            if gtt.trigger_id is not None:
                self._followed[gtt.seq] = _Followed(gtt, active_ns=self._last_change(gtt))
        self._asked = bool(self._followed)

    def _last_change(self, gtt: GttBinding) -> int:
        """When the journal last noted `gtt` placed, resting or being changed (UTC ns),
        before any note that it fired or went from the book: its child came after."""
        changes = [
            at
            for state, _, at in self._journal.gtt_states(gtt.seq)
            if state is not GttState.TRIGGERED
        ]
        return max(changes, default=gtt.recorded_ns)

    def follow(self, gtt: GttBinding, read: BookRead) -> None:
        """Follow `gtt`, which `read` showed resting (a place or modify confirmed): its
        leg's units in `read` are its baseline."""
        followed = _Followed(gtt, read.units.get((gtt.symbol, gtt.exchange, gtt.product)))
        followed.active_ns = read.read_ns
        self._followed[gtt.seq] = followed
        self._held.discard(gtt.seq)
        self._next_read_ns = max(self._next_read_ns, read.read_ns + round(GTT_BOOK_EVERY_S * _NS))

    def take_over(self, gtt: GttBinding, read: BookRead) -> None:
        """Judge `gtt` by `read` from now: a call's confirmation found it no longer
        resting (fired, gone or ended). Its child, if any, came after it was recorded."""
        self._followed.setdefault(gtt.seq, _Followed(gtt, active_ns=gtt.recorded_ns))
        self._followed[gtt.seq].gtt = gtt
        self._held.discard(gtt.seq)
        self.observe(read)

    def forget(self, seq: int) -> None:
        """Stop following GTT `seq` (its cancel was confirmed)."""
        self._followed.pop(seq, None)
        self._held.discard(seq)

    def hold(self, seq: int) -> None:
        """A call of GTT `seq` is pending: its confirmation judges it, not the follower."""
        self._held.add(seq)

    def release(self, seq: int) -> None:
        """GTT `seq`'s call was answered: the follower judges it again."""
        self._held.discard(seq)

    def last_active(self, seq: int) -> int | None:
        """When the last read that showed GTT `seq` resting began (UTC ns), or None."""
        followed = self._followed.get(seq)
        return None if followed is None else followed.active_ns

    def legs(self) -> set[LegKey]:
        """The legs of the GTTs followed, whose units a GTT book read reads first."""
        return {followed.leg for followed in self._followed.values()}

    def ask(self) -> None:
        """Read the GTT book at the next step (after a GTT call the broker refused)."""
        self._asked = True

    def read_now(self) -> None:
        """Read the GTT book now and judge it (the session's `ReadGtts`: the start's read
        and the close's)."""
        self._asked = True
        self.due()

    def order_seen(self, update: OrderUpdate) -> None:
        """The order stream showed an order the journal does not know: read at once when a
        GTT followed rests on its symbol (3.1)."""
        where = (update.symbol, update.exchange)
        if any(f.since_ns is None and f.leg[:2] == where for f in self._followed.values()):
            self._asked = True

    def bind_child(self, seq: int, order_id: str, kernel_order: int) -> None:
        """Bind the child the follower found for GTT `seq` as the kernel's order
        `kernel_order`, and note the GTT fired (`inbox`'s `BindGttChild`). Raises
        JournalError as `Journal.bind_gtt_child` does."""
        self._journal.bind_gtt_child(seq, order_id, kernel_order)
        self._journal.set_gtt_state(seq, GttState.FIRED, note=f"its child is order {order_id}")

    # ----- the reads --------------------------------------------------------------------

    def due(self) -> None:
        """Read what is due (module docstring) and judge it."""
        if self._busy() or not self._followed:
            return
        now = self._now()
        periodic = (
            self._market_open is not None
            and self._market_open(now)
            and now >= self._next_read_ns
            and any(f.since_ns is None for f in self._followed.values())
        )
        if self._asked or periodic:
            self._asked = False
            self._next_read_ns = now + round(GTT_BOOK_EVERY_S * _NS)
            self.observe(self._reader.gtt_book(self.legs()))
        searching = [f for f in self._followed.values() if f.since_ns is not None]
        if any(f.next_ns <= now for f in searching):
            rows = self._reader.order_rows()
            for followed in searching:
                if followed.next_ns <= now:
                    self._search(followed, rows, now)

    def observe(self, read: BookRead) -> None:
        """Judge every GTT followed and not held by `read` (module docstring); a read that
        failed changes nothing."""
        if read.rows is None:
            return
        for followed in list(self._followed.values()):
            seq = followed.gtt.seq
            if (
                seq in self._held
                or followed.since_ns is not None
                or followed.gtt.trigger_id is None
            ):
                continue
            row = read.rows.get(followed.gtt.trigger_id)
            seen = Seen.GONE if row is None else row.seen
            if seen.rests:
                self._resting(followed, row, read)
            elif seen in (Seen.TRIGGERED, Seen.GONE):
                self._fired_or_gone(followed, row, seen, read.read_ns)
            else:
                event, note = _ENDINGS[seen]
                self._end(followed, event, note.format(status=row.status if row else ""))

    def _resting(self, followed: _Followed, row: GttRow | None, read: BookRead) -> None:
        followed.active_ns = read.read_ns
        units = read.units.get(followed.leg)
        if units is not None:
            followed.units_before = units
        if followed.gtt.state is GttState.PLACED:
            self._journal.set_gtt_state(followed.gtt.seq, GttState.ACTIVE)
            followed.gtt = dataclasses.replace(followed.gtt, state=GttState.ACTIVE)
        if row is not None and row.seen is Seen.UNKNOWN and not followed.warned:
            followed.warned = True
            text = (
                f"GTT {row.trigger_id} of {followed.gtt.strategy} on {followed.gtt.exchange}:"
                f"{followed.gtt.symbol} shows status {row.status!r}, which the engine does not "
                "know: it is taken as resting. Check it in OpenAlgo"
            )
            logger.error(text)
            self._alerts.send(AlertKind.ERROR, text)

    def _fired_or_gone(
        self, followed: _Followed, row: GttRow | None, seen: Seen, at_ns: int
    ) -> None:
        note = "triggered" if seen is Seen.TRIGGERED else "gone from the GTT book"
        if followed.gtt.state is not GttState.TRIGGERED:
            self._journal.set_gtt_state(followed.gtt.seq, GttState.TRIGGERED, note=note)
            followed.gtt = dataclasses.replace(followed.gtt, state=GttState.TRIGGERED)
        followed.since_ns, followed.next_ns = at_ns, at_ns
        followed.child = None if row is None else row.child_order_id
        self._outcome(followed, GttEvent.IN_FLIGHT, message=note)
        if followed.child is not None:
            self._triggered(followed, followed.child)

    # ----- the child --------------------------------------------------------------------

    def _search(self, followed: _Followed, rows: list[Mapping[str, Any]] | None, now: int) -> None:
        followed.next_ns = now + round(CHILD_READ_EVERY_S * _NS)
        if rows is None:
            return
        found = child_candidates(followed.gtt, rows, since_ns=followed.active_ns, taken=self.taken)
        if found.child is not None:
            self._triggered(followed, found.child)
        elif found.every:
            words = f"orders {', '.join(found.every)} could each be its child"
            self._unresolved(followed, words, found.every)
        elif followed.since_ns is not None and now - followed.since_ns >= CHILD_SEARCH_S * _NS:
            self._settle_gone(followed)

    def _settle_gone(self, followed: _Followed) -> None:
        """No child was found in time: the leg's units decide (module docstring)."""
        units = self._reader.units({followed.leg}).get(followed.leg)
        if units is None:
            return
        before = followed.units_before
        if before is None:
            words = "no read showed it resting with its position, so whether it fired is unknown"
            self._unresolved(followed, words)
            return
        move = move_of(followed.gtt, before, units)
        if move is Move.FIRED:
            self._missed(followed)
        elif move is Move.UNCHANGED:
            note = "gone with no child found and the position unchanged: ended outside the session"
            self._end(followed, GttEvent.ENDED_OUTSIDE, note)
        else:
            self._unresolved(followed, f"the position went from {before} to {units} units")

    def _missed(self, followed: _Followed) -> None:
        gtt = followed.gtt
        price = missed_price(gtt, self._reader.trade_book(), self.taken)
        flags = () if price is not None else (PRICE_UNVERIFIED,)
        price = gtt.terms.limit_price if price is None else price
        note = f"fired, its child missed: {gtt.terms.quantity} units at {price:.2f}"
        self._journal.set_gtt_state(gtt.seq, GttState.FIRED, note=note)
        self._followed.pop(gtt.seq, None)
        units = gtt.terms.quantity
        self._outcome(followed, GttEvent.CHILD_MISSED, units=units, price=price, flags=flags)

    def _triggered(self, followed: _Followed, child: str) -> None:
        self._children.add(child)
        self._followed.pop(followed.gtt.seq, None)
        self._outcome(followed, GttEvent.TRIGGERED, order_id=child)

    def _unresolved(
        self, followed: _Followed, words: str, candidates: tuple[str, ...] = ()
    ) -> None:
        """A GTT whose fate the engine cannot tell: a MISMATCH posted, never a guess; the
        journal keeps it TRIGGERED for the user to settle."""
        self._followed.pop(followed.gtt.seq, None)
        self._outcome(followed, GttEvent.MISMATCH, candidates=candidates, message=words)

    def _end(self, followed: _Followed, event: GttEvent, note: str) -> None:
        self._journal.set_gtt_state(followed.gtt.seq, GttState.ENDED, note=note)
        self._followed.pop(followed.gtt.seq, None)
        self._outcome(followed, event, message=note)

    def taken(self, order_id: str) -> bool:
        """Whether an order is the session's own or another GTT's child (3.2 rule 2)."""
        if order_id in self._children or self._journal.intent_of(order_id) is not None:
            return True
        return self._journal.gtt_of_child(order_id) is not None

    def _outcome(
        self,
        followed: _Followed,
        event: GttEvent,
        *,
        order_id: str = "",
        units: int = 0,
        price: float = 0.0,
        flags: tuple[str, ...] = (),
        candidates: tuple[str, ...] = (),
        message: str = "",
    ) -> None:
        gtt = followed.gtt
        outcome = GttOutcome(
            seq=gtt.seq,
            strategy=gtt.strategy,
            gtt_id=gtt.gtt_id,
            trigger_id=gtt.trigger_id or "",
            symbol=gtt.symbol,
            exchange=gtt.exchange,
            product=gtt.product,
            event=event,
            order_id=order_id,
            units=units,
            price=price,
            flags=flags,
            candidates=candidates,
            message=message,
        )
        logger.info("GTT %s of %s: %s", gtt.trigger_id, gtt.strategy, event.value)
        self._post(GttNews(outcome))


__all__ = ["CHILD_READ_EVERY_S", "CHILD_SEARCH_S", "GTT_BOOK_EVERY_S", "GttFollower"]
