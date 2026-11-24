"""The GTT follower of the order worker (20-positional-live 3.1 to 3.3; the lead's ruling
N3a-2): every GTT of the journal the broker holds, judged at each read of the GTT book,
and the child order of one that fired found in the order book. Each finding is a typed
outcome posted to the loop (`_gtt_types.GttOutcome`, as `inbox.GttNews`, by
`_gtt_outcomes`), and the journal notes each state.

**When it reads** (3.1): after every GTT call (the calls' confirmations, `_gtt_send`, whose
reads it judges too: `observe`); every `GTT_BOOK_EVERY_S` while the market is open and a
GTT it follows rests; at once when the order stream shows an order the journal does not
know on a symbol with a GTT (`order_seen`); when the session asks (the start's read and
the close's, `inbox`'s `ReadGtts`: `ask`); and once more before a GTT gone with no child
is decided. Nothing is read while OpenAlgo is silent (`busy`): a read could hold the
worker for the client's whole timeout. A read that fails changes nothing.

**What a read says** (`_gtt_book.Seen`, the table of 3.1): resting, the GTT's leg's units
read just before it become its baseline (`_gtt_reads`), moved by each fill of the session's
own orders on the leg since (`_gtt_followed`), kept in the journal whenever it changes so a
later day judges it too; a resting row that shows another trigger or other units than the
journal's is posted once as OTHER_TERMS and stays followed; an unknown status is taken as
resting, with an ERROR alert once per GTT (3.1 asks for a WARN; the alerts have no such
kind); ended without
firing (cancelled, deleted, disabled), expired or rejected after placement, the GTT ends
in the journal and its outcome is posted (the session places it again, 3.4); fired or
gone, its leg is in flight (IN_FLIGHT posted) while its child is looked for. A GTT taken
as gone that a later read shows resting (a book that left it out of a read) rests again:
RESTING is posted and nothing fired. A GTT whose row cannot be read is not judged by that
read. A GTT with a call of the order worker pending is the call's to judge (`hold`).

**The child** (3.2, 3.3; `_gtt_child`): the sandbox's book names it; otherwise the order
book is read every `CHILD_READ_EVERY_S` for `CHILD_SEARCH_S`: one candidate is the child
(TRIGGERED), several a MISMATCH naming them, never a guess. None by then: the GTT book is
read once more (still resting: RESTING), then the leg's units read with it against its
baseline say whether it fired with the child missed (CHILD_MISSED, priced from the trade
book or at the GTT's limit, flagged), ended outside the session (ENDED_OUTSIDE), or cannot
be told (MISMATCH). A GTT with no baseline (no read showed it resting with its position,
on any day) cannot be told either. The journal keeps a fired GTT TRIGGERED until the
session binds its child (`bind_child`), so a restart before that looks for the child again.

Times are UTC nanoseconds, prices rupees, quantities signed whole units. Used by the
worker's thread only.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._gtt_book import GttRow, Seen
from ._gtt_child import Move, child_candidates, move_of
from ._gtt_followed import ENDINGS, Followed, other_terms, unknown_status
from ._gtt_outcomes import GttOutcomes
from ._gtt_reads import BookRead, GttReader, LegKey
from ._gtt_types import GttEvent
from ._journal_gtt_rows import GttBinding, GttState
from ._types import LOGGER_NAME, AlertKind, AlertsLike, Now, OrderUpdate
from .inbox import Message
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)

#: Seconds between two reads of the GTT book while the market is open and a GTT rests
#: (20-positional-live 3.1).
GTT_BOOK_EVERY_S = 10.0
#: Seconds between two order-book reads while a fired or gone GTT's child is looked for,
#: and how long it is looked for before the leg's units decide (3.2 rule 4).
CHILD_READ_EVERY_S = 2.0
CHILD_SEARCH_S = 60.0
#: The journal's note of a GTT taken as gone that a read shows resting again.
RESTING_AGAIN_NOTE = "a read shows it resting again: it was left out of an earlier read"
_NS = NANOS_PER_SECOND


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
        self._alerts = alerts
        self._now = now
        self._market_open = market_open
        self._busy = busy
        self._followed: dict[int, Followed] = {}
        self._outcomes = GttOutcomes(journal, post, self._followed)
        self._held: set[int] = set()
        self._asked = False
        self._next_read_ns = 0

    # ----- what the GTT calls and the worker tell it --------------------------------------

    def restore(self) -> None:
        """Follow every GTT of the journal the broker holds that has not ended, and read
        the book at the first step when there is any (20-positional-live 4.2 step 5). Its
        child is looked for from its last change in the journal (it was placed or modified
        then), and its baseline is the one the journal kept, then each read that shows it
        resting."""
        for gtt in self._journal.open_gtts():
            if gtt.trigger_id is not None:
                active_ns = self._last_change(gtt)
                self._followed[gtt.seq] = Followed(gtt, gtt.baseline_units, active_ns)
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
        known = self._followed.get(gtt.seq)
        followed = Followed(gtt, gtt.baseline_units, active_ns=read.read_ns)
        if known is not None:
            followed.units_before, followed.own_before = known.units_before, known.own_before
        self._followed[gtt.seq] = followed
        self._held.discard(gtt.seq)
        self._keep_baseline(followed, read)
        self._next_read_ns = max(self._next_read_ns, read.read_ns + round(GTT_BOOK_EVERY_S * _NS))

    def take_over(self, gtt: GttBinding, read: BookRead) -> None:
        """Judge `gtt` by `read` from now: a call's confirmation found it no longer
        resting (fired, gone or ended). Its child, if any, came after it was recorded."""
        fresh = Followed(gtt, gtt.baseline_units, active_ns=gtt.recorded_ns)
        self._followed.setdefault(gtt.seq, fresh).gtt = gtt
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

    def fired_by_units(self, gtt: GttBinding, read: BookRead) -> bool:
        """Whether the units of `gtt`'s leg in `read` moved from its baseline (moved by the
        session's own fills) by the GTT's units on its side: it fired (3.3). False when
        either is not known."""
        followed = self._followed.get(gtt.seq)
        if followed is None:
            return False
        self._shift(followed, read.own.get(followed.leg))
        units, before = read.units.get(followed.leg), followed.units_before
        if before is None or units is None:
            return False
        return move_of(gtt.terms, before, units) is Move.FIRED

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
        """Move each baseline by the session's own fills since, then read what is due
        (module docstring) and judge it."""
        for followed in self._followed.values():
            self._shift(followed, self._reader.own(followed.leg))
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
            if followed.gtt.seq in self._held or followed.gtt.trigger_id is None:
                continue
            row = read.rows.get(followed.gtt.trigger_id)
            seen = Seen.GONE if row is None else row.seen
            if seen is Seen.UNREADABLE:
                continue
            if followed.since_ns is not None:
                if followed.child is None and seen.rests:
                    self._rests_again(followed, row, read)
            elif seen.rests:
                self._resting(followed, row, read)
            elif seen in (Seen.TRIGGERED, Seen.GONE):
                self._fired_or_gone(followed, row, seen, read.read_ns)
            else:
                event, note = ENDINGS[seen]
                self._outcomes.end(followed, event, note.format(status=row.status if row else ""))

    def _resting(self, followed: Followed, row: GttRow | None, read: BookRead) -> None:
        followed.active_ns = read.read_ns
        self._keep_baseline(followed, read)
        if followed.gtt.state is GttState.PLACED:
            self._journal.set_gtt_state(followed.gtt.seq, GttState.ACTIVE)
            followed.gtt = dataclasses.replace(followed.gtt, state=GttState.ACTIVE)
        words = "" if row is None else other_terms(followed, row)
        if words:
            gtt = followed.gtt
            logger.warning("GTT %s of %s: %s", gtt.trigger_id, gtt.strategy, words)
            self._outcomes.post(followed, GttEvent.OTHER_TERMS, message=words)
        text = "" if row is None else unknown_status(followed, row)
        if text:
            logger.error(text)
            self._alerts.send(AlertKind.ERROR, text)

    def _keep_baseline(self, followed: Followed, read: BookRead) -> None:
        """The leg's units in `read`, which showed the GTT resting, are its baseline (or,
        not read, the baseline is moved by the session's own fills); the journal keeps it
        whenever it changes (a later day judges the GTT by it)."""
        units, own = read.units.get(followed.leg), read.own.get(followed.leg)
        if units is None or own is None:
            self._shift(followed, own)
        elif followed.rebase(units, own):
            self._journal.set_gtt_baseline(followed.gtt.seq, units)

    def _shift(self, followed: Followed, own: int | None) -> None:
        """Move `followed`'s baseline by the session's own fills on its leg up to `own`,
        what they have filled by now (None: not known), kept in the journal when it moved."""
        if own is not None and followed.shift(own) and followed.units_before is not None:
            self._journal.set_gtt_baseline(followed.gtt.seq, followed.units_before)

    def _rests_again(self, followed: Followed, row: GttRow | None, read: BookRead) -> None:
        """A GTT taken as gone that `read` shows resting: nothing fired (module docstring)."""
        followed.since_ns, followed.next_ns = None, 0
        self._journal.set_gtt_state(followed.gtt.seq, GttState.ACTIVE, note=RESTING_AGAIN_NOTE)
        followed.gtt = dataclasses.replace(followed.gtt, state=GttState.ACTIVE)
        self._outcomes.post(followed, GttEvent.RESTING, message=RESTING_AGAIN_NOTE)
        self._resting(followed, row, read)

    def _fired_or_gone(
        self, followed: Followed, row: GttRow | None, seen: Seen, at_ns: int
    ) -> None:
        note = "triggered" if seen is Seen.TRIGGERED else "gone from the GTT book"
        if followed.gtt.state is not GttState.TRIGGERED:
            self._journal.set_gtt_state(followed.gtt.seq, GttState.TRIGGERED, note=note)
            followed.gtt = dataclasses.replace(followed.gtt, state=GttState.TRIGGERED)
        followed.since_ns, followed.next_ns = at_ns, at_ns
        followed.child = None if row is None else row.child_order_id
        self._outcomes.post(followed, GttEvent.IN_FLIGHT, message=note)
        if followed.child is not None:
            self._outcomes.triggered(followed, followed.child)

    # ----- the child --------------------------------------------------------------------

    def _search(self, followed: Followed, rows: list[Mapping[str, Any]] | None, now: int) -> None:
        followed.next_ns = now + round(CHILD_READ_EVERY_S * _NS)
        if rows is None:
            return
        found = child_candidates(followed.gtt, rows, since_ns=followed.active_ns, taken=self.taken)
        if found.child is not None:
            self._outcomes.triggered(followed, found.child)
        elif found.every:
            words = f"orders {', '.join(found.every)} could each be its child"
            self._outcomes.unresolved(followed, words, found.every)
        elif followed.since_ns is not None and now - followed.since_ns >= CHILD_SEARCH_S * _NS:
            self._settle_gone(followed)

    def _settle_gone(self, followed: Followed) -> None:
        """No child was found in time: the GTT book read once more, and the leg's units
        read with it, decide (module docstring)."""
        read = self._reader.gtt_book({followed.leg})
        if read.rows is None:
            return
        row = read.rows.get(followed.gtt.trigger_id or "")
        if row is not None and row.seen.rests:
            self._rests_again(followed, row, read)
            return
        self._shift(followed, read.own.get(followed.leg))
        units = read.units.get(followed.leg)
        if units is None or (row is not None and row.seen is Seen.UNREADABLE):
            return
        before = followed.units_before
        if before is None:
            words = "no read showed it resting with its position, so whether it fired is unknown"
            self._outcomes.unresolved(followed, words)
            return
        move = move_of(followed.gtt.terms, before, units)
        if move is Move.FIRED:
            self._outcomes.missed(followed, self._reader.trade_book())
        elif move is Move.UNCHANGED:
            note = "gone with no child found and the position unchanged: ended outside the session"
            self._outcomes.end(followed, GttEvent.ENDED_OUTSIDE, note)
        else:
            self._outcomes.unresolved(followed, f"the position went from {before} to {units} units")

    def taken(self, order_id: str) -> bool:
        """Whether an order is the session's own or another GTT's child (3.2 rule 2)."""
        return self._outcomes.taken(order_id)


__all__ = [
    "CHILD_READ_EVERY_S",
    "CHILD_SEARCH_S",
    "GTT_BOOK_EVERY_S",
    "RESTING_AGAIN_NOTE",
    "GttFollower",
]
