"""The GTT follower of the order worker (20-positional-live 3.1 to 3.3; the lead's ruling
N3a-2): every GTT of the journal the broker holds, judged at each read of the GTT book,
and the child order of one that fired found in the order book. Each finding is a typed
outcome posted to the loop (`_gtt_types.GttOutcome`, as `inbox.GttNews`, by
`_gtt_outcomes`), and the journal notes each state. What it takes in from the GTT calls,
the worker and the session (the journal's GTTs at a start, references, GTTs handed over,
held or forgotten, a child bound) is `_gtt_follow_intake`'s.

**When it reads** (3.1): after every GTT call (the calls' confirmations, `_gtt_send`, whose
reads it judges too: `observe`); every `GTT_BOOK_EVERY_S` while the market is open and a
GTT it follows rests; at once when the order stream shows an order the journal does not
know on a symbol with a GTT (`order_seen`); when the session asks (the start's read and
the close's, `inbox`'s `ReadGtts`: `read_now`); and once more before a GTT gone with no
child is decided. Nothing is read while OpenAlgo is silent (`busy`): a read could hold the
worker for the client's whole timeout. A read that fails changes nothing.

**The references** (`_gtt_units`, the lead's ruling M1 on N3a's review round 5) are kept
here, in one place: each leg's latest settled read, from every read of the GTT book the
GTT parts make (the legs' units are read just before it, `_gtt_reads`) and from the
session's own reads (`reference`); a GTT's, the leg's latest when its place is sent
(`placing`) and again at each read that shows it resting. A GTT taken back from the
journal on a restart has none until such a read (`restore`). A read made while the
session's own fills on the leg had not settled (`_own_fills`) is never a reference.

**What a read says** (`_gtt_book.Seen`, the table of 3.1): resting, the GTT takes its
leg's latest reference; a resting row that shows another trigger or other units than the
journal's is posted once as OTHER_TERMS and stays followed; an unknown status is taken as
resting, with an ERROR alert once per GTT (3.1 asks for a WARN; the alerts have no such
kind); ended without firing (cancelled, deleted, disabled), expired or rejected after
placement, the GTT ends in the journal and its outcome is posted (the session places it
again, 3.4); fired or gone, its leg is in flight (IN_FLIGHT posted) while its child is
looked for. A GTT taken as gone that a later read shows resting (a book that left it out
of a read) rests again: RESTING is posted and nothing fired. A GTT whose row cannot be
read is not judged by that read. A GTT with a call of the order worker pending is the
call's to judge (`hold`), and still takes its leg's reference when a read shows it
resting.

**The child** (3.2, 3.3; `_gtt_child`): the sandbox's book names it, for a GTT that rested
today (one carried from a day it last rested on may have fired on a day no session ran,
whose orders today's books do not hold: 4.3); otherwise the order
book is read every `CHILD_READ_EVERY_S` for `CHILD_SEARCH_S`: one candidate is the child
(TRIGGERED), several a MISMATCH naming them, never a guess. A GTT taken over after the
replace of a modify that cannot be told is looked for by both sets of terms (the lead's
ruling N3b-0 (a): `Followed.also`). None by then: the GTT book is read once more (still
resting: RESTING), then the leg's units read with it against the GTT's reference say
(`_gtt_units.judge`) whether it fired with the child missed (CHILD_MISSED, priced from the
trade book or at the limit of the terms it fired on, flagged), ended outside the session
(ENDED_OUTSIDE; never for a row that says triggered: MISMATCH), or cannot be told
(MISMATCH). A GTT with no reference (no settled read of its leg made by this process when
it could not have fired) is a MISMATCH. A read made while the session's own fills on the
leg had not settled, or whose units could not be read, judges nothing: the book is read
again every `CHILD_READ_EVERY_S` until a read is settled (the newest own fill on the leg
is `_own_fills.POSITION_BOOK_LAG_S` old and no own order works on it), for at most
`_gtt_reads.UNITS_WAIT_S` after the later of when the GTT went and that newest fill (a
fill that comes during the wait moves the bound with it); still not settled then (an own
order still working on the leg, own fills coming one after another, or books that cannot
be read), it is a MISMATCH. The journal keeps a fired GTT TRIGGERED until the session
binds its child (`bind_child`), so a restart before that looks for the child again.

Times are UTC nanoseconds, prices rupees, quantities signed whole units. Used by the
worker's thread only.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

from .._layout import NANOS_PER_SECOND
from .._series import ist_time
from ._gtt_book import GttRow, Seen
from ._gtt_child import CHILD_READ_EVERY_S, CHILD_SEARCH_S, child_candidates
from ._gtt_follow_intake import FollowerIntake
from ._gtt_followed import ENDINGS, Followed, other_terms, unknown_status
from ._gtt_outcomes import GttOutcomes
from ._gtt_reads import BookRead, GttReader, LegKey
from ._gtt_types import GttBookSeen, GttEvent
from ._gtt_units import LegUnits, References, judge
from ._journal_gtt_rows import GttState, GttTerms
from ._types import LOGGER_NAME, AlertKind, AlertsLike, Now, OrderUpdate
from .inbox import Message
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)

#: Seconds between two reads of the GTT book while the market is open and a GTT rests
#: (20-positional-live 3.1).
GTT_BOOK_EVERY_S = 10.0
#: The journal's note of a GTT taken as gone that a read shows resting again.
RESTING_AGAIN_NOTE = "a read shows it resting again: it was left out of an earlier read"
_NS = NANOS_PER_SECOND


class GttFollower(FollowerIntake):
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
        self._followed: dict[int, Followed] = {}
        self._references = References()
        self._outcomes = GttOutcomes(journal, post, self._followed, self._references, now)
        self._held: set[int] = set()
        self._asked = False
        self._next_read_ns = 0
        self._book_every_ns = round(GTT_BOOK_EVERY_S * _NS)

    def units_of(
        self, seq: int, leg: LegKey, terms: GttTerms, read: BookRead, also: GttTerms | None = None
    ) -> LegUnits:
        """What `leg`'s units in `read` say of GTT `seq` of `terms` (or of `also`), against
        its reference (`_gtt_units.judge`; 3.3)."""
        return judge(terms, self._references.of(seq), read, leg, also)

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

    def close_read(self, legs: set[LegKey]) -> GttBookSeen:
        """Read the GTT book now with the units of `legs` and of the GTTs followed, judge
        it, and give what it found as the close's (`GttBookSeen`); while OpenAlgo is silent
        nothing is read, and the book is given as unread (None)."""
        if self._busy():
            return GttBookSeen(self._now(), None, {}, frozenset())
        read = self._reader.gtt_book(self.legs() | legs)
        self.observe(read)
        resting = None
        if read.rows is not None:
            rows = read.rows.values()
            resting = {r.trigger_id: (r.triggers, r.quantity) for r in rows if r.seen.rests}
        return GttBookSeen(read.read_ns, resting, dict(read.units), frozenset(read.settled))

    def order_seen(self, update: OrderUpdate) -> None:
        """The order stream showed an order the journal does not know: read at once when a
        GTT followed rests on its symbol (3.1)."""
        where = (update.symbol, update.exchange)
        if any(f.since_ns is None and f.leg[:2] == where for f in self._followed.values()):
            self._asked = True

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
            self._next_read_ns = now + self._book_every_ns
            self.observe(self._reader.gtt_book(self.legs()))
        searching = [f for f in self._followed.values() if f.since_ns is not None]
        if any(f.next_ns <= now for f in searching):
            rows = self._reader.order_rows()
            for followed in searching:
                if followed.next_ns <= now:
                    self._search(followed, rows, now)

    def observe(self, read: BookRead) -> None:
        """Take `read`'s settled legs as references, then judge every GTT followed and not
        held by it (module docstring); a read of the GTT book that failed changes nothing
        more."""
        self._references.take_read(read)
        if read.rows is None:
            return
        for followed in list(self._followed.values()):
            if followed.gtt.trigger_id is None:
                continue
            row = read.rows.get(followed.gtt.trigger_id)
            seen = Seen.GONE if row is None else row.seen
            if seen.rests:
                self._references.pin(followed.gtt.seq, followed.leg)
            if followed.gtt.seq in self._held or seen is Seen.UNREADABLE:
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
        named = None if row is None else row.child_order_id
        # A child the row names is today's order only when the GTT rested today: one that
        # fired on an earlier day is looked for in today's order book (module docstring).
        followed.child = named if self._rested_today(followed, at_ns) else None
        self._outcomes.post(followed, GttEvent.IN_FLIGHT, message=note)
        if followed.child is not None:
            self._outcomes.triggered(followed, followed.child)

    @staticmethod
    def _rested_today(followed: Followed, at_ns: int) -> bool:
        """Whether `followed`'s last read that showed it resting was on the IST day of
        `at_ns`."""
        rested = followed.active_ns
        return rested is not None and ist_time(rested).date() == ist_time(at_ns).date()

    # ----- the child --------------------------------------------------------------------

    def _search(self, followed: Followed, rows: list[Mapping[str, Any]] | None, now: int) -> None:
        followed.next_ns = now + round(CHILD_READ_EVERY_S * _NS)
        if rows is None:
            return
        found = child_candidates(
            followed.gtt, rows, since_ns=followed.active_ns, taken=self.taken, also=followed.also
        )
        if found.child is not None:
            self._outcomes.triggered(followed, found.child)
        elif found.every:
            words = f"orders {', '.join(found.every)} could each be its child"
            self._outcomes.unresolved(followed, words, found.every)
        elif followed.since_ns is not None and now - followed.since_ns >= CHILD_SEARCH_S * _NS:
            self._settle_gone(followed, followed.since_ns)

    def _settle_gone(self, followed: Followed, since_ns: int) -> None:
        """No child was found since the GTT fired or went at `since_ns`: the GTT book read
        once more, and the leg's units read with it against the GTT's reference, decide
        (`_gtt_outcomes.GttOutcomes.gone`; module docstring)."""
        read = self._reader.gtt_book({followed.leg})
        self._references.take_read(read)
        if read.rows is None:
            return
        row = read.rows.get(followed.gtt.trigger_id or "")
        if row is not None and row.seen.rests:
            self._references.pin(followed.gtt.seq, followed.leg)
            self._rests_again(followed, row, read)
            return
        if row is not None and row.seen is Seen.UNREADABLE:
            return
        gtt = followed.gtt
        found = self.units_of(gtt.seq, followed.leg, gtt.terms, read, followed.also)
        self._outcomes.gone(
            followed,
            found,
            waited_out=read.waited_out(followed.leg, since_ns),
            trades=self._reader.trade_book,
            triggered=row is not None and row.seen is Seen.TRIGGERED,
        )

    def taken(self, order_id: str) -> bool:
        """Whether an order is the session's own or another GTT's child (3.2 rule 2)."""
        return self._outcomes.taken(order_id)


__all__ = [
    "GTT_BOOK_EVERY_S",
    "RESTING_AGAIN_NOTE",
    "GttFollower",
]
