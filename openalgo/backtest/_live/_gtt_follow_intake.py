"""What the GTT follower (`_gtt_follow`) takes in from the GTT calls, the order worker and
the session, split from it so each module keeps one job (the lead's ruling N3b-0 (b)): the
journal's GTTs taken back at a start, the session's reads of a leg's units taken as
references, each GTT handed over by its call's confirmation, the GTTs a call holds, the
terms of a modify the broker may hold instead of the journal's, and a fired GTT's child
bound. `FollowerIntake` is a part of `_gtt_follow.GttFollower`, whose
state it changes; its module docstring says what each of these means for a GTT.

Times are UTC nanoseconds, quantities signed whole units. Used by the worker's thread only.
"""

from __future__ import annotations

from ._gtt_followed import Followed
from ._gtt_reads import BookRead, GttReader, LegKey
from ._gtt_units import References
from ._journal_gtt_rows import GttBinding, GttState, GttTerms
from .journal import Journal


class FollowerIntake:
    """The follower's intake (module docstring), mixed into `_gtt_follow.GttFollower`,
    which holds the reader, the journal, the GTTs followed, their references, the GTTs a
    call holds, whether a read was asked for, when the next read on the clock is due, and
    `observe`, which judges a read."""

    _reader: GttReader
    _journal: Journal
    _followed: dict[int, Followed]
    _references: References
    _held: set[int]
    _asked: bool
    _next_read_ns: int
    _book_every_ns: int

    def observe(self, read: BookRead) -> None:
        """Judge `read` (`_gtt_follow.GttFollower.observe`)."""
        raise NotImplementedError

    def restore(self) -> None:
        """Follow every GTT of the journal the broker holds that has not ended, and read
        the book at the first step when there is any (20-positional-live 4.2 step 5). Its
        child is looked for from its last change in the journal (it was placed or modified
        then). It has no reference until a settled read shows it resting (`_gtt_units`)."""
        for gtt in self._journal.open_gtts():
            if gtt.trigger_id is not None:
                self._followed[gtt.seq] = Followed(gtt, active_ns=self._last_change(gtt))
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

    def reference(self, leg: LegKey, units: int, read_ns: int) -> bool:
        """Take the session's read of `leg` made at `read_ns` (UTC ns), `units` at the
        broker, as the leg's reference when the leg is settled for it
        (`_own_fills.OwnFills.settled` at `read_ns`: no own order on the leg works now, and
        no own fill on it was heard from `POSITION_BOOK_LAG_S` before the read until now);
        the session's own units on the leg then are the order path's count now
        (`OwnFills.units`). It is the leg's latest reference unless one read later is held
        (`_gtt_follow` module docstring). True when it was taken; a read that is not
        settled is ignored (False): it never becomes a reference. Raises JournalError when
        the unbound intents cannot be read.

        The session gives its start read of the position book and holdings
        (`_broker_units`), made after `OrderPath.restore` at every start (a same-day
        restart included: the day's own fills the order path took back are counted), for
        each leg it may hold a GTT on, and every later settled position-book read of the
        reconciler; it gives each at once after the read, on the worker's thread (through
        the worker's queue, as `ReadGtts` reaches `read_now`), before an order of its own
        on the leg is sent. Without a reference from before its place, a GTT has none
        until a settled read shows it resting (the GTT parts' reads: after its calls, and
        every `GTT_BOOK_EVERY_S` while the market is open): until then its cancel once
        gone from the book is CANCEL_REFUSED and the GTT gone a MISMATCH, and an
        unanswered place with none from before it is UNRESOLVED."""
        own = self._reader.own
        if not own.settled(leg, read_ns):
            return False
        self._references.take(leg, units, own.units(leg), read_ns)
        return True

    def placing(self, seq: int, leg: LegKey) -> None:
        """GTT `seq`'s place on `leg` is about to be sent: the leg's latest reference is
        the GTT's."""
        self._references.pin(seq, leg)

    def follow(self, gtt: GttBinding, read: BookRead) -> None:
        """Follow `gtt`, which `read` showed resting (a place or modify confirmed): its
        leg's latest reference is its."""
        followed = Followed(gtt, active_ns=read.read_ns)
        self._followed[gtt.seq] = followed
        self._held.discard(gtt.seq)
        self._references.pin(gtt.seq, followed.leg)
        self._next_read_ns = max(self._next_read_ns, read.read_ns + self._book_every_ns)

    def take_over(self, gtt: GttBinding, read: BookRead, also: GttTerms | None = None) -> None:
        """Judge `gtt` by `read` from now: a call's confirmation found it no longer
        resting (fired, gone or ended). Its child, if any, came after it was recorded.
        `also` are the terms of a modify the broker may hold instead of the journal's (the
        replace of a modify that cannot be told): its child is looked for, and its leg's
        units judged, by both."""
        followed = self._followed.setdefault(gtt.seq, Followed(gtt, active_ns=gtt.recorded_ns))
        followed.gtt, followed.also = gtt, also
        self._held.discard(gtt.seq)
        self.observe(read)

    def may_hold(self, seq: int, terms: GttTerms) -> None:
        """The broker may hold `terms` for GTT `seq` instead of the journal's (a modify that
        cannot be told whose cancel to place it again was refused): its child is looked
        for, and its leg's units judged, by both (`take_over`'s `also`)."""
        followed = self._followed.get(seq)
        if followed is not None:
            followed.also = terms

    def forget(self, seq: int) -> None:
        """Stop following GTT `seq` (its cancel was confirmed, or its place was not
        made), its reference with it."""
        self._followed.pop(seq, None)
        self._held.discard(seq)
        self._references.drop(seq)

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

    def bind_child(self, seq: int, order_id: str, kernel_order: int) -> None:
        """Bind the child the follower found for GTT `seq` as the kernel's order
        `kernel_order`, and note the GTT fired (`inbox`'s `BindGttChild`). Raises
        JournalError as `Journal.bind_gtt_child` does."""
        self._journal.bind_gtt_child(seq, order_id, kernel_order)
        self._journal.set_gtt_state(seq, GttState.FIRED, note=f"its child is order {order_id}")


__all__ = ["FollowerIntake"]
