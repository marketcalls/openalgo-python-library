"""The confirmations of the order worker's GTT calls, split from the GTT sender
(`_gtt_send`) so each module keeps one job (the lead's ruling N3b-0 (b)): each call waits
(`Pending`) for reads of the GTT book (`_gtt_confirm.judge`), the order book where its
verdict needs it (`_gtt_checks`), and is answered to the loop once they decide, the GTT
then handed to the follower (`_gtt_follow`). `Confirmations` is a part of
`_gtt_send.GttSender`, whose calls it uses to place again and to cancel.

**The replace of a modify that cannot be told** (REPLACE_CANCEL; 20-positional-live 3.1
"Unanswered calls"; the lead's ruling N3b-0 (a)): the broker may hold the GTT's terms
before the modify or the modify's, so its child is looked for in the order book by both,
and its leg's units judged by both. A child found by either refuses the cancel for the
replace: the answer is UNRESOLVED and nothing is placed again, and the follower takes the
GTT over with both sets of terms.

Times are UTC nanoseconds. Used by the worker's thread only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .._layout import NANOS_PER_SECOND
from ._gtt_calls import CallResult
from ._gtt_checks import again_verdict, child_verdict
from ._gtt_child import child_candidates
from ._gtt_confirm import READ_AGAIN_AFTER_S, Pending, Stage, Step, Verdict, judge, misses_from_ns
from ._gtt_follow import GttFollower
from ._gtt_reads import BookRead, GttReader
from ._gtt_types import GttAnswerKind, GttRequest, gtt_key
from ._journal_gtt_rows import GttBinding, GttState, GttTerms
from ._types import LOGGER_NAME
from ._worker_queue import WorkerQueue
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)

#: The journal's notes of a place that was never found, and of a GTT cancelled to be
#: placed again.
NOT_FOUND_NOTE = "its place was unanswered and the GTT book did not show it: placed again"
REPLACED_NOTE = "cancelled to be placed again: its modify could not be told"
_NS = NANOS_PER_SECOND


class Confirmations:
    """The confirmations (module docstring), mixed into `_gtt_send.GttSender`, which holds
    the reader, the follower, the journal, the worker's queue, the clock, the calls
    waiting (`_pending`, by journal GTT), and the calls `_place`, `_send_cancel`,
    `_answer` and `_bound`."""

    _reader: GttReader
    _follower: GttFollower
    _journal: Journal
    _queue: WorkerQueue
    _now: Callable[[], int]
    _pending: dict[int, Pending]

    def _place(self, request: GttRequest, stage: Stage, *, again: bool = False) -> None:
        raise NotImplementedError

    def _send_cancel(self, stage: Stage, request: GttRequest, gtt: GttBinding) -> None:
        raise NotImplementedError

    def _answer(
        self,
        request: GttRequest,
        kind: GttAnswerKind,
        *,
        seq: int | None = None,
        trigger_id: str = "",
        message: str = "",
        kept: bool = False,
    ) -> None:
        raise NotImplementedError

    def _bound(self, trigger_id: str) -> bool:
        raise NotImplementedError

    def due(self) -> None:
        """Read again for each call whose confirmation is due: while OpenAlgo is silent
        too, as the order path reads for an unbound intent, since nothing more is sent for
        the GTT until its call is settled, and OpenAlgo's own answer to the read lifts the
        hold."""
        now = self._now()
        for pending in list(self._pending.values()):
            if pending.next_ns <= now and self._pending.get(pending.seq) is pending:
                self._confirm(pending)

    def _start(self, pending: Pending, result: CallResult) -> None:
        """Wait for `pending`'s confirmation, its call answered by `result` just now."""
        pending.finished, pending.sent_ns = result.finished, result.sent_ns
        pending.misses_from_ns = misses_from_ns(
            pending.stage, finished=result.finished, sent_ns=result.sent_ns, answered_ns=self._now()
        )
        self._pending[pending.seq] = pending
        self._queue.hold(gtt_key(pending.request))
        self._follower.hold(pending.seq)
        self._confirm(pending)

    def _confirm(self, pending: Pending) -> None:
        request = pending.request
        legs = self._follower.legs() | {(request.symbol, request.exchange, request.product)}
        read = self._reader.gtt_book(legs)
        self._follower.observe(read)
        self._act(pending, self._judge(pending, read), read)

    def _judge(self, pending: Pending, read: BookRead) -> Verdict:
        """`_gtt_confirm.judge`, given what the leg's units say of the place's GTT."""
        request = pending.request
        leg = (request.symbol, request.exchange, request.product)
        fall = self._follower.units_of(pending.seq, leg, request.terms, read)
        return judge(pending, read, self._bound, fall)

    def _act(self, pending: Pending, verdict: Verdict, read: BookRead) -> None:
        step = verdict.step
        if step is Step.WAIT:
            pending.next_ns = self._now() + round(READ_AGAIN_AFTER_S * _NS)
        elif step is Step.BIND:
            self._journal.bind_gtt(pending.seq, verdict.trigger_id)
            pending.trigger_id, pending.misses = verdict.trigger_id, 0
            self._act(pending, self._judge(pending, read), read)
        elif step is Step.PLACE_AGAIN:
            self._drop(pending)
            self._journal.set_gtt_state(pending.seq, GttState.NOT_PLACED, note=NOT_FOUND_NOTE)
            self._place(pending.request, pending.stage, again=True)
        elif step is Step.NOT_FOUND:
            self._drop(pending)
            self._journal.set_gtt_state(pending.seq, GttState.NOT_PLACED, note=verdict.message)
            self._answer(pending.request, GttAnswerKind.UNRESOLVED, message=verdict.message)
        elif step is Step.REPLACE:
            self._drop(pending)
            logger.warning(
                "GTT %s: %s; cancelled and placed again", pending.trigger_id, verdict.message
            )
            gtt = self._journal.gtt(pending.seq)
            if gtt is not None:
                self._journal.set_gtt_state(gtt.seq, GttState.CANCELLING, note=verdict.message)
                self._send_cancel(Stage.REPLACE_CANCEL, pending.request, gtt)
        elif step in (Step.CHILD_CHECK, Step.FIRED_CHECK):
            self._book_check(pending, verdict, read)
        else:
            self._finish(pending, verdict, read)

    def _book_check(self, pending: Pending, verdict: Verdict, read: BookRead) -> None:
        """Read the order book before deciding (one not read is read again): a cancelled GTT
        absent or ended (CHILD_CHECK) is confirmed only when no child of it is in the book
        (3.1) and, gone from the book, a settled read shows its leg's units unchanged
        (`_gtt_checks.child_verdict`); a replace's cancel looks for the child, and judges
        the units, by the modify's terms too (module docstring); an unanswered place
        (FIRED_CHECK) is placed again only when no order of its child's shape was placed
        since it was sent (`again_verdict`)."""
        rows, gtt = self._reader.order_rows(), self._journal.gtt(pending.seq)
        if rows is None or gtt is None:
            pending.next_ns = self._now() + round(READ_AGAIN_AFTER_S * _NS)
            return
        taken = self._follower.taken
        if verdict.step is Step.FIRED_CHECK:
            found = child_candidates(gtt, rows, since_ns=pending.sent_ns, taken=taken)
            self._act(pending, again_verdict(pending, found), read)
            return
        also = pending.request.terms if pending.stage is Stage.REPLACE_CANCEL else None
        since = self._follower.last_active(pending.seq)
        found = child_candidates(gtt, rows, since_ns=since, taken=taken, also=also)
        leg = (gtt.symbol, gtt.exchange, gtt.product)
        units = self._follower.units_of(gtt.seq, leg, gtt.terms, read, also)
        self._act(pending, child_verdict(pending, verdict.seen, found, units, read), read)

    def _finish(self, pending: Pending, verdict: Verdict, read: BookRead) -> None:
        """Answer `pending` as `verdict` says: noted in the journal, answered, then the GTT
        handed to the follower, whose outcomes come after the answer (`_gtt_send` module
        docstring)."""
        self._drop(pending)
        seq, request, kind = pending.seq, pending.request, verdict.kind
        replace = pending.stage is Stage.REPLACE_CANCEL
        if replace and kind is GttAnswerKind.CANCELLED:
            self._journal.set_gtt_state(seq, GttState.CANCELLED, note=REPLACED_NOTE)
            self._follower.forget(seq)
            self._place(request, Stage.REPLACE_PLACE)
            return
        if kind is None or not pending.trigger_id:
            # An unanswered place found several times, or that may have landed and fired:
            # the journal keeps it requested, bound to none, for the user to settle.
            self._answer(request, GttAnswerKind.UNRESOLVED, message=verdict.message)
            return
        if replace:
            kind = GttAnswerKind.UNRESOLVED
        elif pending.stage is Stage.REPLACE_PLACE and kind is GttAnswerKind.PLACED:
            kind = GttAnswerKind.MODIFIED
        resting = self._note(seq, kind, request, verdict)
        trigger_id = pending.trigger_id
        self._answer(request, kind, seq=seq, trigger_id=trigger_id, message=verdict.message)
        also = request.terms if replace else None
        self._hand_over(seq, kind, read, resting=resting, also=also)

    def _note(self, seq: int, kind: GttAnswerKind, request: GttRequest, verdict: Verdict) -> bool:
        """Note the answer in the journal: cancelled, or its terms the modify's when it
        was modified, and active while it rests. True when the GTT rests."""
        if kind is GttAnswerKind.CANCELLED:
            self._journal.set_gtt_state(seq, GttState.CANCELLED)
            return False
        if kind is GttAnswerKind.MODIFIED:
            self._journal.set_gtt_terms(seq, request.terms)
        resting = verdict.seen is None
        if resting:
            self._journal.set_gtt_state(seq, GttState.ACTIVE, note=verdict.message)
        return resting

    def _hand_over(
        self,
        seq: int,
        kind: GttAnswerKind,
        read: BookRead,
        *,
        resting: bool,
        also: GttTerms | None = None,
    ) -> None:
        """Tell the follower: a cancelled GTT is forgotten; one still resting is followed,
        this read the latest that showed it resting; one no longer resting is the
        follower's to judge from this read, with the terms of a modify the broker may hold
        instead (`also`, the replace's)."""
        if kind is GttAnswerKind.CANCELLED:
            self._follower.forget(seq)
            return
        gtt = self._journal.gtt(seq)
        if gtt is None:
            return
        if resting:
            self._follower.follow(gtt, read)
        else:
            self._follower.take_over(gtt, read, also)

    def _drop(self, pending: Pending) -> None:
        if self._pending.get(pending.seq) is pending:
            del self._pending[pending.seq]
        self._queue.release(gtt_key(pending.request))
        if not pending.trigger_id:
            # An unanswered place bound to no GTT: nothing of it is followed.
            self._follower.forget(pending.seq)


__all__ = ["NOT_FOUND_NOTE", "REPLACED_NOTE", "Confirmations"]
