"""The order worker's GTT sender (20-positional-live 3.1, 3.5, 3.6; the lead's ruling
N3a-1): each `GttRequest` sent through `_gtt_calls`, journal first, confirmed by reads of
the GTT book (`_gtt_confirm`), and answered to the loop as one `GttAnswer` (posted as
`inbox.GttNews`).

- **Journal first**: a place is written (`record_gtt`) before its call and bound to the
  broker's trigger ID on success (`bind_gtt`); every other state is noted as it comes
  (`set_gtt_state`, and `set_gtt_terms` for a confirmed modify).
- **One call at a time per GTT**: while a call waits for its confirmation, the GTT's
  requests are held in the worker's queue, and the follower leaves the GTT to the
  confirmation (`_gtt_follow.GttFollower.hold`). The confirmation's reads serve the
  follower too. An unanswered call the server may still be working on is waited for, as
  the order path waits for an unbound intent (`_gtt_confirm.misses_from_ns`).
- **The answer first**: a call's answer is posted before the follower takes the GTT on,
  so the loop learns a GTT was placed before it learns it fired or ended (`GttNews`).
- **Refusals** (3.5): the answer carries the broker's message. A leg's stop GTT refused
  when the journal already holds a refusal of a stop GTT of the same leg (strategy,
  symbol, exchange and product, as `kept_refusals` and its preflight check key it) from an
  earlier trading day is kept (`keep_refusal`): the strategy is refused at every later
  start until the user changes its product or `positional_stops`. A first day's refusal
  is the day's alone, and so is a strategy's GTT entry's: only a stop's refusal exits a
  leg every day, and `positional_stops` is no remedy for an entry.
- **Held back**: while OpenAlgo is silent (a timeout or a dropped connection) a request is
  put back and noticed as `NotSent(silent)` at most every `HELD_NOTICE_EVERY_S`, as an
  order's; after a mode guard lock it is `NotSent(locked)`.

Times are UTC nanoseconds. Used by the worker's thread only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .._layout import NANOS_PER_SECOND
from ._gtt_calls import CallKind, CallResult, GttCalls
from ._gtt_checks import again_verdict, child_verdict
from ._gtt_child import child_candidates
from ._gtt_confirm import READ_AGAIN_AFTER_S, Pending, Stage, Step, Verdict, judge, misses_from_ns
from ._gtt_follow import GttFollower
from ._gtt_reads import BookRead, GttReader
from ._gtt_types import GttAction, GttAnswer, GttAnswerKind, GttRequest, gtt_key
from ._journal_gtt_rows import GttBinding, GttState, refusal_note
from ._types import LOGGER_NAME, GuardLike, Now, Purpose
from ._worker_queue import WorkerQueue
from .inbox import GttNews, Message, NotSent, NotSentReason
from .journal import Journal
from .orders import OrderPath

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a GTT request held back waits before it is tried again, and between two
#: notices of it: as an order's (`_worker_send`).
REQUEUE_AFTER_S = 0.5
HELD_NOTICE_EVERY_S = 5.0
#: The journal's notes of a place that was never found, and of a GTT cancelled to be
#: placed again.
NOT_FOUND_NOTE = "its place was unanswered and the GTT book did not show it: placed again"
REPLACED_NOTE = "cancelled to be placed again: its modify could not be told"
_NS = NANOS_PER_SECOND


class GttSender:
    """The sender (module docstring) of the order worker: `calls` and `reader` reach
    OpenAlgo, `follower` follows the GTTs it confirms; `journal` is handle W, `queue` the
    worker's, `path` the order path (its silence), `guard` the mode guard; `post` puts a
    message on the loop's inbox and `now` gives UTC nanoseconds."""

    def __init__(
        self,
        calls: GttCalls,
        reader: GttReader,
        follower: GttFollower,
        *,
        journal: Journal,
        queue: WorkerQueue,
        path: OrderPath,
        guard: GuardLike,
        post: Callable[[Message], None],
        now: Now,
    ) -> None:
        self._calls = calls
        self._reader = reader
        self._follower = follower
        self._journal = journal
        self._queue = queue
        self._path = path
        self._guard = guard
        self._post = post
        self._now = now
        self._pending: dict[int, Pending] = {}
        self._notices: dict[int, int] = {}

    @property
    def follower(self) -> GttFollower:
        """The GTT follower the confirmations hand each GTT to."""
        return self._follower

    def execute(self, request: GttRequest) -> None:
        """Send `request` (module docstring). Raises JournalError when the journal fails."""
        if self._guard.locked:
            why = "the mode guard is locked: the session sends nothing more"
            self._post(NotSent(request, NotSentReason.LOCKED, why))
            return
        if self._path.silent_since_ns is not None:
            self._hold_back(request)
            return
        self._notices.pop(request.request_id, None)
        if request.action is GttAction.PLACE:
            self._place(request, Stage.PLACE)
        elif request.action is GttAction.MODIFY:
            self._modify(request)
        else:
            self._cancel(request)

    def due(self) -> None:
        """Read again for each call whose confirmation is due: while OpenAlgo is silent
        too, as the order path reads for an unbound intent, since nothing more is sent for
        the GTT until its call is settled, and OpenAlgo's own answer to the read lifts the
        hold."""
        now = self._now()
        for pending in list(self._pending.values()):
            if pending.next_ns <= now and self._pending.get(pending.seq) is pending:
                self._confirm(pending)

    # ----- the calls --------------------------------------------------------------------

    def _place(self, request: GttRequest, stage: Stage, *, again: bool = False) -> None:
        seq = self._journal.record_gtt(
            request.strategy,
            request.gtt_id,
            request.request_id,
            request.symbol,
            request.exchange,
            request.product,
            request.terms,
            purpose=request.purpose,
        )
        self._follower.placing(seq, (request.symbol, request.exchange, request.product))
        result = self._calls.place(request)
        if result.kind is CallKind.ACCEPTED:
            self._journal.bind_gtt(seq, result.trigger_id)
            self._start(Pending(stage, request, seq, result.trigger_id), result)
            return
        if result.kind is CallKind.UNANSWERED:
            self._start(Pending(stage, request, seq, answered=False, again=again), result)
            return
        self._follower.forget(seq)
        if result.kind is CallKind.REFUSED:
            self._journal.set_gtt_state(seq, GttState.NOT_PLACED, note=refusal_note(result.message))
            kept = self._keep(request, result.message)
            self._answer(request, GttAnswerKind.REFUSED, message=result.message, kept=kept)
            self._follower.ask()
        else:
            self._answer(request, GttAnswerKind.UNRESOLVED, message=result.message)

    def _modify(self, request: GttRequest) -> None:
        gtt = self._known(request)
        if gtt is None:
            return
        self._journal.set_gtt_state(gtt.seq, GttState.MODIFYING)
        trigger_id = gtt.trigger_id or ""
        result = self._calls.modify(request, trigger_id)
        if result.kind in (CallKind.ACCEPTED, CallKind.UNANSWERED):
            answered = result.kind is CallKind.ACCEPTED
            pending = Pending(Stage.MODIFY, request, gtt.seq, trigger_id, answered, gtt.terms)
            self._start(pending, result)
            return
        self._journal.set_gtt_state(gtt.seq, GttState.ACTIVE, note="its modify was refused")
        self._refused(request, gtt, GttAnswerKind.MODIFY_REFUSED, result)

    def _cancel(self, request: GttRequest) -> None:
        gtt = self._known(request)
        if gtt is None:
            return
        self._journal.set_gtt_state(gtt.seq, GttState.CANCELLING)
        self._send_cancel(Stage.CANCEL, request, gtt)

    def _send_cancel(self, stage: Stage, request: GttRequest, gtt: GttBinding) -> None:
        trigger_id = gtt.trigger_id or ""
        result = self._calls.cancel(request.strategy, trigger_id)
        if result.kind in (CallKind.ACCEPTED, CallKind.UNANSWERED):
            answered = result.kind is CallKind.ACCEPTED
            self._start(Pending(stage, request, gtt.seq, trigger_id, answered), result)
            return
        self._journal.set_gtt_state(gtt.seq, GttState.ACTIVE, note="its cancel was refused")
        if stage is Stage.REPLACE_CANCEL:
            words = f"its modify could not be told, and its cancel was refused: {result.message}"
            result = CallResult(result.kind, message=words)
            self._refused(request, gtt, GttAnswerKind.UNRESOLVED, result)
        else:
            self._refused(request, gtt, GttAnswerKind.CANCEL_REFUSED, result)

    def _known(self, request: GttRequest) -> GttBinding | None:
        """The journal's GTT a modify or cancel is of, when the broker still holds it;
        else the request is answered now: nothing is left to change."""
        gtt = None if request.seq is None else self._journal.gtt(request.seq)
        held = gtt is not None and gtt.trigger_id is not None and not gtt.state.terminal
        if held and gtt is not None and gtt.state is not GttState.TRIGGERED:
            return gtt
        state = "not in the journal" if gtt is None else gtt.state.value
        fired = gtt is not None and gtt.state in (GttState.TRIGGERED, GttState.FIRED)
        if request.action is GttAction.MODIFY:
            kind = GttAnswerKind.MODIFY_REFUSED
        else:
            ended = gtt is not None and gtt.state.terminal
            kind = GttAnswerKind.CANCELLED if ended and not fired else GttAnswerKind.CANCEL_REFUSED
        words = f"GTT {request.seq} is {state}: there is nothing to change"
        self._answer(request, kind, seq=request.seq, message=words)
        return None

    def _refused(
        self, request: GttRequest, gtt: GttBinding, kind: GttAnswerKind, result: CallResult
    ) -> None:
        self._answer(
            request, kind, seq=gtt.seq, trigger_id=gtt.trigger_id or "", message=result.message
        )
        self._follower.release(gtt.seq)
        self._follower.ask()

    def _keep(self, request: GttRequest, message: str) -> bool:
        """Keep a stop GTT's refusal across starts when an earlier trading day's refusal
        of a stop GTT of the same leg is in the journal (module docstring)."""
        if request.purpose is not Purpose.STOP:
            return False
        leg = (request.strategy, request.symbol, request.exchange, request.product)
        if not self._journal.refused_days(*leg, Purpose.STOP):
            return False
        self._journal.keep_refusal(*leg, message)
        logger.error("the stop GTT of %s was refused again on a later day: kept", request.leg)
        return True

    # ----- the confirmations ------------------------------------------------------------

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
        (`_gtt_checks.child_verdict`; a modify's cancel judges the modify's units too); an
        unanswered place (FIRED_CHECK) is placed again only when no order of its child's
        shape was placed since it was sent (`again_verdict`)."""
        rows, gtt = self._reader.order_rows(), self._journal.gtt(pending.seq)
        if rows is None or gtt is None:
            pending.next_ns = self._now() + round(READ_AGAIN_AFTER_S * _NS)
            return
        taken = self._follower.taken
        if verdict.step is Step.FIRED_CHECK:
            found = child_candidates(gtt, rows, since_ns=pending.sent_ns, taken=taken)
            self._act(pending, again_verdict(pending, found), read)
            return
        since = self._follower.last_active(pending.seq)
        found = child_candidates(gtt, rows, since_ns=since, taken=taken)
        leg = (gtt.symbol, gtt.exchange, gtt.product)
        also = pending.request.terms if pending.stage is Stage.REPLACE_CANCEL else None
        units = self._follower.units_of(gtt.seq, leg, gtt.terms, read, also)
        self._act(pending, child_verdict(pending, verdict.seen, found, units, read), read)

    def _finish(self, pending: Pending, verdict: Verdict, read: BookRead) -> None:
        """Answer `pending` as `verdict` says: noted in the journal, answered, then the GTT
        handed to the follower, whose outcomes come after the answer (module docstring)."""
        self._drop(pending)
        seq, request, kind = pending.seq, pending.request, verdict.kind
        if pending.stage is Stage.REPLACE_CANCEL and kind is GttAnswerKind.CANCELLED:
            self._journal.set_gtt_state(seq, GttState.CANCELLED, note=REPLACED_NOTE)
            self._follower.forget(seq)
            self._place(request, Stage.REPLACE_PLACE)
            return
        if kind is None or not pending.trigger_id:
            # An unanswered place found several times, or that may have landed and fired:
            # the journal keeps it requested, bound to none, for the user to settle.
            self._answer(request, GttAnswerKind.UNRESOLVED, message=verdict.message)
            return
        if pending.stage is Stage.REPLACE_CANCEL:
            kind = GttAnswerKind.UNRESOLVED
        elif pending.stage is Stage.REPLACE_PLACE and kind is GttAnswerKind.PLACED:
            kind = GttAnswerKind.MODIFIED
        resting = self._note(seq, kind, request, verdict)
        trigger_id = pending.trigger_id
        self._answer(request, kind, seq=seq, trigger_id=trigger_id, message=verdict.message)
        self._hand_over(seq, kind, read, resting=resting)

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

    def _hand_over(self, seq: int, kind: GttAnswerKind, read: BookRead, *, resting: bool) -> None:
        """Tell the follower: a cancelled GTT is forgotten; one still resting is followed,
        this read the latest that showed it resting; one no longer resting is the
        follower's to judge from this read."""
        if kind is GttAnswerKind.CANCELLED:
            self._follower.forget(seq)
            return
        gtt = self._journal.gtt(seq)
        if gtt is None:
            return
        if resting:
            self._follower.follow(gtt, read)
        else:
            self._follower.take_over(gtt, read)

    def _drop(self, pending: Pending) -> None:
        if self._pending.get(pending.seq) is pending:
            del self._pending[pending.seq]
        self._queue.release(gtt_key(pending.request))
        if not pending.trigger_id:
            # An unanswered place bound to no GTT: nothing of it is followed.
            self._follower.forget(pending.seq)

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
        answer = GttAnswer(request, kind, seq, trigger_id, message, kept)
        logger.info("GTT request %d (%s): %s", request.request_id, request.leg, kind.value)
        self._post(GttNews(answer))

    def _bound(self, trigger_id: str) -> bool:
        return self._journal.gtt_by_trigger(trigger_id) is not None

    def _hold_back(self, request: GttRequest) -> None:
        now = self._now()
        self._queue.put_back(request, now + round(REQUEUE_AFTER_S * _NS))
        noticed = self._notices.get(request.request_id)
        if noticed is None or now - noticed >= HELD_NOTICE_EVERY_S * _NS:
            self._notices[request.request_id] = now
            why = "OpenAlgo has not answered since a timeout or a dropped connection"
            self._post(NotSent(request, NotSentReason.SILENT, why))


__all__ = ["HELD_NOTICE_EVERY_S", "NOT_FOUND_NOTE", "REPLACED_NOTE", "REQUEUE_AFTER_S", "GttSender"]
