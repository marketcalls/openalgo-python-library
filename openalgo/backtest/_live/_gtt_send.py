"""The order worker's GTT sender (20-positional-live 3.1, 3.5, 3.6; the lead's ruling
N3a-1): each `GttRequest` sent through `_gtt_calls`, journal first, confirmed by reads of
the GTT book (`_gtt_confirm`, waited for by `_gtt_send_confirm`), and answered to the loop
as one `GttAnswer` (posted as `inbox.GttNews`).

- **Journal first**: a place is written (`record_gtt`) before its call and bound to the
  broker's trigger ID on success (`bind_gtt`); every other state is noted as it comes
  (`set_gtt_state`, and `set_gtt_terms` for a confirmed modify). A place a restart finds
  under way and unbound is settled as an unanswered place (`_settle_place`).
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
from ._gtt_confirm import Pending, Stage
from ._gtt_follow import GttFollower
from ._gtt_reads import GttReader
from ._gtt_send_confirm import NOT_FOUND_NOTE, REPLACED_NOTE, Confirmations
from ._gtt_types import GttAction, GttAnswer, GttAnswerKind, GttRequest
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
_NS = NANOS_PER_SECOND


class GttSender(Confirmations):
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
        if request.action is GttAction.PLACE and request.seq is not None:
            self._settle_place(request, request.seq)
        elif request.action is GttAction.PLACE:
            self._place(request, Stage.PLACE)
        elif request.action is GttAction.MODIFY:
            self._modify(request)
        else:
            self._cancel(request)

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

    def _settle_place(self, request: GttRequest, seq: int) -> None:
        """A place the journal shows under way and unbound since before a restart (journal
        GTT `seq`; the lead's ruling N5a-5): settled as an unanswered place the server may
        still be working on, sent when the journal recorded it (`_gtt_confirm`): one of its
        shape resting is bound and answered PLACED; none, and a settled read against a
        reference from before the place shows no fall, is placed again once; anything else
        is UNRESOLVED. Nothing is sent for it before that verdict. A GTT the journal no
        longer shows under way (settled meanwhile) is answered UNRESOLVED."""
        gtt = self._journal.gtt(seq)
        if gtt is None or gtt.trigger_id is not None or gtt.state is not GttState.REQUESTED:
            state = "not in the journal" if gtt is None else gtt.state.value
            words = f"GTT {seq} is {state}, not a place under way: it is not settled again"
            self._answer(request, GttAnswerKind.UNRESOLVED, message=words)
            return
        under_way = CallResult(CallKind.UNANSWERED, sent_ns=gtt.recorded_ns, finished=False)
        self._start(Pending(Stage.PLACE, request, seq, answered=False), under_way)

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
            # A cancel refused may be a GTT that fired on either set of terms.
            self._follower.may_hold(gtt.seq, request.terms)
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
