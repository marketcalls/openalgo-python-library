"""The GTT map of one strategy's kernel session (20-positional-live 2 and 3; the lead's
ruling N3b-2): the kernel's GTT commands turned into the order worker's GTT requests, and
each GTT answer and finding of the worker turned into the one kernel call it calls for, in
the style of the venue map's orders (`venue_map`).

- **Commands to requests.** A "place_gtt" becomes a `GttRequest` (purpose STOP for a
  leg's stop, ENTRY for a strategy's GTT order, which the command names by its order id;
  the trigger's side of the price as `Direction`, which the worker sends in OpenAlgo's
  field of that side, 3.6). A "modify_gtt" or "cancel_gtt" names the journal's GTT the
  place's answer bound (`seq`): one made before that answer waits here, in order, and
  goes with it. A trailing modify (the same units) is ADJUST, any other change of a stop
  PROTECT, as an order's (`_venue_requests`); every change of a GTT entry is ADJUST.
- **Answers to kernel calls.** PLACED `apply_gtt_placed`; REFUSED of a place
  `apply_gtt_refused` (a leg's stop: the kernel exits the leg and locks the day's entries,
  decision 11.7); REFUSED of a modify (its replacement was refused, the old GTT
  cancelled: nothing rests) `apply_gtt_modify_refused`, then `apply_gtt_cancelled`, after
  which the kernel places the leg's stop again (a second refusal exits it); MODIFIED,
  MODIFY_REFUSED, CANCELLED and CANCEL_REFUSED their calls; UNRESOLVED none: the session
  locks the strategy's entries and alerts (`_session_gtts`), never an exit.
- **Findings to kernel calls.** TRIGGERED and CHILD_MISSED `apply_gtt_triggered`, whose
  child order the session binds (`_session_gtts`); EXPIRED and REJECTED
  `apply_gtt_ended`; ENDED_OUTSIDE `apply_gtt_cancelled` as not asked for, after which
  the kernel places the stop again or exits flagged `stop_crossed`; IN_FLIGHT and
  RESTING none; MISMATCH and OTHER_TERMS none: the session locks and alerts.

- **A restart** (`take_back`): each GTT the kernel restores is taken back with the
  journal's GTT and the broker's trigger ID it is bound to (`_session_restore_gtts`); one
  whose state the restart cannot tell is unsettled: the kernel's later commands about it
  are not sent and its findings apply nothing (the session has locked the strategy's
  entries and alerted).

A result call the kernel refuses (`VenueRefusalError`) is reported in the result
(`GttHandled.refused`), never swallowed: the session alerts and locks. Quantities are
whole units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import LiveError, VenueRefusalError
from ._gtt_types import (
    BindGttChild,
    Direction,
    GttAction,
    GttAnswer,
    GttAnswerKind,
    GttEvent,
    GttOutcome,
    GttRequest,
)
from ._journal_gtt_rows import GttTerms
from ._types import LOGGER_NAME, Now, Purpose
from ._venue_requests import Priority

logger = logging.getLogger(LOGGER_NAME)

#: The kernel's GTT commands (the binding's `gtts` module).
GTT_COMMANDS = frozenset({"place_gtt", "modify_gtt", "cancel_gtt"})
#: How a GTT that ended at the venue is named to `apply_gtt_ended` (the binding's
#: `GTT_ENDS`).
_ENDS = {GttEvent.EXPIRED: "expired", GttEvent.REJECTED: "rejected"}
_ACTIONS = {"place_gtt": GttAction.PLACE, "modify_gtt": GttAction.MODIFY}
#: The kernel call of each answer that is one call alone.
_CALLS = {
    GttAnswerKind.MODIFY_REFUSED: "apply_gtt_modify_refused",
    GttAnswerKind.CANCELLED: "apply_gtt_cancelled",
    GttAnswerKind.CANCEL_REFUSED: "apply_gtt_cancel_refused",
}


class GttApplied(enum.Enum):
    """What one GTT answer or finding did to the kernel (module docstring)."""

    NOTHING = "nothing"
    PLACED = "placed"
    REFUSED = "refused"
    MODIFIED = "modified"
    MODIFY_REFUSED = "modify_refused"
    CANCELLED = "cancelled"
    CANCEL_REFUSED = "cancel_refused"
    TRIGGERED = "triggered"
    CHILD_MISSED = "child_missed"
    ENDED = "ended"
    ENDED_OUTSIDE = "ended_outside"
    #: No kernel call: the session locks the strategy's entries and alerts.
    UNRESOLVED = "unresolved"


#: What each answer of `_CALLS` applies.
_APPLIED = {
    GttAnswerKind.MODIFY_REFUSED: GttApplied.MODIFY_REFUSED,
    GttAnswerKind.CANCELLED: GttApplied.CANCELLED,
    GttAnswerKind.CANCEL_REFUSED: GttApplied.CANCEL_REFUSED,
}


@dataclass(frozen=True)
class GttHandled:
    """What a GTT answer or finding did (`applied`), the requests it released for the
    order worker (modifies and cancels that waited for a place's answer, a child's
    binding), the kernel order of a fired GTT's child (`child`), and the words of a
    result call the kernel refused (`refused`, "" for none)."""

    applied: GttApplied
    requests: tuple[GttRequest | BindGttChild, ...] = ()
    child: int | None = None
    refused: str = ""


@dataclass
class KernelGtt:
    """What the session knows of one of the kernel's GTTs: what it is for, the journal's
    GTT and the broker's trigger ID once a place was answered, the commands waiting for
    them, the terms last asked for, since when (UTC ns) a place handed to the order worker
    waits for its answer (None before the hand-over and once answered), the kernel order
    of its child once it fired, whether a cancel of it was handed to the order worker and
    not answered (`cancel_sent`), whether a restart could not tell where it stands
    (`unsettled`, module docstring), and why the close's read of the GTT book did not
    confirm it, once the kernel's session had closed and could hear of it no more
    (`unconfirmed`, `_session_carry`; "" for none)."""

    purpose: Purpose
    asked: GttTerms
    seq: int | None = None
    trigger_id: str = ""
    deferred: list[Mapping[str, Any]] = field(default_factory=list)
    placing_ns: int | None = None
    child: int | None = None
    cancel_sent: bool = False
    unsettled: bool = False
    unconfirmed: str = ""


class GttMap:
    """The GTT map of `engine` (an `Engine.session`), the session of strategy `strategy`
    (module docstring): `markets` are its instruments' (symbol, exchange) in registration
    order, `product` the strategy's, `request_ids` the session's request numbers (shared
    with every venue map), `now` gives UTC nanoseconds."""

    def __init__(
        self,
        engine: Any,
        *,
        strategy: str,
        markets: Sequence[tuple[str, str]],
        product: str,
        request_ids: Iterator[int],
        now: Now,
    ) -> None:
        self._engine = engine
        self._strategy = strategy
        self._markets = tuple(markets)
        self._product = product
        self._request_ids = request_ids
        self._now = now
        self._gtts: dict[int, KernelGtt] = {}

    def gtt(self, gtt_id: int) -> KernelGtt | None:
        """What the session knows of kernel GTT `gtt_id`, or None."""
        return self._gtts.get(gtt_id)

    def handed(self, request: GttRequest) -> None:
        """`request` was handed to the order worker: a cancel's GTT may be cancelled at the
        broker from now (`KernelGtt.cancel_sent`), and a place waits for its answer from
        now (`KernelGtt.placing_ns`; one the exit guard holds has not been handed over)."""
        known = self._gtts.get(request.gtt_id)
        if known is None:
            return
        if request.action is GttAction.CANCEL:
            known.cancel_sent = True
        elif request.action is GttAction.PLACE:
            known.placing_ns = self._now()

    def gtt_of_child(self, kernel_order: int) -> KernelGtt | None:
        """The GTT whose child the kernel follows as `kernel_order`, or None."""
        return next((known for known in self._gtts.values() if known.child == kernel_order), None)

    def placing(self) -> list[tuple[int, int]]:
        """(kernel GTT, since when its place waits, UTC ns) of each leg's stop GTT whose
        place was handed over and not answered yet, in GTT order."""
        return sorted(
            (gtt_id, known.placing_ns)
            for gtt_id, known in self._gtts.items()
            if known.placing_ns is not None and known.purpose is Purpose.STOP
        )

    def take_back(
        self, view: Mapping[str, Any], *, seq: int | None, trigger_id: str, unsettled: bool
    ) -> None:
        """Take back at a restart the kernel's GTT `view` (an `Engine.gtt` dict): the
        journal's GTT `seq` bound to the broker's `trigger_id`, or `unsettled` (module
        docstring)."""
        purpose = Purpose.STOP if view["purpose"] == "stop" else Purpose.ENTRY
        known = KernelGtt(purpose, _terms(view), seq=seq, trigger_id=trigger_id)
        known.unsettled = unsettled
        self._gtts[int(view["gtt_id"])] = known

    # ----- commands to requests --------------------------------------------------------

    def requests(self, command: Mapping[str, Any]) -> list[GttRequest]:
        """The order worker's requests for one of the kernel's GTT commands: none for a
        modify or cancel that waits for its place's answer, nor for a GTT a restart left
        unsettled (module docstring). Raises LiveError for a command this map cannot
        send."""
        kind, gtt_id = command.get("command"), int(command["gtt_id"])
        unsettled = self._gtts.get(gtt_id)
        if unsettled is not None and unsettled.unsettled:
            logger.warning(
                "%s: kernel GTT g%d was left unsettled by the restart: its %s is not sent",
                self._strategy,
                gtt_id,
                kind,
            )
            return []
        if kind == "place_gtt":
            if gtt_id in self._gtts:
                raise LiveError(f"kernel GTT g{gtt_id} of {self._strategy} is placed twice")
            purpose = Purpose.STOP if command.get("order_id") is None else Purpose.ENTRY
            self._gtts[gtt_id] = KernelGtt(purpose, _terms(command))
            return [self._request(command, None)]
        known = self._gtts.get(gtt_id)
        if known is None or kind not in ("modify_gtt", "cancel_gtt"):
            raise LiveError(f"the kernel handed a GTT command this map cannot send: {command!r}")
        if known.seq is None:
            known.deferred.append(command)
            return []
        return [self._request(command, known.seq)]

    def _request(self, command: Mapping[str, Any], seq: int | None) -> GttRequest:
        gtt_id = int(command["gtt_id"])
        known = self._gtts[gtt_id]
        terms = _terms(command)
        action = _ACTIONS.get(str(command["command"]), GttAction.CANCEL)
        symbol, exchange = self._markets[int(command["instrument"])]
        request = GttRequest(
            request_id=next(self._request_ids),
            action=action,
            purpose=known.purpose,
            strategy=self._strategy,
            gtt_id=gtt_id,
            symbol=symbol,
            exchange=exchange,
            product=str(command["product"]),
            side=terms.side,
            trigger_price=terms.trigger_price,
            limit_price=terms.limit_price,
            quantity=terms.quantity,
            direction=Direction.BELOW if command["trigger_below"] else Direction.ABOVE,
            priority=_priority(known, action, terms),
            seq=seq,
        )
        if action is not GttAction.CANCEL:
            known.asked = terms
        return request

    # ----- answers and findings --------------------------------------------------------

    def answered(self, answer: GttAnswer) -> GttHandled:
        """Apply the order worker's answer to one of this map's GTT requests (module
        docstring)."""
        request = answer.request
        known = self._gtts.get(request.gtt_id)
        if known is None:
            raise LiveError(f"GTT answer of g{request.gtt_id}, which {self._strategy} never placed")
        kind, gtt_id = answer.kind, request.gtt_id
        if request.action is GttAction.CANCEL:
            known.cancel_sent = False
        if kind is GttAnswerKind.UNRESOLVED:
            known.placing_ns = None
            return GttHandled(GttApplied.UNRESOLVED)
        if kind in (GttAnswerKind.PLACED, GttAnswerKind.MODIFIED):
            known.seq, known.trigger_id, known.placing_ns = answer.seq, answer.trigger_id, None
            placed = kind is GttAnswerKind.PLACED
            call = self._engine.apply_gtt_placed if placed else self._engine.apply_gtt_modified
            applied = GttApplied.PLACED if placed else GttApplied.MODIFIED
            return GttHandled(applied, self._released(known), refused=self._call(call, gtt_id))
        if kind is GttAnswerKind.REFUSED:
            known.placing_ns, known.deferred = None, []
            if request.action is GttAction.PLACE:
                return GttHandled(GttApplied.REFUSED, refused=self._call(self._refused, gtt_id))
            refused = self._call(self._engine.apply_gtt_modify_refused, gtt_id)
            refused = refused or self._call(self._engine.apply_gtt_cancelled, gtt_id)
            return GttHandled(GttApplied.REFUSED, refused=refused)
        applied = _APPLIED[kind]
        call = getattr(self._engine, _CALLS[kind])
        return GttHandled(applied, refused=self._call(call, gtt_id))

    def found(self, outcome: GttOutcome) -> GttHandled:
        """Apply a finding of the GTT follower about one of this map's GTTs (module
        docstring). A finding about a GTT this session never placed nor took back (one of
        an earlier day the follower follows from the journal: unit N5 restores those), or
        one a restart left unsettled, changes nothing in the kernel."""
        known = self._gtts.get(outcome.gtt_id)
        event, gtt_id = outcome.event, outcome.gtt_id
        if known is None or known.unsettled:
            logger.warning(
                "%s: GTT %s (%s) is not one this session placed: nothing applied",
                self._strategy,
                outcome.trigger_id,
                event.value,
            )
            return GttHandled(GttApplied.NOTHING)
        if event in (GttEvent.TRIGGERED, GttEvent.CHILD_MISSED):
            return self._triggered(known, outcome)
        if event in _ENDS:
            refused = self._call(self._engine.apply_gtt_ended, gtt_id, _ENDS[event])
            return GttHandled(GttApplied.ENDED, refused=refused)
        if event is GttEvent.ENDED_OUTSIDE:
            refused = self._call(self._engine.apply_gtt_cancelled, gtt_id)
            return GttHandled(GttApplied.ENDED_OUTSIDE, refused=refused)
        if event in (GttEvent.MISMATCH, GttEvent.OTHER_TERMS):
            return GttHandled(GttApplied.UNRESOLVED)
        return GttHandled(GttApplied.NOTHING)

    def _triggered(self, known: KernelGtt, outcome: GttOutcome) -> GttHandled:
        """The kernel follows the fired GTT's child as an order of its own; a child found
        is bound in the journal by the order worker (`BindGttChild`)."""
        try:
            child = int(self._engine.apply_gtt_triggered(outcome.gtt_id, self._now()))
        except VenueRefusalError as error:
            return GttHandled(GttApplied.NOTHING, refused=_words(outcome.gtt_id, error))
        known.child = child
        if outcome.event is GttEvent.CHILD_MISSED:
            return GttHandled(GttApplied.CHILD_MISSED, child=child)
        bind = BindGttChild(outcome.seq, outcome.order_id, child)
        return GttHandled(GttApplied.TRIGGERED, (bind,), child=child)

    def _refused(self, gtt_id: int) -> None:
        self._engine.apply_gtt_refused(gtt_id)

    def _released(self, known: KernelGtt) -> tuple[GttRequest, ...]:
        deferred, known.deferred = known.deferred, []
        return tuple(self._request(command, known.seq) for command in deferred)

    def _call(self, call: Callable[..., Any], gtt_id: int, *args: Any) -> str:
        """Make a result call on kernel GTT `gtt_id`: "" when the kernel took it, else
        the words of its refusal (module docstring)."""
        try:
            call(gtt_id, *args)
        except VenueRefusalError as error:
            return _words(gtt_id, error)
        return ""


def _words(gtt_id: int, error: VenueRefusalError) -> str:
    words = f"the engine refused a result of GTT g{gtt_id} ({error.kind}): {error}"
    logger.error(words)
    return words


def _terms(command: Mapping[str, Any]) -> GttTerms:
    """The terms of a GTT command. Raises LiveError for units that are not whole."""
    units = float(command["quantity_units"])
    if not units.is_integer():
        raise LiveError(f"kernel GTT g{command['gtt_id']} is for {units} units: whole units only")
    return GttTerms(
        str(command["side"]),
        float(command["trigger_price"]),
        float(command["price"]),
        int(units),
    )


def _priority(known: KernelGtt, action: GttAction, terms: GttTerms) -> Priority:
    """A strategy's GTT order is ADJUST (its place ENTER); a leg's stop PROTECT, but a
    trailing modify (the same units) ADJUST."""
    if known.purpose is not Purpose.STOP:
        return Priority.ENTER if action is GttAction.PLACE else Priority.ADJUST
    trailing = action is GttAction.MODIFY and terms.quantity == known.asked.quantity
    return Priority.ADJUST if trailing else Priority.PROTECT


__all__ = ["GTT_COMMANDS", "GttApplied", "GttHandled", "GttMap", "KernelGtt"]
