"""The venue map's requests the session withdraws before they are sent (`Withdrawals`, a part
of `venue_map.VenueMap`).

- **An entry or add the kernel cancelled while it waits for the worker** (18-multi-strategy
  2.5a rules 5, 6 and 12). The order rate throttle can hold an entry or add in the order
  worker's queue for seconds; meanwhile the kernel may cancel it (its daily profit target
  or daily loss on a tick, the kill switch, the safe stop). A cancel of an order not bound
  yet waits for the bind (2.5 E), so the placement would still be sent, and fill, after
  the strategy's entries locked. So the cancel of an entry or add whose placement has had
  no answer yet also asks the worker to withdraw that placement (`withdrawal`, a
  `WithdrawPlace`): while it still waits the worker answers `NotSent(cancelled)`, and the
  kernel is told it was cancelled (`cancelled_place`, `apply_cancel`), as for a placement
  cancelled in the batch that made it (`_session_trader`). Once the worker has taken it,
  nothing changes: the cancel goes when the placement binds. A stop or an exit is never
  withdrawn this way: the throttle takes them first, and a closing order is never dropped.
- **An entry or add waiting when the strategy's entries lock** (2.4.0 item M8-X2). The
  kernel cancels a working entry for a daily profit target, the kill switch or the safe
  stop, but not for its other locks (a dead feed, a reconciliation, failed exits), which
  stop entries at submission only. An entry decided before such a lock, and held back by
  the order rate, would still go after it. So once the strategy's entries are locked,
  each placement of an entry or add handed over with no answer yet is withdrawn too
  (`withdraw_locked`, which `_session_trader.Trader.commands` adds to the commands of
  each loop step): while it still waits the worker answers
  `NotSent(entries_locked)`, and the kernel takes it as an entry refused under the lock
  (`locked_place`, `apply_reject` with `live_lock`, the code a submission gets then).
  Once the worker has taken it, it goes as sent before the lock.
- **A cancel the exit guard withdraws** (16-session-design 3.3 step 5 "As built", round
  7): never sent, the kernel takes it as refused (`withdraw_cancel`); held again, it
  waits for its answer again (`restore_cancel`).

Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .._engine import call
from ._order_state import ENTRY_PURPOSES
from ._session_unsent import LOCKED_CODE
from ._venue_requests import Applied, CancelRequest, Handled, OrderRequest, PlaceRequest
from ._venue_saved import VenueOrder
from ._worker_queue import WithdrawPlace
from .inbox import NotSentReason


class Withdrawals:
    """The venue map's withdrawals (module docstring). Mixed into `venue_map.VenueMap`,
    whose state it reads (the engine, its orders by kernel order, the requests waiting for
    an answer) and whose `_order` and `_answer` it calls."""

    _engine: Any
    _orders: dict[int, VenueOrder]
    _issued: dict[int, OrderRequest]
    #: The requests whose withdrawal for the strategy's entry locks was asked already.
    _withdrawing: set[int]
    _order: Callable[[int], VenueOrder]
    _answer: Callable[..., bool]

    def withdrawal(self, command: Mapping[str, Any]) -> list[WithdrawPlace]:
        """The worker's withdrawal of the placement of kernel `command`'s order, an order
        not bound yet: one `WithdrawPlace` when `command` is a cancel and the order is an
        entry or add whose placement has had no answer; none otherwise."""
        if command.get("command") != "cancel":
            return []
        kernel_order = int(command["order_id"])
        return [
            WithdrawPlace(request)
            for request in self._issued.values()
            if isinstance(request, PlaceRequest)
            and request.kernel_order == kernel_order
            and request.intent.purpose in ENTRY_PURPOSES
        ]

    def withdraw_locked(self) -> list[WithdrawPlace]:
        """The worker's withdrawal of each placement of an entry or add handed over with no
        answer yet, once the strategy's entries are locked (module docstring), each asked
        once; none while no lock is set. The kernel's locks are read only while such a
        placement waits: most loop steps have none."""
        waiting = self._waiting_entries()
        locks = sorted(call(self._engine.entry_locks)) if waiting else []
        if not locks:
            return []
        words = f"the strategy's entries were locked ({', '.join(locks)}) before it was sent"
        for request in waiting:
            self._withdrawing.add(request.request_id)
        return [WithdrawPlace(r, NotSentReason.ENTRIES_LOCKED, words) for r in waiting]

    def locked_place(self, kernel_order: int) -> Handled:
        """The worker withdrew the placement of `kernel_order` for the strategy's entry
        locks (`NotSent(entries_locked)`): it never reached the broker, so the kernel takes
        it as an entry refused under the lock (`apply_reject` with `live_lock`) and the
        commands that waited for the bind are dropped."""
        order = self._order(kernel_order)
        order.ended, order.deferred = True, []
        if self._answer(self._engine.apply_reject, kernel_order, LOCKED_CODE):
            return Handled(Applied.REJECTED)
        return Handled(Applied.NOTHING)

    def _waiting_entries(self) -> list[PlaceRequest]:
        return [
            request
            for request in self._issued.values()
            if isinstance(request, PlaceRequest)
            and request.intent.purpose in ENTRY_PURPOSES
            and request.request_id not in self._withdrawing
        ]

    def cancelled_place(self, kernel_order: int) -> Handled:
        """The worker withdrew the placement of `kernel_order` (`NotSent(cancelled)`): it
        never reached the broker, so the kernel's cancel is confirmed (`apply_cancel`) and
        the commands that waited for the bind are dropped."""
        order = self._order(kernel_order)
        order.ended, order.deferred = True, []
        if self._answer(self._engine.apply_cancel, kernel_order):
            return Handled(Applied.CANCELLED)
        return Handled(Applied.NOTHING)

    def withdraw_cancel(self, request: CancelRequest) -> bool:
        """A cancel this map made that the session will never send (the exit guard held it
        and withdraws it: 16-session-design 3.3 step 5 "As built", round 7): it waits for
        no answer, and the kernel takes it as refused (`apply_cancel_refused`), so the
        order works on and the kernel may fit it to the position. True when the kernel
        took it. Raises LiveError for an order this map did not place."""
        self._issued.pop(request.request_id, None)
        order = self._order(request.kernel_order)
        if order.cancel is not None and order.cancel.request_id == request.request_id:
            order.cancel = None
        return self._answer(self._engine.apply_cancel_refused, request.kernel_order)

    def restore_cancel(self, request: CancelRequest) -> None:
        """A cancel `withdraw_cancel` withdrew that the session holds again (the exit
        guard puts it back when the adoption it made way for is refused): it waits for
        its answer again, should it ever be sent. The kernel took it as refused, so the
        order works on there as at the broker, where it was never sent; a later refusal of
        it is the "not_pending" race. Raises LiveError for an order this map did not
        place."""
        order = self._order(request.kernel_order)
        self._issued[request.request_id] = request
        if order.cancel is None:
            order.cancel = request


__all__ = ["Withdrawals"]
