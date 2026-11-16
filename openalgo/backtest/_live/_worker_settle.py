"""Settling an order whose modify or cancel went unanswered or was refused, for the order
worker (16-session-design 2.4 step 7, 4.3, 4.5).

From such an answer (or a cancel the worker could not send) until a targeted read
settles it, the order is unsettled: the worker takes none of its further requests. It
reads the order's row with `OrderPath.check_order` (exempt from the reconciler's
in-flight rule) on the back-off `ORDER_CHECK_DELAYS_S` from the answer, and the row's
verdict (`_venue_requests.modify_verdict`, `cancel_verdict`, the one rule the venue map
applies too) settles it: a modify by its terms or the old terms, a cancel by a terminal
or a working row, either by a terminal row, whose news then decides. A row that settles
nothing is read again. A modify not settled within `MODIFY_SETTLE_DEADLINE_S` is
settled expired (`inbox.Settled` with no check), which the loop takes as refused, with an
ERROR alert: the kernel then sends the wanted terms again, and a broker that had applied
them accepts the same terms again. A cancel is read until a row settles it, since the
exit owner waits and the stop protects meanwhile. The order's terminal news settles it
too, before any read.

Times are UTC nanoseconds. Used by the worker's thread only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._order_check import OrderCheck
from ._order_state import BACKOFF_NS
from ._venue_requests import CancelRequest, ModifyRequest, Verdict, cancel_verdict, modify_verdict
from .inbox import Settled

#: Seconds between the reads of an unsettled order, the last repeated: the order path's
#: back-off (`BACKOFF_NS`), so a settling read keeps the pace of the polls.
ORDER_CHECK_DELAYS_S = tuple(step / NANOS_PER_SECOND for step in BACKOFF_NS)
#: Seconds from a modify's answer within which a row must settle it (4.3).
MODIFY_SETTLE_DEADLINE_S = 10.0
_DEADLINE_NS = round(MODIFY_SETTLE_DEADLINE_S * NANOS_PER_SECOND)

#: A modify or a cancel of one broker order.
Change = ModifyRequest | CancelRequest


@dataclass
class Unsettled:
    """An order's modify or cancel waiting for a row: unsettled since `since_ns`, read
    `reads` times, next due at `due_ns`."""

    request: Change
    since_ns: int
    due_ns: int
    reads: int = 0


class Settlements:
    """The worker's unsettled orders, by broker order ID (module docstring)."""

    def __init__(self) -> None:
        self._waiting: dict[str, Unsettled] = {}

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._waiting

    def add(self, request: Change, since_ns: int) -> None:
        """`request` was unanswered or refused at `since_ns`: its order waits for a row."""
        first_ns = since_ns + BACKOFF_NS[0]
        self._waiting[request.order_id] = Unsettled(request, since_ns, first_ns)

    def ended(self, order_id: str) -> Unsettled | None:
        """The order's terminal news settles it: the one taken off, or None."""
        return self._waiting.pop(order_id, None)

    def expired(self, now_ns: int) -> list[Unsettled]:
        """The modifies past their deadline at `now_ns`, taken off, oldest first."""
        late = [
            waiting
            for waiting in self._waiting.values()
            if isinstance(waiting.request, ModifyRequest)
            and now_ns - waiting.since_ns >= _DEADLINE_NS
        ]
        for waiting in late:
            del self._waiting[waiting.request.order_id]
        return sorted(late, key=lambda waiting: waiting.since_ns)

    def due(self, now_ns: int) -> list[Unsettled]:
        """The orders whose next read is due at `now_ns`, oldest first."""
        due = [waiting for waiting in self._waiting.values() if waiting.due_ns <= now_ns]
        return sorted(due, key=lambda waiting: waiting.due_ns)

    def judge(self, waiting: Unsettled, check: OrderCheck, now_ns: int) -> Settled | None:
        """The `Settled` message when `check` (read at `now_ns`) settles `waiting`, which is
        then taken off; None when it settles nothing, and the next read is scheduled."""
        request = waiting.request
        if isinstance(request, ModifyRequest):
            verdict = modify_verdict(request, check)
        else:
            verdict = cancel_verdict(check)
        if verdict is not Verdict.UNSETTLED:
            self._waiting.pop(request.order_id, None)
            return Settled(request, check)
        waiting.reads += 1
        waiting.due_ns = now_ns + BACKOFF_NS[min(waiting.reads, len(BACKOFF_NS) - 1)]
        return None


def expired_text(waiting: Unsettled, now_ns: int) -> str:
    """The ERROR alert of a modify no row settled in time."""
    request = waiting.request
    seconds = (now_ns - waiting.since_ns) // NANOS_PER_SECOND
    return (
        f"The modify of order {request.order_id} ({request.intent.symbol} "
        f"{request.intent.exchange}, {request.terms.quantity} units {request.terms.price_type}"
        f" at trigger {request.terms.trigger_price}) has no answer after {seconds} s and its "
        "row cannot be read: the engine keeps the old terms and sends the new ones again. "
        "Check the order in OpenAlgo."
    )


__all__ = [
    "MODIFY_SETTLE_DEADLINE_S",
    "ORDER_CHECK_DELAYS_S",
    "Change",
    "Settlements",
    "Unsettled",
    "expired_text",
]
