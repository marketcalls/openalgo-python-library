"""The live order rate throttle (18-multi-strategy 2.5a rule 12): at most
`max_orders_per_second` order calls (`placeorder`, `modifyorder` and `cancelorder`
together) in any one-second window of the session. Always on.

- **The window** is the order path's `RateBudget` (`_order_rate`): the order path counts
  each order call it makes, and the order worker (`worker`) takes an order request off its
  queue only while the window has room (`OrderThrottle.take`), so the order path never
  waits for it. Requests that make no order call (book reads, order reads, probes, the
  release of a held fill) are never held.
- **Waiting.** A request the full window holds back waits in the worker's queue, which
  takes the most urgent first (16-session-design 2.4): stops, exits, protective modifies
  and cancels (priority 0), then trailing modifies and the cancels of entries (1), then
  entries and adds (2).
- **Held too long.** A request held back `NOTICE_AFTER_S` is noted once
  (`inbox.Throttled`; the loop writes the event `order_throttled` and alerts once a
  session, `throttled_news`), a strategy's order by its strategy and purpose, a
  reconciler's cancel of an orphan by its broker order ID. An entry or add held back
  `WITHDRAW_AFTER_S` is not sent: it is withdrawn as `NotSent` with the reason
  `order_rate`, which the loop gives the kernel as a rejection, so it no longer counts for
  the trade caps (`engine::daily_caps`) as an entry that ended with no fill. An exit, a
  stop, a modify or a cancel is never dropped. The waits are measured from the first time
  the full window held the request back; each held request's record keeps the request
  itself, so no later request is taken for it.

`DEFAULT_ORDERS_PER_SECOND` is OpenAlgo's `ORDER_RATE_LIMIT` of 10 a second less 2 for the
trader's own orders on the same server (`_order_rate`); a server set higher takes a higher
figure through `bt.live(max_orders_per_second=...)`. Times are UTC nanoseconds. Used by the
worker's thread only, but for `throttled_news`, which the loop runs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._order_rate import DEFAULT_ORDERS_PER_SECOND, RateBudget
from ._order_state import ENTRY_PURPOSES
from ._types import LOGGER_NAME, AlertKind, Now
from ._venue_requests import PlaceRequest, Priority
from ._worker_queue import CancelOrphan, Request, WorkerQueue
from .inbox import Message, NotSent, NotSentReason, Throttled

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a request may wait for the order rate before it is noted (2.5a rule 12).
NOTICE_AFTER_S = 2.0
#: Seconds an entry or add may wait for the order rate before it is not sent (rule 12).
WITHDRAW_AFTER_S = 5.0
#: The event the loop writes for a request held back (rule 12).
THROTTLED_EVENT = "order_throttled"
#: The purpose the event names for a reconciler's cancel of an orphan (no strategy's).
ORPHAN_PURPOSE = "cancel_orphan"
_NOTICE_NS = round(NOTICE_AFTER_S * NANOS_PER_SECOND)
_WITHDRAW_NS = round(WITHDRAW_AFTER_S * NANOS_PER_SECOND)


@dataclass
class _Held:
    """A request the full window holds back: the request (kept, so its `id` names no
    other request while the record lives), since when, and whether it was noted."""

    request: Request
    since_ns: int
    noticed: bool = False


def _kept(held: _Held | None, request: Request) -> _Held | None:
    """`held` when it is `request`'s own record, else None (a record of another request
    under the same `id`)."""
    return held if held is not None and held.request is request else None


def _is_entry(request: Request) -> bool:
    """True for the placement of an entry or an add, the one request ever withdrawn."""
    return isinstance(request, PlaceRequest) and request.intent.purpose in ENTRY_PURPOSES


class OrderThrottle:
    """The throttle of one session's order worker (module docstring): `per_second` order
    calls at most in any second (a whole number, at least 1). `post` puts a message on the
    loop's inbox and `finished` tells the worker a request it was handed is done with (a
    withdrawn one); `now` gives UTC nanoseconds. `budget` is the window, which the worker
    gives its order path.

    Raises LiveError for a `per_second` below 1 (`RateBudget`)."""

    def __init__(
        self,
        now: Now,
        per_second: int = DEFAULT_ORDERS_PER_SECOND,
        *,
        post: Callable[[Message], None],
        finished: Callable[[], None],
    ) -> None:
        self.budget = RateBudget(now, per_second=per_second)
        self._now = now
        self._post = post
        self._finished = finished
        # By id, each order request the full window has held back and is still ready;
        # each record holds its request, and is kept only for that same request.
        self._held: dict[int, _Held] = {}

    def take(self, queue: WorkerQueue, most: Priority) -> Request | None:
        """The next request of `queue` of priority `most` or more urgent (the queue's
        `take`), an order request only while the window has room; first, while it has
        none, the requests held back are noted or withdrawn (module docstring)."""
        now = self._now()
        room = self.budget.wait_seconds() == 0.0
        if not room:
            self._hold(queue, now)
        request = queue.take(now, most=most, orders=room)
        if request is not None:
            self._held.pop(id(request), None)
        return request

    def _hold(self, queue: WorkerQueue, now: int) -> None:
        """Note, or withdraw, each ready order request the full window holds back."""
        ready, before = queue.ready_orders(now), self._held
        self._held = {id(r): _kept(before.get(id(r)), r) or _Held(r, now) for r in ready}
        for request in ready:
            held = self._held[id(request)]
            waited_s = (now - held.since_ns) / NANOS_PER_SECOND
            if _is_entry(request) and now - held.since_ns >= _WITHDRAW_NS:
                self._withdraw(queue, request, waited_s)
            elif now - held.since_ns >= _NOTICE_NS and not held.noticed:
                held.noticed = True
                self._post(Throttled(request, waited_s, withdrawn=False))

    def _withdraw(self, queue: WorkerQueue, request: Request, waited_s: float) -> None:
        queue.remove(request)
        del self._held[id(request)]
        logger.warning("withdrawn after %.1f s waiting for the order rate: %r", waited_s, request)
        self._post(Throttled(request, waited_s, withdrawn=True))
        why = f"waited {waited_s:.1f} s for the order rate"
        self._post(NotSent(request, NotSentReason.THROTTLED, why))
        self._finished()


def throttled_news(session: Any, message: Throttled, *, alerted: bool) -> bool:
    """The loop's part (module docstring): the event `order_throttled` for `message`, and
    an ERROR alert unless the session has `alerted` already; returns whether it has now.
    `session` is a `_session_loop.Session`."""
    request = message.request
    figure = session.parts.spec.arguments.max_orders_per_second
    if isinstance(request, CancelOrphan):
        named = {"strategy": None, "order_id": request.order_id, "purpose": ORPHAN_PURPOSE}
        what = f"the cancel of order {request.order_id}, which no strategy holds,"
    else:
        intent = request.intent
        named = {
            "strategy": intent.strategy,
            "request_id": request.request_id,
            "purpose": intent.purpose.value,
        }
        what = f"{intent.strategy}'s {intent.purpose.value.lower()} order of {intent.symbol}"
    session.event(
        THROTTLED_EVENT,
        **named,
        waited_s=round(message.waited_s, 3),
        reason=NotSentReason.THROTTLED.value if message.withdrawn else None,
    )
    if alerted:
        return True
    fate = "was not sent" if message.withdrawn else "waits"
    session.alert(
        AlertKind.ERROR,
        f"order calls wait for the order rate (at most {figure} a second, "
        f"max_orders_per_second): {what} {fate} after {message.waited_s:.1f} s; exits and "
        f"stops go first, and an entry or add that waits {WITHDRAW_AFTER_S:g} s is not sent",
    )
    return True


__all__ = [
    "DEFAULT_ORDERS_PER_SECOND",
    "NOTICE_AFTER_S",
    "ORPHAN_PURPOSE",
    "THROTTLED_EVENT",
    "WITHDRAW_AFTER_S",
    "OrderThrottle",
    "throttled_news",
]
