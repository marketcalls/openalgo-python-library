"""The order budget of the engine's process, for the order path (`orders`): place,
modify and cancel share OpenAlgo's per-IP order limit with the user's own manual use
(08-live-trading section 8). The order worker takes an order request only while the budget
has room (`_order_throttle`, 18-multi-strategy 2.5a rule 12). Times are UTC nanoseconds."""

from __future__ import annotations

import collections

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._types import Now

#: OpenAlgo's order budget per client IP (`ORDER_RATE_LIMIT`, "10 per second", in the
#: server's .sample.env and restx_api/place_order.py); the engine counts place, modify and
#: cancel together, as the server does.
OPENALGO_ORDERS_PER_SECOND = 10
#: Order calls a second left to the trader's own orders on the same server.
MANUAL_ORDERS_PER_SECOND = 2
#: The engine's budget unless `bt.live(max_orders_per_second=...)` sets another: OpenAlgo's
#: less the trader's own (18-multi-strategy 2.5a rule 12).
DEFAULT_ORDERS_PER_SECOND = OPENALGO_ORDERS_PER_SECOND - MANUAL_ORDERS_PER_SECOND


class RateBudget:
    """The order budget of the engine's process: at most `per_second` place, modify and
    cancel requests in any one second (a whole number, at least 1;
    `DEFAULT_ORDERS_PER_SECOND` by default).

    `now` gives UTC nanoseconds. Raises LiveError for a `per_second` that is not a whole
    number of at least 1.
    """

    def __init__(self, now: Now, *, per_second: int = DEFAULT_ORDERS_PER_SECOND) -> None:
        if isinstance(per_second, bool) or not isinstance(per_second, int) or per_second < 1:
            raise LiveError(
                f"per_second is {per_second!r}: the engine's order budget is a whole number "
                f"of order calls a second, at least 1 ({DEFAULT_ORDERS_PER_SECOND} by "
                f"default: OpenAlgo's {OPENALGO_ORDERS_PER_SECOND} less "
                f"{MANUAL_ORDERS_PER_SECOND} for the trader's own orders)"
            )
        self._now = now
        self.per_second = per_second
        self._taken_ns: collections.deque[int] = collections.deque()

    def wait_seconds(self) -> float:
        """Seconds until a request may be sent; 0.0 when one may be sent now."""
        now = self._now()
        while self._taken_ns and now - self._taken_ns[0] >= NANOS_PER_SECOND:
            self._taken_ns.popleft()
        if len(self._taken_ns) < self.per_second:
            return 0.0
        return (self._taken_ns[0] + NANOS_PER_SECOND - now) / NANOS_PER_SECOND

    def take(self) -> None:
        """Count a request sent now. Raises LiveError when the budget is spent: wait
        `wait_seconds()` first."""
        wait = self.wait_seconds()
        if wait > 0:
            raise LiveError(f"the order rate budget is spent for {wait:.3f} s more")
        self._taken_ns.append(self._now())


__all__ = [
    "DEFAULT_ORDERS_PER_SECOND",
    "MANUAL_ORDERS_PER_SECOND",
    "OPENALGO_ORDERS_PER_SECOND",
    "RateBudget",
]
