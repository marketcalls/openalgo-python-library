"""The order budget of the engine's process, for the order path (`orders`): place,
modify and cancel share OpenAlgo's per-IP order limit with the user's own manual use
(08-live-trading section 8). Times are UTC nanoseconds."""

from __future__ import annotations

import collections

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._types import Now

#: OpenAlgo's order budget per client IP (`ORDER_RATE_LIMIT`, "10 per second", in the
#: server's restx_api/place_order.py); the engine counts place, modify and cancel together.
OPENALGO_ORDERS_PER_SECOND = 10
#: The engine uses at most half of it, leaving room for manual use (08 section 8).
MAX_ORDERS_PER_SECOND = OPENALGO_ORDERS_PER_SECOND // 2


class RateBudget:
    """The order budget of the engine's process: at most `per_second` (1 to 5) place,
    modify and cancel requests in any one second, half of OpenAlgo's 10.

    `now` gives UTC nanoseconds. Raises LiveError for a `per_second` outside 1 to 5.
    """

    def __init__(self, now: Now, *, per_second: int = MAX_ORDERS_PER_SECOND) -> None:
        if not 1 <= per_second <= MAX_ORDERS_PER_SECOND:
            raise LiveError(
                f"per_second is {per_second!r}: the engine uses 1 to {MAX_ORDERS_PER_SECOND} "
                f"orders a second, at most half of OpenAlgo's {OPENALGO_ORDERS_PER_SECOND}"
            )
        self._now = now
        self._per_second = per_second
        self._taken_ns: collections.deque[int] = collections.deque()

    def wait_seconds(self) -> float:
        """Seconds until a request may be sent; 0.0 when one may be sent now."""
        now = self._now()
        while self._taken_ns and now - self._taken_ns[0] >= NANOS_PER_SECOND:
            self._taken_ns.popleft()
        if len(self._taken_ns) < self._per_second:
            return 0.0
        return (self._taken_ns[0] + NANOS_PER_SECOND - now) / NANOS_PER_SECOND

    def take(self) -> None:
        """Count a request sent now. Raises LiveError when the budget is spent: wait
        `wait_seconds()` first."""
        wait = self.wait_seconds()
        if wait > 0:
            raise LiveError(f"the order rate budget is spent for {wait:.3f} s more")
        self._taken_ns.append(self._now())


__all__ = ["MAX_ORDERS_PER_SECOND", "OPENALGO_ORDERS_PER_SECOND", "RateBudget"]
