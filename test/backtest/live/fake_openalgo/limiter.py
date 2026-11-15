"""The server's rate limits, kept on the fake's manual clock.

Every REST route has its own limit, counted per caller in a moving window (limiter.py on
the server: `strategy="moving-window"`, keyed by the remote address; each route's
`@limiter.limit`): 10 calls per second for the order and data routes (ORDER_RATE_LIMIT
and API_RATE_LIMIT, restx_api/place_order.py and the others) and 30 per minute for the
Telegram routes (TELEGRAM_RATE_LIMIT, restx_api/telegram_bot.py). A call is refused when
the limit-th latest accepted call of its route is no older than the window; a refused
call is not counted (the limiter's memory storage, `acquire_entry`). Times are UTC
nanoseconds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from openalgo.backtest._layout import NANOS_PER_SECOND


@dataclass(frozen=True)
class Limit:
    """`calls` per `span` of `unit` ("second" or "minute"), as the limiter states it."""

    calls: int
    span: int
    unit: str

    @property
    def window_ns(self) -> int:
        """The window, UTC nanoseconds."""
        return self.span * _UNIT_SECONDS[self.unit] * NANOS_PER_SECOND

    @property
    def words(self) -> str:
        """The limiter's description of the limit: the body of its 429."""
        return f"{self.calls} per {self.span} {self.unit}"


_UNIT_SECONDS = {"second": 1, "minute": 60}
#: The limit of every route not in `LIMITS`.
DEFAULT = Limit(10, 1, "second")
LIMITS = {"telegram/notify": Limit(30, 1, "minute")}


def limit_of(endpoint: str) -> Limit:
    """The rate limit of a route."""
    return LIMITS.get(endpoint, DEFAULT)


class RateLimiter:
    """The routes' moving windows of accepted calls."""

    def __init__(self) -> None:
        self._accepted: dict[str, deque[int]] = {}

    def allow(self, endpoint: str, now_ns: int) -> bool:
        """Count a call to `endpoint` at `now_ns` and say whether the limit lets it
        through (a refused call is not counted)."""
        limit = limit_of(endpoint)
        accepted = self._accepted.setdefault(endpoint, deque())
        while accepted and accepted[0] < now_ns - limit.window_ns:
            accepted.popleft()
        if len(accepted) >= limit.calls:
            return False
        accepted.append(now_ns)
        return True


__all__ = ["DEFAULT", "LIMITS", "Limit", "RateLimiter", "limit_of"]
