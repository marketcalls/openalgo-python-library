"""The health of the WebSocket feed during a live session (08-live-trading section 8).
Part of `bars`, which re-exports it.

Only a dead feed (no tick on any symbol) locks entries; a quiet symbol (an illiquid one,
while the others tick) does not. Times are UTC nanoseconds from the injected clock.
"""

from __future__ import annotations

import math
import numbers

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._types import Now

#: Seconds without a tick on any symbol after which the feed is dead: REST quotes fill
#: in after 10 seconds without ticks (08-live-trading section 2a).
DEFAULT_DEAD_AFTER_S = 10.0


def _positive_seconds(value: float, name: str) -> int:
    """A positive, finite number of seconds as nanoseconds, or ConfigError."""
    usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (usable and math.isfinite(value) and value > 0):
        raise ConfigError(f"{name} must be a positive number of seconds, got {value!r}")
    return round(value * NANOS_PER_SECOND)


def _session_bounds(session: tuple[int, int]) -> tuple[int, int]:
    """(open, close) as ints, or ConfigError unless they are instants with open first."""
    bounds = tuple(session) if isinstance(session, tuple | list) else ()
    usable = len(bounds) == 2 and all(
        isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in bounds
    )
    if not (usable and bounds[0] < bounds[1]):
        raise ConfigError(
            f"session is (open, close) in UTC nanoseconds, open first; got {session!r}"
        )
    return int(bounds[0]), int(bounds[1])


class FeedHealth:
    """The health of the WebSocket feed during a session (08-live-trading section 8).

    "Feed dead" is no tick on any symbol for `dead_after_s` seconds while the session is
    open, and locks entries; "quiet" is one symbol without a tick while the feed lives
    (an illiquid symbol), so the two never hold at once. Silence is counted from the
    later of the last tick, the session open and the moment the health was created, so
    a session that starts late gives the feed its full time. A tick counts at its
    receive time (`Tick.ts_recv_ns`), so a loop busy with other work does not read a
    live feed as dead; the injected clock `now` (UTC nanoseconds) says how long ago that
    was. A session whose symbols are all illiquid cannot tell a quiet market from a dead
    feed: with no symbol ticking, both read as dead.

    `session` is (open, close) of the session, UTC nanoseconds (`Calendar.session`,
    `TickBarBuilder.session`). Raises ConfigError for a `dead_after_s` that is not a
    positive number of seconds or a session whose open is not before its close.
    """

    def __init__(
        self,
        now: Now,
        *,
        session: tuple[int, int],
        dead_after_s: float = DEFAULT_DEAD_AFTER_S,
    ) -> None:
        self._session = _session_bounds(session)
        self._dead_after_ns = _positive_seconds(dead_after_s, "dead_after_s")
        self._now = now
        self._watch_from_ns = max(self._session[0], now())
        self._last_tick_ns: dict[tuple[str, str], int] = {}
        self._last_any_ns: int | None = None

    def on_tick(self, symbol: str, exchange: str, ts_recv_ns: int) -> None:
        """Note a tick of a symbol received at `ts_recv_ns` (UTC nanoseconds). A tick
        handled after a later one of the same symbol leaves the later time standing."""
        key = (symbol, exchange)
        self._last_tick_ns[key] = max(ts_recv_ns, self._last_tick_ns.get(key, ts_recv_ns))
        self._last_any_ns = max(ts_recv_ns, self._last_any_ns or ts_recv_ns)

    def last_tick_ns(self, symbol: str, exchange: str) -> int | None:
        """When the symbol's last tick was received (UTC nanoseconds), None before its
        first."""
        return self._last_tick_ns.get((symbol, exchange))

    def feed_dead(self) -> bool:
        """True when no symbol has ticked for `dead_after_s` while the session is open."""
        return self._dead(self._now())

    def quiet(self, symbol: str, exchange: str, seconds: float) -> bool:
        """True when the symbol has not ticked for `seconds` while the session is open
        and the feed lives. Raises ConfigError unless `seconds` is a positive number."""
        quiet_ns = _positive_seconds(seconds, "seconds")
        now_ns = self._now()
        if not self._open(now_ns) or self._dead(now_ns):
            return False
        return now_ns - self._silent_since(self.last_tick_ns(symbol, exchange)) >= quiet_ns

    def _open(self, now_ns: int) -> bool:
        return self._session[0] <= now_ns < self._session[1]

    def _dead(self, now_ns: int) -> bool:
        silent_ns = now_ns - self._silent_since(self._last_any_ns)
        return self._open(now_ns) and silent_ns >= self._dead_after_ns

    def _silent_since(self, last_tick_ns: int | None) -> int:
        if last_tick_ns is None:
            return self._watch_from_ns
        return max(last_tick_ns, self._watch_from_ns)


__all__ = ["DEFAULT_DEAD_AFTER_S", "FeedHealth"]
