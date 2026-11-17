"""The unprotected windows of a session's legs (16-session-design 4.2): when a guarded
leg's position held more units than its confirmed broker-held stop covered, and why.

A window opens at the receive time of the news that left units uncovered and closes at
the receive time of the word that covers them, or of the leg going flat. A change of
reason closes the window and opens the next at the same instant, so each window has one
reason (`REASONS`). The windows are the result's `unprotected` table, and their seconds
the summary's `unprotected_seconds` (a window still open counts up to the moment asked).

Times are UTC nanoseconds, quantities whole units. Used by the loop only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .._layout import NANOS_PER_SECOND

#: Why a leg is unprotected (the result's `unprotected` reasons).
STOP_UNCONFIRMED = "stop unconfirmed"
STOP_HELD = "stop held"
STOP_ENDED = "stop ended"
EXITING = "exiting"
EXIT_RETRIES = "exit retries"
UNBOUND_ADD = "unbound add"
HELD_CANCEL = "held cancel"
AMBIGUOUS_ORDER = "ambiguous order"
REASONS = (
    STOP_UNCONFIRMED,
    STOP_HELD,
    STOP_ENDED,
    EXITING,
    EXIT_RETRIES,
    UNBOUND_ADD,
    HELD_CANCEL,
    AMBIGUOUS_ORDER,
)


@dataclass(frozen=True)
class Unprotected:
    """A window in which `units` of strategy `strategy`'s position in `exchange:symbol`
    had no confirmed stop at the broker, from `from_ns` to `to_ns` (None while it lasts),
    for `reason` (one of `REASONS`)."""

    strategy: str
    symbol: str
    exchange: str
    from_ns: int
    to_ns: int | None
    units: int
    reason: str

    def seconds(self, now_ns: int) -> float:
        """The window's length in seconds, up to `now_ns` while it lasts."""
        end = now_ns if self.to_ns is None else self.to_ns
        return max(0, end - self.from_ns) / NANOS_PER_SECOND


class Windows:
    """Every leg's unprotected windows (module docstring)."""

    def __init__(self) -> None:
        self._open: dict[str, Unprotected] = {}
        self._closed: list[Unprotected] = []

    def note(
        self, leg: tuple[str, str, str], uncovered: int, reason: str, span: tuple[int, int]
    ) -> Unprotected | None:
        """The leg (strategy, symbol, exchange) has `uncovered` units without a confirmed
        stop for `reason`, after the news received from `span[0]` to `span[1]` (UTC
        nanoseconds): open, close or switch its window. A window opens at the first news
        that may have left units uncovered and closes at the last, the moment the session
        knew them covered (a word received during a long call may be known to be the
        stop's only when the call ends). Returns the window opened now, or None."""
        strategy, symbol, exchange = leg
        first_ns, last_ns = span
        current = self._open.get(strategy)
        if current is not None and uncovered > 0 and current.reason == reason:
            return None
        if current is not None:
            first_ns = max(last_ns, current.from_ns)
            self._closed.append(replace(current, to_ns=first_ns))
            del self._open[strategy]
        if uncovered <= 0:
            return None
        opened = Unprotected(strategy, symbol, exchange, first_ns, None, uncovered, reason)
        self._open[strategy] = opened
        return opened

    def current(self, strategy: str) -> Unprotected | None:
        """The window of `strategy`'s leg still open, or None."""
        return self._open.get(strategy)

    def rows(self) -> tuple[Unprotected, ...]:
        """Every window, closed ones first in the order they closed, then those open."""
        return (*self._closed, *self._open.values())

    def seconds(self, now_ns: int) -> float:
        """The windows' seconds summed, those still open counted up to `now_ns`."""
        return sum(window.seconds(now_ns) for window in self.rows())


__all__ = [
    "AMBIGUOUS_ORDER",
    "EXITING",
    "EXIT_RETRIES",
    "HELD_CANCEL",
    "REASONS",
    "STOP_ENDED",
    "STOP_HELD",
    "STOP_UNCONFIRMED",
    "UNBOUND_ADD",
    "Unprotected",
    "Windows",
]
