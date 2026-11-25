"""What happened inside the fake OpenAlgo, in order, for a test to assert on.

Every REST call, every fault applied, every tick and order update sent or lost, every
fill, every GTT that fired and every switch flipped is an `Entry`. A call the fake does
not model, or a failure inside the fake itself, is a problem: the SDK turns any
exception of its HTTP client into
an "unknown_error" answer, which would hide it, so the fixture fails the test on any
problem instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

#: The kinds of entry.
CALL = "call"
FAULT = "fault"
TICK = "tick"
ORDER_UPDATE = "order_update"
FILL = "fill"
SWITCH = "switch"
TELEGRAM = "telegram"
PROBLEM = "problem"
#: A GTT fired and placed its child order.
GTT = "gtt"
KINDS = frozenset({CALL, FAULT, TICK, ORDER_UPDATE, FILL, SWITCH, TELEGRAM, PROBLEM, GTT})


@dataclass(frozen=True)
class Entry:
    """One thing that happened at `ts_ns` (UTC nanoseconds): its `kind` (one of `KINDS`),
    what it was (`what`: an endpoint, a fault's name, an order ID) and its details."""

    ts_ns: int
    kind: str
    what: str
    detail: dict[str, Any] = field(default_factory=dict)


class FakeLog:
    """The fake's log: entries in the order they happened. Safe to write from the feed's
    thread and the test's alike."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._lock = threading.Lock()

    def record(self, ts_ns: int, kind: str, what: str, **detail: Any) -> Entry:
        """Add an entry and return it. Raises ValueError for an unknown kind."""
        if kind not in KINDS:
            raise ValueError(f"unknown log kind {kind!r}: use one of {sorted(KINDS)}")
        entry = Entry(ts_ns, kind, what, dict(detail))
        with self._lock:
            self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[Entry]:
        """Every entry so far, oldest first (a copy)."""
        with self._lock:
            return list(self._entries)

    def of(self, kind: str, what: str | None = None) -> list[Entry]:
        """The entries of one kind, oldest first; only those about `what` when given."""
        return [e for e in self.entries if e.kind == kind and (what is None or e.what == what)]

    def faults(self) -> list[str]:
        """The name of every fault applied so far, in order."""
        return [entry.what for entry in self.of(FAULT)]

    def calls(self, endpoint: str | None = None) -> list[Entry]:
        """The REST calls answered so far (of one endpoint when given), oldest first."""
        return self.of(CALL, endpoint)

    def problems(self) -> list[Entry]:
        """Calls the fake does not model and failures inside the fake: a test with any
        fails."""
        return self.of(PROBLEM)


__all__ = [
    "CALL",
    "FAULT",
    "FILL",
    "GTT",
    "KINDS",
    "ORDER_UPDATE",
    "PROBLEM",
    "SWITCH",
    "TELEGRAM",
    "TICK",
    "Entry",
    "FakeLog",
]
