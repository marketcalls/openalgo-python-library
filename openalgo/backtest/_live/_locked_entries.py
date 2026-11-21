"""The strategies whose entries are locked, as the order worker reads them (2.4.0 item
P3-1).

The kernel keeps each strategy's entry locks (`lock_entries`, 16-session-design R9), and
the loop sets and lifts them through `_session_bookkeeping.Bookkeeping.lock` and `unlock`.
An entry or add decided before a lock may still wait in the order worker's queue (the
order rate holds it). The loop withdraws such a placement when it hands over its next
commands (`_venue_withdraw`, 2.4.0 item M8-X2), but the worker runs on its own thread and
may take the entry off its queue between the loop step that locks the strategy's entries
and the one that hands it the withdrawal. So the call that changes a strategy's locks in
the kernel copies them here too, and the worker reads them just before it sends an entry
or add (`_worker_send`): a locked strategy's entry or add is answered
`NotSent(entries_locked)` and never sent, and the loop rejects it with `live_lock` as for
a withdrawal. Exits, stops, modifies and cancels never read it.

The kernel stays the one source of truth: this record only mirrors its reasons, copied in
the same call that changes them. The loop's thread writes; any thread reads.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence


def locked_words(reasons: Sequence[str]) -> str:
    """Why an entry or add of a strategy whose entries are locked for `reasons` was not
    sent, as its `NotSent(entries_locked)` says it."""
    return f"the strategy's entries were locked ({', '.join(reasons)}) before it was sent"


class LockedEntries:
    """The lock reasons of each strategy whose entries are locked (module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reasons: dict[str, tuple[str, ...]] = {}

    def note(self, strategy: str, reasons: Iterable[str]) -> None:
        """Strategy `strategy`'s entries are now locked for `reasons` (the kernel's, after
        a change); none means they are no longer locked."""
        locked = tuple(sorted(reasons))
        with self._lock:
            if locked:
                self._reasons[strategy] = locked
            else:
                self._reasons.pop(strategy, None)

    def reasons(self, strategy: str) -> tuple[str, ...]:
        """The reasons strategy `strategy`'s entries are locked for, sorted; empty while
        its entries are allowed."""
        with self._lock:
            return self._reasons.get(strategy, ())


__all__ = ["LockedEntries", "locked_words"]
