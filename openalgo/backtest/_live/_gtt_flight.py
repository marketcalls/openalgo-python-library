"""The legs a GTT holds in flight for the reconciler (20-positional-live 3.2, last
paragraph; review H3; the lead's ruling N3a-2).

While a leg's GTT has fired or is gone from the book and the GTT follower has not resolved
it, the leg is in flight: an order of the child's shape is neither another's (the foreign
order rule of 16-session-design 5.2 step 2) nor an external trade (5.3 rows 1 and 2), and
a fall of the position by the GTT's units is not a mismatch. The reconciler therefore
compares no such leg (`_reconcile_steps.in_flight`) until the follower's outcome resolves
it; then its comparisons apply those rules as before. A leg in flight past the
reconciler's limit is compared late, as any (16-session-design 5.1).

Fed with the GTT news of the order worker in the order the loop takes it (`take`): an
IN_FLIGHT outcome holds its GTT's leg, any outcome that resolves it lets it go, TRIGGERED
too, before the child is bound to the kernel: the session binds it in the loop step that
takes TRIGGERED (unit N3b, `_gtt_types`), or the child's fill is compared as external. Legs are
named by their strategy: a strategy trades one symbol in one product. Used by the loop
only.
"""

from __future__ import annotations

from ._gtt_types import GttAnswer, GttEvent, GttOutcome


class GttFlight:
    """The strategies whose leg a GTT holds in flight (module docstring)."""

    def __init__(self) -> None:
        # By strategy: the journal numbers of its GTTs fired or gone and not resolved.
        self._flying: dict[str, set[int]] = {}

    def take(self, news: GttAnswer | GttOutcome) -> None:
        """Take one GTT news item of the order worker (answers change nothing here)."""
        if not isinstance(news, GttOutcome):
            return
        seqs = self._flying.setdefault(news.strategy, set())
        if news.event is GttEvent.IN_FLIGHT:
            seqs.add(news.seq)
        else:
            seqs.discard(news.seq)
        if not seqs:
            del self._flying[news.strategy]

    def holds(self, strategy: str) -> bool:
        """True while a GTT of `strategy` has fired or is gone and is not resolved."""
        return strategy in self._flying


__all__ = ["GttFlight"]
