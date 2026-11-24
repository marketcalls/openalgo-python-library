"""How the GTT follower (`_gtt_follow`) posts what it found of a GTT, split from it so each
module keeps one job (20-positional-live 3.2, 3.3; the lead's ruling N3a-2): each finding
a typed outcome (`_gtt_types.GttOutcome`, as `inbox.GttNews`), the journal noted first.
Triggered (its child found), its child missed (the position's move applied at the trade
book's price, or at the GTT's limit flagged `price_unverified`), ended outside the session,
expired or rejected, and a mismatch the engine never guesses through stop the following
of the GTT; the follower posts the others (in flight, resting again, other terms) and
follows on. The orders already taken (the session's own, or another GTT's child) are never
a GTT's child (3.2 rule 2: `taken`).

Prices are rupees, quantities whole units. Used by the worker's thread only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from ._gtt_child import missed_price
from ._gtt_followed import Followed, outcome_of
from ._gtt_types import PRICE_UNVERIFIED, GttEvent
from ._journal_gtt_rows import GttState
from ._types import LOGGER_NAME
from .inbox import GttNews, Message
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)


class GttOutcomes:
    """The outcomes (module docstring) of the follower's GTTs, `followed` by journal
    number (an outcome that ends the following removes its GTT); `journal` is handle W and
    `post` puts a message on the loop's inbox."""

    def __init__(
        self,
        journal: Journal,
        post: Callable[[Message], None],
        followed: dict[int, Followed],
    ) -> None:
        self._journal = journal
        self._post = post
        self._followed = followed
        self._children: set[str] = set()

    def post(self, followed: Followed, event: GttEvent, **fields: Any) -> None:
        """Post the outcome `event` of `followed`'s GTT (`_gtt_followed.outcome_of`)."""
        gtt = followed.gtt
        logger.info("GTT %s of %s: %s", gtt.trigger_id, gtt.strategy, event.value)
        self._post(GttNews(outcome_of(followed, event, **fields)))

    def triggered(self, followed: Followed, child: str) -> None:
        """`followed`'s GTT fired and placed order `child`, which the session binds."""
        self._children.add(child)
        self._followed.pop(followed.gtt.seq, None)
        self.post(followed, GttEvent.TRIGGERED, order_id=child)

    def missed(self, followed: Followed, tradebook: Mapping[str, Any] | None) -> None:
        """`followed`'s GTT fired with its child missed: its units at the price of
        `tradebook`'s trades (the trade book's answer, None when not read), or at its limit
        flagged; the journal notes it fired."""
        gtt = followed.gtt
        price = missed_price(gtt, tradebook, self.taken)
        flags = () if price is not None else (PRICE_UNVERIFIED,)
        price = gtt.terms.limit_price if price is None else price
        note = f"fired, its child missed: {gtt.terms.quantity} units at {price:.2f}"
        self._journal.set_gtt_state(gtt.seq, GttState.FIRED, note=note)
        self._followed.pop(gtt.seq, None)
        units = gtt.terms.quantity
        self.post(followed, GttEvent.CHILD_MISSED, units=units, price=price, flags=flags)

    def unresolved(self, followed: Followed, words: str, candidates: tuple[str, ...] = ()) -> None:
        """A GTT whose fate the engine cannot tell: a MISMATCH posted, never a guess; the
        journal keeps it TRIGGERED for the user to settle."""
        self._followed.pop(followed.gtt.seq, None)
        self.post(followed, GttEvent.MISMATCH, candidates=candidates, message=words)

    def end(self, followed: Followed, event: GttEvent, note: str) -> None:
        """`followed`'s GTT ended without firing (`event`): ENDED in the journal."""
        self._journal.set_gtt_state(followed.gtt.seq, GttState.ENDED, note=note)
        self._followed.pop(followed.gtt.seq, None)
        self.post(followed, event, message=note)

    def taken(self, order_id: str) -> bool:
        """Whether an order is the session's own or another GTT's child (3.2 rule 2)."""
        if order_id in self._children or self._journal.intent_of(order_id) is not None:
            return True
        return self._journal.gtt_of_child(order_id) is not None


__all__ = ["GttOutcomes"]
