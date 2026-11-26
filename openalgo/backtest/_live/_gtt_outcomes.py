"""How the GTT follower (`_gtt_follow`) posts what it found of a GTT, split from it so each
module keeps one job (20-positional-live 3.2, 3.3; the lead's ruling N3a-2): each finding
a typed outcome (`_gtt_types.GttOutcome`, as `inbox.GttNews`), the journal noted first.
Triggered (its child found), its child missed (the position's move applied at the trade
book's price, or at the GTT's limit flagged `price_unverified`, and `missed_day` when it
fired on a trading day no session ran, 20-positional-live 4.3), ended outside the session,
expired or rejected, and a mismatch the engine never guesses through stop the following
of the GTT; the follower posts the others (in flight, resting again, other terms) and
follows on. The orders already taken (the session's own, or another GTT's child) are never
a GTT's child (3.2 rule 2: `taken`).

**A GTT gone with no child found** (`gone`, 3.3; the lead's ruling M4 on N3a's review
round 5) is decided by what its leg's units say against its reference (`_gtt_units`):
fallen (or, for a strategy's GTT entry, risen) by its units, it fired with its child
missed; unchanged, it ended outside the session, unless the book's row says it triggered
(the lead's ruling N3b-0 (a): a child of it may work unlisted, so it is a MISMATCH); moved
by anything else, or with no reference, a MISMATCH. A read made while the session's own
fills on the leg had not settled, or whose units could not be read, decides nothing until
the wait for a settled read is over (`_gtt_reads.BookRead.waited_out`): then it is a
MISMATCH too.

Prices are rupees, quantities whole units. Used by the worker's thread only.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Mapping
from typing import Any

from .._series import ist_time
from ..calendar import default_calendar
from ._gtt_child import missed_price
from ._gtt_followed import Followed, outcome_of
from ._gtt_types import MISSED_DAY, PRICE_UNVERIFIED, GttEvent
from ._gtt_units import UNTOLD, LegUnits, References, UnitsSign, went
from ._journal_gtt_rows import GttState, GttTerms
from ._preflight_carry import missed_days
from ._types import LOGGER_NAME, Now
from .inbox import GttNews, Message
from .journal import Journal

logger = logging.getLogger(LOGGER_NAME)


class GttOutcomes:
    """The outcomes (module docstring) of the follower's GTTs, `followed` by journal
    number with their `references` (an outcome that ends the following removes its GTT and
    its reference); `journal` is handle W and `post` puts a message on the loop's inbox."""

    def __init__(
        self,
        journal: Journal,
        post: Callable[[Message], None],
        followed: dict[int, Followed],
        references: References,
        now: Now,
    ) -> None:
        self._journal = journal
        self._post = post
        self._followed = followed
        self._references = references
        self._now = now
        self._children: set[str] = set()

    def post(self, followed: Followed, event: GttEvent, **fields: Any) -> None:
        """Post the outcome `event` of `followed`'s GTT (`_gtt_followed.outcome_of`)."""
        gtt = followed.gtt
        logger.info("GTT %s of %s: %s", gtt.trigger_id, gtt.strategy, event.value)
        self._post(GttNews(outcome_of(followed, event, **fields)))

    def triggered(self, followed: Followed, child: str) -> None:
        """`followed`'s GTT fired and placed order `child`, which the session binds."""
        self._children.add(child)
        self._stop(followed.gtt.seq)
        self.post(followed, GttEvent.TRIGGERED, order_id=child)

    def missed(
        self, followed: Followed, terms: GttTerms, tradebook: Mapping[str, Any] | None
    ) -> None:
        """`followed`'s GTT fired on `terms` with its child missed: their units at the
        price of `tradebook`'s trades (the trade book's answer, None when not read), or at
        their limit flagged `price_unverified`, and `missed_day` too when a trading day of
        its exchange lies between its last resting read's day and today: it fired on a day
        no session ran, whose trades today's trade book does not hold (20-positional-live
        4.3); the journal notes it fired."""
        gtt = followed.gtt
        price = missed_price(gtt, terms, tradebook, self.taken)
        flags: tuple[str, ...] = () if price is not None else (PRICE_UNVERIFIED,)
        if price is None and self._missed_days(followed):
            flags += (MISSED_DAY,)
        price = terms.limit_price if price is None else price
        note = f"fired, its child missed: {terms.quantity} units at {price:.2f}"
        self._journal.set_gtt_state(gtt.seq, GttState.FIRED, note=note)
        self._stop(gtt.seq)
        units = terms.quantity
        self.post(followed, GttEvent.CHILD_MISSED, units=units, price=price, flags=flags)

    def unresolved(self, followed: Followed, words: str, candidates: tuple[str, ...] = ()) -> None:
        """A GTT whose fate the engine cannot tell: a MISMATCH posted, never a guess; the
        journal keeps it TRIGGERED for the user to settle."""
        self._stop(followed.gtt.seq)
        self.post(followed, GttEvent.MISMATCH, candidates=candidates, message=words)

    def end(self, followed: Followed, event: GttEvent, note: str, held: int | None = None) -> None:
        """`followed`'s GTT ended without firing (`event`): ENDED in the journal; `held`
        is its leg's units at the broker in the read that judged it, when one did."""
        self._journal.set_gtt_state(followed.gtt.seq, GttState.ENDED, note=note)
        self._stop(followed.gtt.seq)
        self.post(followed, event, message=note, held=held)

    def gone(
        self,
        followed: Followed,
        found: LegUnits,
        *,
        waited_out: bool,
        trades: Callable[[], Mapping[str, Any] | None],
        triggered: bool = False,
    ) -> None:
        """`followed`'s GTT is gone with no child found: `found` is what its leg's units
        say against its reference, `waited_out` whether the wait for the leg to settle is
        over, `trades()` reads the trade book, and `triggered` is True when the book's row
        says it fired (module docstring)."""
        if found.sign is UnitsSign.FIRED:
            self.missed(followed, found.terms or followed.gtt.terms, trades())
        elif found.sign is UnitsSign.UNCHANGED and triggered:
            words = (
                "the GTT book shows it triggered, yet no child of it was found and the "
                "position is unchanged: its child may work unlisted, so it is not taken as "
                "ended outside the session"
            )
            self.unresolved(followed, words)
        elif found.sign is UnitsSign.UNCHANGED:
            note = "gone with no child found and the position unchanged: ended outside the session"
            self.end(followed, GttEvent.ENDED_OUTSIDE, note, held=found.units)
        elif found.sign is UnitsSign.OTHER:
            self.unresolved(followed, went(found))
        elif found.sign is UnitsSign.NO_REFERENCE or waited_out:
            self.unresolved(followed, f"{UNTOLD[found.sign]}: whether it fired cannot be told")

    def _missed_days(self, followed: Followed) -> tuple[dt.date, ...]:
        """The trading days of `followed`'s GTT's exchange after its last resting read's
        IST day and before today's (none when it has none)."""
        if followed.active_ns is None:
            return ()
        rested, today = ist_time(followed.active_ns).date(), ist_time(self._now()).date()
        return missed_days(followed.gtt.exchange, rested, today, default_calendar())

    def _stop(self, seq: int) -> None:
        self._followed.pop(seq, None)
        self._references.drop(seq)

    def taken(self, order_id: str) -> bool:
        """Whether an order is the session's own or another GTT's child (3.2 rule 2)."""
        if order_id in self._children or self._journal.intent_of(order_id) is not None:
            return True
        return self._journal.gtt_of_child(order_id) is not None


__all__ = ["GttOutcomes"]
