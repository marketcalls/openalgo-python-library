"""A GTT the GTT follower follows (`_gtt_follow`; 20-positional-live 3.1 to 3.3): its
journal binding with what the follower knows of it since (`Followed`), the endings a book
row shows and the outcome each is (`ENDINGS`), and the outcome posted to the loop for it
(`outcome_of`). Pure: the follower reads the books and the journal.

A GTT whose modify could not be told and whose cancel to place it again was refused for
its firing is followed with the modify's terms too (`Followed.also`, the lead's ruling
N3b-0 (a)): the broker may hold either, so its child is looked for, and its leg's units
judged, by both.

Times are UTC nanoseconds, prices rupees, quantities signed whole units.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._gtt_book import GttRow, Seen
from ._gtt_reads import LegKey
from ._gtt_types import GttEvent, GttOutcome
from ._journal_gtt_rows import GttBinding, GttTerms

#: How each ending the book shows is noted, and the outcome it is.
ENDINGS = {
    Seen.ENDED: (GttEvent.ENDED_OUTSIDE, "ended outside the session: the book shows {status}"),
    Seen.EXPIRED: (GttEvent.EXPIRED, "ended by the broker's expiry"),
    Seen.REJECTED: (GttEvent.REJECTED, "rejected by the broker after its placement"),
}


@dataclass
class Followed:
    """A GTT followed: its journal binding, when the last read that showed it resting
    began, and once it fired or is gone, since when and when its child is looked for next
    (the sandbox's child, when its row names one); whether its unknown status was alerted,
    the broker's trigger prices and units a resting row showed against the journal's terms
    that were reported (`other_terms`), and the terms of a modify the broker may hold
    instead of the journal's (`also`, module docstring; None for none)."""

    gtt: GttBinding
    active_ns: int | None = None
    since_ns: int | None = None
    next_ns: int = 0
    child: str | None = None
    warned: bool = False
    reported_terms: tuple[tuple[float, ...], int] | None = None
    also: GttTerms | None = None

    @property
    def leg(self) -> LegKey:
        """The leg the GTT holds: (symbol, exchange, product)."""
        return self.gtt.symbol, self.gtt.exchange, self.gtt.product


def other_terms(followed: Followed, row: GttRow) -> str:
    """Words for a resting `row` of `followed`'s GTT that does not show the journal's
    terms, or "" when it shows them or they were reported already (its trigger and units
    as the broker holds them against the journal's, 3.1)."""
    terms = followed.gtt.terms
    if row.shows(terms):
        followed.reported_terms = None
        return ""
    held = (row.triggers, row.quantity)
    if held == followed.reported_terms:
        return ""
    followed.reported_terms = held
    triggers = ", ".join(f"{price:.2f}" for price in row.triggers)
    return (
        f"the broker holds a trigger of {triggers} for {row.quantity} units, the journal "
        f"{terms.trigger_price:.2f} for {terms.quantity} units"
    )


def unknown_status(followed: Followed, row: GttRow) -> str:
    """The ERROR alert's words for a resting `row` of `followed`'s GTT whose status the
    engine does not know (taken as resting, 3.1), or "" when it knows it or alerted it
    already; it is alerted once per GTT."""
    if row.seen is not Seen.UNKNOWN or followed.warned:
        return ""
    followed.warned = True
    gtt = followed.gtt
    return (
        f"GTT {row.trigger_id} of {gtt.strategy} on {gtt.exchange}:{gtt.symbol} shows status "
        f"{row.status!r}, which the engine does not know: it is taken as resting. Check it "
        "in OpenAlgo"
    )


def outcome_of(
    followed: Followed,
    event: GttEvent,
    *,
    order_id: str = "",
    units: int = 0,
    price: float = 0.0,
    flags: tuple[str, ...] = (),
    candidates: tuple[str, ...] = (),
    message: str = "",
    held: int | None = None,
) -> GttOutcome:
    """The outcome `event` of `followed`'s GTT, with what `GttOutcome` says it carries."""
    gtt = followed.gtt
    return GttOutcome(
        seq=gtt.seq,
        strategy=gtt.strategy,
        gtt_id=gtt.gtt_id,
        trigger_id=gtt.trigger_id or "",
        symbol=gtt.symbol,
        exchange=gtt.exchange,
        product=gtt.product,
        event=event,
        order_id=order_id,
        units=units,
        price=price,
        flags=flags,
        candidates=candidates,
        message=message,
        held=held,
    )


__all__ = ["ENDINGS", "Followed", "other_terms", "outcome_of", "unknown_status"]
