"""A GTT the GTT follower follows (`_gtt_follow`; 20-positional-live 3.1 to 3.3): its
journal binding with what the follower knows of it since (`Followed`), the endings a book
row shows and the outcome each is (`ENDINGS`), and the outcome posted to the loop for it
(`outcome_of`). Pure: the follower reads the books and the journal.

Times are UTC nanoseconds, prices rupees, quantities signed whole units.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._gtt_book import Seen
from ._gtt_reads import LegKey
from ._gtt_types import GttEvent, GttOutcome
from ._journal_gtt_rows import GttBinding

#: How each ending the book shows is noted, and the outcome it is.
ENDINGS = {
    Seen.ENDED: (GttEvent.ENDED_OUTSIDE, "ended outside the session: the book shows {status}"),
    Seen.EXPIRED: (GttEvent.EXPIRED, "ended by the broker's expiry"),
    Seen.REJECTED: (GttEvent.REJECTED, "rejected by the broker after its placement"),
}


@dataclass
class Followed:
    """A GTT followed: its journal binding, its leg's units before the last read that
    showed it resting (its baseline, kept in the journal across days) and when that read
    began, and once it fired or is gone, since when and when its child is looked for next
    (the sandbox's child, when its row names one); whether its unknown status was alerted."""

    gtt: GttBinding
    units_before: int | None = None
    active_ns: int | None = None
    since_ns: int | None = None
    next_ns: int = 0
    child: str | None = None
    warned: bool = False

    @property
    def leg(self) -> LegKey:
        """The leg the GTT holds: (symbol, exchange, product)."""
        return self.gtt.symbol, self.gtt.exchange, self.gtt.product


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
    )


__all__ = ["ENDINGS", "Followed", "outcome_of"]
