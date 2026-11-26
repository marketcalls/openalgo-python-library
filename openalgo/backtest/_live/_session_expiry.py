"""A carried derivatives leg on the last trading day of its contract (design
20-positional-live section 9; the lead's ruling N5b-3).

A stock future, an option and an MCX contract settled by delivery are exited by the kernel
at the square-off of their positional exit day (`_positional_exit`, `expiry_exit`), so none
is carried past it. A future settled in cash (an index future, an MCX cash contract) is
left to the exchange, which settles it at its expiry: the kernel cancels its stop GTT at
the close of its last trading day. The next start, on the contract traded then (one that
still names the expired contract is refused at its plan's first step), names the settlement
and that last close in a warning and a SESSION alert (`_carry_contracts`); no exit is
recorded, so no result counts its profit or loss after that close (the lead's ruling 2 of
review round 4 of unit N5a). Such a leg is carried with no stop by design: the close's read
of the GTT book does not judge it (`settles_today`), and its line at the end says the
exchange settles it (`settles_words`: a stop GTT cancelled, or under
`positional_stops="rearm_daily"` no stop needed from the close), with no ERROR alert of a
missing stop and no place in the close's WARN alert of rearm_daily legs left without a stop.

Days are IST dates. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._positional_exit import PHYSICAL
from ._plan_catch_up import INSTRUMENT
from ._stops_gtt import overnight, stop_gtt

#: The line of a leg the exchange settles today (module docstring), its stop a GTT.
SETTLES_WORDS = (
    "the exchange settles it at its expiry today, so its stop GTT was cancelled at the "
    "close; the next start, on the contract traded then, names the settlement and today's "
    "last close"
)
#: The same, for a leg whose stop is a day stop (`positional_stops="rearm_daily"`), which
#: the close ends as any day's: no GTT was ever placed.
SETTLES_DAY_STOP_WORDS = (
    "the exchange settles it at its expiry today, so it needs no stop from the close; the "
    "next start, on the contract traded then, names the settlement and today's last close"
)


def settles_words(trader: Any) -> str:
    """The line of `trader`'s leg the exchange settles today, by how its stop is held."""
    return SETTLES_WORDS if stop_gtt(trader) else SETTLES_DAY_STOP_WORDS


def settles_today(trader: Any, day: Any) -> bool:
    """Whether `trader`'s carried leg is a future settled in cash whose last trading day is
    `day`, the session's (module docstring)."""
    if not overnight(trader):
        return False
    instrument = trader.part.setup.instruments[INSTRUMENT]
    last = instrument.last_trading_day or instrument.expiry
    return instrument.kind == "future" and instrument.settlement != PHYSICAL and last == day


__all__ = ["SETTLES_DAY_STOP_WORDS", "SETTLES_WORDS", "settles_today", "settles_words"]
