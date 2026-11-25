"""A carried derivatives leg on the last trading day of its contract (design
20-positional-live section 9; the lead's ruling N5b-3).

A stock future, an option and an MCX contract settled by delivery are exited by the kernel
at the square-off of their positional exit day (`_positional_exit`, `expiry_exit`), so none
is carried past it. A future settled in cash (an index future, an MCX cash contract) is
left to the exchange, which settles it at its expiry: the kernel cancels its stop GTT at
the close of its last trading day, and the next start records its exit at that last close
(`restore_carried_state`'s "expired"). Such a leg is carried with no stop by design: the
close's read of the GTT book does not judge it (`settles_today`), and its line at the end
says the exchange settles it (`SETTLES_WORDS`), with no ERROR alert of a missing stop.

Days are IST dates. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._positional_exit import PHYSICAL
from ._plan_catch_up import INSTRUMENT
from ._stops_gtt import overnight

#: The line of a leg the exchange settles today (module docstring).
SETTLES_WORDS = (
    "the exchange settles it at its expiry today, so its stop GTT was cancelled at the "
    "close; the next start records its exit at today's last close"
)


def settles_today(trader: Any, day: Any) -> bool:
    """Whether `trader`'s carried leg is a future settled in cash whose last trading day is
    `day`, the session's (module docstring)."""
    if not overnight(trader):
        return False
    instrument = trader.part.setup.instruments[INSTRUMENT]
    last = instrument.last_trading_day or instrument.expiry
    return instrument.kind == "future" and instrument.settlement != PHYSICAL and last == day


__all__ = ["SETTLES_WORDS", "settles_today"]
