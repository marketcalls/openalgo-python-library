"""A guarded leg left bare (16-session-design 4.2, 4.8; the K3 hand-off on a rejected
partial target): units open with no stop of the leg working or waiting to be sent,
nothing else working or waiting to close them, and no exit retry of the session owning
the leg.

The kernel places a leg's stop again only after a stop that ended by itself (R3); a stop
it cancelled for an exit stays cancelled. When the exits of the position together close
it all (two targets reached on one tick, a `reduce()` then an `exit()` in one bar) the
stop is cancelled for them, and if a partial one among them is then rejected while the
rest fill, the units it was for have no stop and nothing to close them until the
square-off. The partial exit is not asked for again (`_session_exits`: it would close
more than the strategy asked for); instead the session places the leg's stop again over
what is left (`protect`, R7) and sends an ERROR alert, once each time the leg is left
bare. The alert says what the leg is (units with no stop and nothing working to close
them), not how it got there: an exit filled in part and then cancelled leaves it bare the
same way. Any order working on the side that closes the position covers it, a reversal
among them: the kernel cancels the short's stop before it sends `buy(..., reverse=True)`,
and the buy closes the short before it opens the long (2.4.0 item P2-3 a). A leg that
never had a stop, one protect by state holds, one the reconciler's exit guard holds (the
broker's books show fewer units), one a closing order's rejection holds on a shared
instrument (18-multi-strategy 4.14a), and every leg once the safe stop has begun (its
flatten exits the position) are left to their own rules.

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._engine import call
from ._session_trader import Trader
from ._stops_kernel import PROTECT_PLACED, protect
from ._stops_report import protect_words, while_running
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)


class BareLegs:
    """The bare legs of one session (module docstring). `session` is the
    `_session_loop.Session`, `stops` its `StopsSupervisor`."""

    def __init__(self, session: Any, stops: Any) -> None:
        self._session = session
        self._stops = stops
        self._bare: set[str] = set()

    def due(self, now_ns: int) -> None:
        """Place the stop again of each leg newly left bare (module docstring)."""
        del now_ns  # a leg is bare by its state alone
        for trader in self._session.traders:
            if not self._is_bare(trader):
                self._bare.discard(trader.name)
            elif trader.name not in self._bare:
                self._bare.add(trader.name)
                self._protect(trader)

    def _is_bare(self, trader: Trader) -> bool:
        session = self._session
        if trader.finished or session.exits.owns(trader.name):
            return False
        if session.control is not None and session.control.stopping:
            return False
        if not self._stops.leg(trader).guarded or self._stops.protection.holds(trader):
            return False
        # While the broker's books show fewer units, a stop for the engine's units could
        # take the broker through zero (the reconciler's exit guard holds the leg); after
        # a closing order's rejection on a shared instrument, the broker may not hold them
        # (`_margin_reject`).
        if session.reconcile.holds(trader) or session.margin.holds(trader):
            return False
        # Any closing order the kernel has, sent or not, a stop or a reversal among them,
        # covers the leg or is on its way to.
        held = trader.open_units()
        return held != 0 and not _closing_orders(trader, held)

    def _protect(self, trader: Trader) -> None:
        session, name = self._session, trader.name
        outcome = protect(trader)
        session.event("protect", strategy=name, outcome=outcome, bare=True)
        units = abs(trader.open_units())
        if outcome not in PROTECT_PLACED:
            logger.warning("%s: a bare leg's stop not placed again (%s)", name, outcome)
        held, then = protect_words(outcome)
        session.alert(
            AlertKind.ERROR,
            f"{name}: {units:g} {trader.symbol} {trader.exchange} had no stop at the broker "
            f"and nothing working to close them: {held}; {while_running(then)}",
        )


def _closing_orders(trader: Trader, held_units: float) -> list[int]:
    """The kernel orders of `trader` working on the side that closes its position of
    `held_units` (signed): its stops and exits, and a reversal, an entry beyond the
    position on that side, which closes it before it opens the other side."""
    closing = "SELL" if held_units > 0 else "BUY"
    engine = trader.engine
    return [order for order in trader.working() if call(engine.order, order)["side"] == closing]


__all__ = ["BareLegs"]
