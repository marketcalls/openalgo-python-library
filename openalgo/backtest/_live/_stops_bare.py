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
same way. Its stops and exits cover it, and so does a MARKET reversal: the kernel cancels
the short's stop before it sends `buy(..., reverse=True)`, and the buy closes the short
before it opens the long (2.4.0 item P2-3 a). A resting reversal (a LIMIT buy below the
market) does not: it may never fill, so it protects nothing against a rise (2.4.0 item
M8-X1); the session refuses it in the loop step that made it (`_session_unsent`, item
P3-2), after this rule has run. A leg that
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
        # A closing order the kernel has, sent or not, a stop or a market reversal among
        # them, covers the leg or is on its way to.
        return trader.open_units() != 0 and not _closing_orders(trader)

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


def _closing_orders(trader: Trader) -> list[int]:
    """The kernel orders of `trader` certain to close its position: its stops and exits
    (reduce-only), and a MARKET reversal, the entry the exit owner holds for the stop's
    cancel, which closes the position before it opens the other side. A resting reversal
    (LIMIT, SL or SL-M) may never fill, so it covers nothing (2.4.0 item M8-X1)."""
    closing = "SELL" if trader.open_units() > 0 else "BUY"
    reversals = []
    for kernel_order in trader.entry_orders():
        order = call(trader.engine.order, kernel_order)
        if order["order_type"] == "MARKET" and order["side"] == closing:
            reversals.append(kernel_order)
    return trader.exit_orders() + reversals


__all__ = ["BareLegs"]
