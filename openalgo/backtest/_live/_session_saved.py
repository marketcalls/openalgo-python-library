"""What journal L keeps of each strategy for a restart (16-session-design 2.5 E and F, 3.6):
written by the loop, read back by the restart.

Per strategy and trading day, `put_rms_state(strategy, saved, day=day)` with `saved` a
plain-data dict:

| Key | What |
|---|---|
| "live_state" | the kernel session's `live_state()` (K4, R14) |
| "venue_map" | the venue map's records of the orders the kernel still works (`_venue_saved`) |
| "adopted" | the signed units the reconciler adopted from other orders' trades (5.3) |

It is saved before each batch of the kernel's commands reaches the order worker (design
rule 4: a restart never meets an order the saved state lacks) and after each slice a
strategy releases (the trail's extremes move on ticks without commands). A kernel session
that has finished saves nothing: no session resumes at its close. A journal that cannot
write raises JournalError, which stops the session (3.5).

Quantities are signed units, prices rupees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._kill_switch import ACCOUNT_LOSS
from ._plan_catch_up import INSTRUMENT
from ._session_exits import EXIT_FAILED_LOCK
from ._session_news import RECONCILE_LOCK
from ._session_trader import Trader
from ._venue_requests import LIVE_PRODUCT
from ._venue_saved import SavedOrder, read_records

#: The roles of the kernel's saved orders that are leg stops (the binding's ORDER_ROLES).
LEG_STOP_ROLE = "leg_stop"
#: The entry locks that last the day, set again at a restart from the saved state (6.2):
#: a reconciliation mismatch, an exit that failed every attempt, the account's daily loss.
DAY_LOCKS = frozenset({RECONCILE_LOCK, EXIT_FAILED_LOCK, ACCOUNT_LOSS})


@dataclass(frozen=True)
class Saved:
    """A strategy's saved state, read back: the kernel's `live_state`, the venue map's
    orders by saved kernel id, whether each saved working order was sent (`sent`), and the
    units the reconciler had adopted."""

    live_state: dict[str, Any]
    orders: dict[int, SavedOrder]
    sent: dict[int, bool]
    adopted: int

    @property
    def units(self) -> int:
        """The saved position of the strategy's instrument in MIS, signed units."""
        for position in self.live_state.get("positions", []):
            if position.get("instrument") == INSTRUMENT and position.get("product") == LIVE_PRODUCT:
                return round(float(position["quantity_units"]))
        return 0

    @property
    def locks(self) -> frozenset[str]:
        """The saved entry locks that last the day (`DAY_LOCKS`)."""
        return frozenset(self.live_state.get("desk", {}).get("entry_locks", [])) & DAY_LOCKS

    @property
    def stop(self) -> SavedOrder | None:
        """The saved order of the leg's broker-held stop, None when the leg had none."""
        for order in self.live_state.get("orders", []):
            if order.get("role") == LEG_STOP_ROLE and order.get("sent"):
                return self.orders.get(int(order["order"]))
        return None


def save(session: Any, trader: Trader) -> None:
    """Save `trader`'s state to `session`'s journal L for its day (module docstring);
    nothing once its kernel session finished. Raises JournalError when the journal
    cannot write it."""
    if trader.finished:
        return
    saved = {
        "live_state": call(trader.engine.live_state),
        "venue_map": trader.venue.saved(),
        "adopted": session.reconcile.adopted(trader.name),
    }
    session.parts.journal.put_rms_state(trader.name, saved, day=session.parts.spec.day)


def read_saved(state: Mapping[str, Any] | None, strategy: str) -> Saved | None:
    """A strategy's saved state from `rms_state` (None: nothing saved). Raises LiveError
    for one that cannot be read, naming the strategy: a restart never guesses."""
    if state is None:
        return None
    try:
        live_state = dict(state["live_state"])
        orders = read_records(state["venue_map"])
        sent = {int(order["order"]): bool(order["sent"]) for order in live_state["orders"]}
        adopted = int(state.get("adopted", 0))
    except (KeyError, TypeError, ValueError, LiveError) as error:
        raise LiveError(
            f"the saved state of {strategy} in the journal cannot be read ({error}): the "
            "restart cannot resume it. Check its positions in OpenAlgo, then start under "
            "another account name (account=) once they are closed"
        ) from None
    return Saved(live_state, orders, sent, adopted)


__all__ = ["DAY_LOCKS", "LEG_STOP_ROLE", "Saved", "read_saved", "save"]
