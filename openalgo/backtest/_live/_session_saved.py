"""What journal L keeps of each strategy for a restart (16-session-design 2.5 E and F, 3.6):
written by the loop, read back by the restart.

Per strategy and trading day, `put_rms_state(strategy, saved, day=day)` with `saved` a
plain-data dict:

| Key | What |
|---|---|
| "live_state" | the kernel session's `live_state()` (K4, R14) |
| "venue_map" | the venue map's records of the orders the kernel still works (`_venue_saved`) |
| "adopted" | the signed units the reconciler adopted from other orders' trades (5.3) |
| "figures" | the strategy's figures of the day as that state holds them (`_session_figures`) |

It is saved before each batch of the kernel's commands reaches the order worker (design
rule 4: a restart never meets an order the saved state lacks) and after each slice a
strategy releases (the trail's extremes move on ticks without commands), with the
strategy's `self.state` when it changed (`save_released`), and the session's own counts
of the day after them (`save_session`, under `_session_figures.SESSION_KEY`); the close
saves them once more after its final fills (`_session_close`). A kernel session that has
finished saves nothing: no session resumes at its close. A journal that cannot write
raises JournalError, which stops the session (3.5).

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
from ._session_figures import FIGURES, SESSION_KEY
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

    def leg_stop(self, order: int) -> bool:
        """True when saved kernel order `order` is a leg's broker-held stop."""
        return any(
            int(saved["order"]) == order and saved.get("role") == LEG_STOP_ROLE
            for saved in self.live_state.get("orders", [])
        )

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
        FIGURES: session.figures.strategy(trader),
    }
    session.parts.journal.put_rms_state(trader.name, saved, day=session.parts.spec.day)


def save_session(session: Any, *, unprotected_s: float | None = None, named: int = 0) -> None:
    """Save `session`'s own counts of the day to journal L (module docstring), with the
    close's `unprotected_s` and `named` mismatches (`DayFigures.session_counts`). Raises
    JournalError when the journal cannot write them."""
    counts = session.figures.session_counts(session, unprotected_s=unprotected_s, named=named)
    session.parts.journal.put_rms_state(SESSION_KEY, counts, day=session.parts.spec.day)


def save_released(session: Any, names: set[str]) -> None:
    """The loop's step F: each strategy of `session` named in `names` (those that released
    a slice) saved, and its `self.state` when it changed, to journal L, then the session's
    counts. Raises JournalError when the journal cannot write them."""
    for trader in session.traders:
        if trader.name in names:
            save(session, trader)
            state = trader.state_changed()
            if state is not None:
                session.parts.journal.put_user_state(trader.name, state)
    save_session(session)


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


__all__ = [
    "DAY_LOCKS",
    "LEG_STOP_ROLE",
    "Saved",
    "read_saved",
    "save",
    "save_released",
    "save_session",
]
