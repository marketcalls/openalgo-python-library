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
| "fill_comparison" | its fills against a backtest, the day's so far (`_fill_comparison`) |
| "gtt_bindings" | the journal's GTT each kernel GTT is bound to (`_carry_restore.gtt_bindings`) |

It is saved before each batch of the kernel's commands reaches the order worker (design
rule 4: a restart never meets an order the saved state lacks) and after each slice a
strategy releases (the trail's extremes move on ticks without commands), with the
strategy's `self.state` when it changed (`save_released`), and the session's own counts
of the day after them (`save_session`, under `_session_figures.SESSION_KEY`); the close
saves them once more after its final fills (`_session_close`). Each save teaches the
session the level of each target exit the live state shows working
(`_session_slippage.TargetLevels`), which the fill comparison sets the exit's fill
against. A kernel session that has finished saves nothing: no session resumes at its
close. A journal that cannot write raises JournalError, which stops the session (3.5).

Quantities are signed units, prices rupees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._carry_restore import gtt_bindings
from ._carry_start import BINDINGS_KEY
from ._fill_comparison import COMPARED, slippage_of, strategy_rows
from ._kill_switch import ACCOUNT_LOSS, ACCOUNT_PROFIT
from ._plan_catch_up import INSTRUMENT
from ._session_exits import EXIT_FAILED_LOCK
from ._session_figures import FIGURES, SESSION_KEY
from ._session_gtts import GTT_REFUSED_LOCK
from ._session_news import RECONCILE_LOCK
from ._session_trader import Trader
from ._venue_saved import SavedOrder, read_records

#: The roles of the kernel's saved orders that are leg stops (the binding's ORDER_ROLES).
LEG_STOP_ROLE = "leg_stop"
#: The entry locks that last the day, set again at a restart from the saved state (6.2):
#: a reconciliation mismatch, an exit that failed every attempt, the account's daily loss
#: limit and daily profit target (18-multi-strategy 2.5a rule 8), and a stop GTT the
#: broker refused at placement (20-positional-live 3.5, decision 11.7: the kernel's own
#: lock, which its restore does not set again either).
DAY_LOCKS = frozenset(
    {RECONCILE_LOCK, EXIT_FAILED_LOCK, ACCOUNT_LOSS, ACCOUNT_PROFIT, GTT_REFUSED_LOCK}
)


@dataclass(frozen=True)
class Saved:
    """A strategy's saved state, read back: the kernel's `live_state`, the venue map's
    orders by saved kernel id, whether each saved working order was sent (`sent`), the
    units the reconciler had adopted, the product the strategy trades, whose position the
    state is read for, and the journal's GTT each saved kernel GTT is bound to."""

    live_state: dict[str, Any]
    orders: dict[int, SavedOrder]
    sent: dict[int, bool]
    adopted: int
    product: str
    bindings: dict[int, int] = field(default_factory=dict)

    @property
    def units(self) -> int:
        """The saved position of the strategy's instrument in its product, signed units."""
        position = self._position()
        return 0 if position is None else round(float(position["quantity_units"]))

    @property
    def opened_ns(self) -> int | None:
        """When the saved position opened, UTC nanoseconds: its oldest open tranche's
        opening; None when it holds none."""
        position = self._position()
        tranches = [] if position is None else position.get("tranches", [])
        opened = [int(t["opened_at_ns"]) for t in tranches if float(t["open_units"])]
        return min(opened) if opened else None

    def _position(self) -> dict[str, Any] | None:
        """The saved position of the strategy's instrument in its product, None when it
        has none."""
        for position in self.live_state.get("positions", []):
            if position.get("instrument") == INSTRUMENT and position.get("product") == self.product:
                return position
        return None

    @property
    def instrument(self) -> tuple[str, str] | None:
        """(symbol, exchange) of the instrument the saved state trades, None when it names
        none (the kernel's restore then refuses it as unreadable)."""
        instruments = self.live_state.get("instruments") or []
        if len(instruments) <= INSTRUMENT:
            return None
        found = instruments[INSTRUMENT]
        return str(found.get("symbol")), str(found.get("exchange"))

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
    live_state = call(trader.engine.live_state)
    session.levels.learn(trader.name, live_state)
    rows = strategy_rows(session, trader)
    saved = {
        "live_state": live_state,
        "venue_map": trader.venue.saved(),
        "adopted": session.reconcile.adopted(trader.name),
        FIGURES: session.figures.strategy(trader, slippage_of(rows)),
        COMPARED: session.figures.rows.saved(trader.name, rows),
        BINDINGS_KEY: gtt_bindings(trader),
    }
    session.parts.journal.put_rms_state(trader.name, saved, day=session.parts.spec.day)


def save_session(session: Any, *, unprotected_s: float | None = None, named: int = 0) -> None:
    """Save `session`'s own counts of the day to journal L (module docstring), with the
    close's `unprotected_s` and `named` mismatches (`DayFigures.session_counts`). Raises
    JournalError when the journal cannot write them."""
    counts = session.figures.session_counts(session, unprotected_s=unprotected_s, named=named)
    session.parts.journal.put_rms_state(SESSION_KEY, counts, day=session.parts.spec.day)


def saved_commands(session: Any) -> list[Any]:
    """The loop's step E: every strategy's commands since the last call, as requests
    (`_worker_queue.Request`); a strategy that made any is saved first, and the session's
    counts after them (design rule 4). Raises JournalError when the journal cannot write
    them."""
    requests: list[Any] = []
    for trader in session.traders:
        made = trader.commands()
        if made:
            save(session, trader)
        requests += made
    if requests:
        save_session(session)
    return requests


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


def read_saved(state: Mapping[str, Any] | None, strategy: str, product: str) -> Saved | None:
    """A strategy's saved state from `rms_state` (None: nothing saved), read for the
    position in `product`, the strategy's. Raises LiveError for one that cannot be read,
    naming the strategy: a restart never guesses."""
    if state is None:
        return None
    try:
        live_state = dict(state["live_state"])
        orders = read_records(state["venue_map"])
        sent = {int(order["order"]): bool(order["sent"]) for order in live_state["orders"]}
        adopted = int(state.get("adopted", 0))
        bindings = {int(row["gtt"]): int(row["seq"]) for row in state.get(BINDINGS_KEY, [])}
    except (KeyError, TypeError, ValueError, LiveError) as error:
        raise LiveError(
            f"the saved state of {strategy} in the journal cannot be read ({error}): the "
            "restart cannot resume it. Check its positions in OpenAlgo, then start under "
            "another account name (account=) once they are closed"
        ) from None
    return Saved(live_state, orders, sent, adopted, product, bindings)


def another_instrument(
    saved: Saved | None, strategy: str, symbol: str, exchange: str
) -> str | None:
    """Why the restart refuses `strategy` when its saved state is of another instrument than
    the one the session trades for it, `symbol` on `exchange`, else None: the restart
    cannot resume that state, and never starts the strategy afresh in its place, since a
    saved state may hold working orders. A state is saved under its strategy class's name,
    so the text says to rename the class for a new instrument or contract, or to keep the
    saved symbol."""
    found = None if saved is None else saved.instrument
    if found is None or found == (symbol, exchange):
        return None
    saved_symbol, saved_exchange = found
    return (
        f"{strategy}: its saved state is of {saved_symbol} on {saved_exchange}, and this "
        f"session trades {symbol} on {exchange}: the restart cannot resume it, and does not "
        "start it afresh, since a saved state may hold working orders. For a new instrument "
        "or contract, give the strategy class a new name (its state is saved under its "
        f"name); to resume it, keep its symbol {saved_symbol} on {saved_exchange}. Check its "
        "positions and orders in OpenAlgo"
    )


__all__ = [
    "DAY_LOCKS",
    "LEG_STOP_ROLE",
    "Saved",
    "another_instrument",
    "read_saved",
    "save",
    "save_released",
    "save_session",
    "saved_commands",
]
