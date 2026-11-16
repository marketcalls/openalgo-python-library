"""The kill switch of a live session (16-session-design 5.4, 08-live-trading section 8;
unit 11): a strategy's entries locked for the day and its position exited, by the
quantity the engine and the broker agree on.

`KillSwitch.fire(trader, reason)`, `reason` one of `KILL_REASONS` (the kernel's exit
reason, and the name of the entry lock):

1. The strategy's entries are locked with `reason` (a latch: nothing in the session
   lifts it) and its working entry and add orders are cancelled.
2. Once those cancels have ended, or `KILL_CANCEL_WAIT_S` after the fire, the position
   is exited through the kernel's exit owner (`request_exit` with `reason`: a risk exit,
   which the exit retry schedule asks for again when it is rejected), but only when a
   position-book read taken since shows the broker holding at least the engine's units
   on the same side (`exit_agreed`). The engine's exit is of its whole position, so this
   is what keeps it from taking the broker's position through zero. With fewer units at
   the broker, none or the other side, or a book that cannot be read, nothing is sent:
   the book is read again every `KILL_READ_EVERY_S` (a book that lags a fill catches
   up), and from the second such read in a row an ERROR alert says so, once. The
   position is looked at again whenever it changes: its units, or its tranches (an
   entry whose cancel was refused fills after the exit, say, opening it again).
3. Once the engine is flat with nothing working, one more read, and an ERROR alert says
   whether the broker is flat too or what is left to close by hand.

After the mode guard locked, the kill switch cannot act (every order would go to the
other book, 16-session-design 6.2): it alerts and sends nothing. The risk watch fires it
for the account's daily loss limit; the reconciler never does (decided 2026-11-16: a
mismatch locks entries and alerts). Times are UTC nanoseconds, quantities units. Used by
the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._plan_catch_up import INSTRUMENT
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import LIVE_PRODUCT
from ._worker_queue import ReadPositions
from .inbox import PositionRows

logger = logging.getLogger(LOGGER_NAME)

#: The kill switch fired by the account's daily loss limit (16-session-design 6.3).
ACCOUNT_LOSS = "account_loss"
#: The kill switch fired for any other reason.
KILL_SWITCH = "kill_switch"
#: The reasons the kill switch takes: the kernel's exit reasons for it
#: (`RISK_EXIT_REASONS` of the binding).
KILL_REASONS = (ACCOUNT_LOSS, KILL_SWITCH)
#: Seconds the kill switch waits for the cancels of the strategy's entries before it
#: exits what the strategy holds: as the safe stop waits for its last words
#: (`_session_stop.SAFE_STOP_STATE_WAIT_S`).
KILL_CANCEL_WAIT_S = 5.0
#: Seconds between two position-book reads while one is needed and none came: the stops
#: supervisor's pace for a held cancel (16-session-design 4.11).
KILL_READ_EVERY_S = 2.0
#: Reads in a row that disagree with the engine before the kill switch alerts: as two
#: consecutive mismatches make a reconciliation mismatch (08-live-trading section 7).
KILL_DISAGREEMENTS_ALERTED = 2

_NS = NANOS_PER_SECOND


def exit_agreed(engine_units: float, broker_units: int | None) -> bool:
    """True when the engine's exit of its whole position (`engine_units`, signed) cannot
    take the broker's position (`broker_units`, signed; None for a book that could not be
    read) through zero: the broker holds at least as many units on the same side."""
    if broker_units is None or engine_units == 0:
        return False
    return engine_units * broker_units > 0 and abs(broker_units) >= abs(engine_units)


#: A position as the kill switch tells one from another: its signed units and the
#: (number, opening time) of each open tranche. Numbers start again with each position,
#: so a position closed and opened again is told by its tranches' opening times.
_Held = tuple[float, tuple[tuple[int, int], ...]]


@dataclass
class _Kill:
    """One fired strategy: why and when, the position its last exit request was for, the
    broker's units in the last read taken since the kill switch last asked (`read`), the
    reads in a row that disagreed, and what it has alerted."""

    reason: str
    fired_ns: int
    exit_for: _Held | None = None
    broker: int | None = None
    read: bool = False
    disagreements: int = 0
    alerted: bool = False
    reported: bool = False


class KillSwitch:
    """The kill switch of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._fired: dict[str, _Kill] = {}
        self._asked_ns: int | None = None
        self._locked_alerted = False

    def fired(self, name: str) -> str | None:
        """The reason the kill switch of strategy `name` was fired for, None if not."""
        kill = self._fired.get(name)
        return None if kill is None else kill.reason

    def fire(self, trader: Trader, reason: str) -> bool:
        """Fire the kill switch of `trader` for `reason` (module docstring); True when it
        was not fired before (a second fire changes nothing). Raises ConfigError for a
        reason not in `KILL_REASONS`."""
        if reason not in KILL_REASONS:
            raise ConfigError(f"the kill switch fires for one of {KILL_REASONS}, not {reason!r}")
        if trader.name in self._fired:
            return False
        session = self._session
        self._fired[trader.name] = _Kill(reason, session.now())
        if trader.lock(reason):
            session.event("lock", strategy=trader.name, reason=reason)
        for kernel_order in trader.entry_orders():
            trader.cancel(kernel_order)
        session.event("kill_switch", strategy=trader.name, reason=reason)
        logger.warning("%s: kill switch (%s)", trader.name, reason)
        return True

    def positions(self, message: PositionRows) -> None:
        """A position-book read: it answers the kill switch when it began after its last
        ask."""
        if self._asked_ns is None or message.read_ns < self._asked_ns:
            return
        for name, kill in self._fired.items():
            trader = self._session.trader_named(name)
            kill.broker = _net_units(message.positionbook, trader.symbol, trader.exchange)
            kill.read = True

    def duties(self, now_ns: int) -> None:
        """Step D: move each fired strategy on (module docstring), and ask for one
        position-book read when one of them needs it."""
        session = self._session
        if not self._fired:
            return
        if session.locked:
            self._cannot_act()
            return
        wants_read = False
        for name, kill in self._fired.items():
            trader = session.trader_named(name)
            if not trader.finished:
                wants_read |= self._move(trader, kill, now_ns)
        due = self._asked_ns is None or now_ns >= self._asked_ns + round(KILL_READ_EVERY_S * _NS)
        if wants_read and due:
            self._asked_ns = now_ns
            for kill in self._fired.values():
                kill.read = False
            session.hand_over([ReadPositions()])

    def _move(self, trader: Trader, kill: _Kill, now_ns: int) -> bool:
        """One strategy's step; True when it needs a position-book read."""
        held = _held(trader)
        units = held[0]
        if units == 0:
            kill.exit_for = None
            if trader.working() or kill.reported:
                return False
            if kill.read:
                self._report(trader, kill)
            return not kill.read
        kill.reported = False
        waiting = trader.entry_orders() and now_ns < kill.fired_ns + round(KILL_CANCEL_WAIT_S * _NS)
        if waiting or held == kill.exit_for:
            return False
        if not kill.read:
            return True
        kill.read = False
        if not exit_agreed(units, kill.broker):
            kill.disagreements += 1
            self._disagree(trader, kill, units)
            return True
        kill.disagreements = 0
        kill.exit_for = held
        what = trader.request_exit(kill.reason)
        self._session.event("kill_exit", strategy=trader.name, units=units, outcome=what)
        return False

    def _disagree(self, trader: Trader, kill: _Kill, units: float) -> None:
        self._session.event(
            "kill_disagrees", strategy=trader.name, units=units, broker_units=kill.broker
        )
        if kill.alerted or kill.disagreements < KILL_DISAGREEMENTS_ALERTED:
            return
        kill.alerted = True
        book = "cannot be read" if kill.broker is None else f"shows {kill.broker} units"
        self._session.alert(
            AlertKind.ERROR,
            f"kill switch ({kill.reason}): {trader.name} holds {units:g} units of "
            f"{trader.symbol} {trader.exchange} and OpenAlgo's position book {book}: no exit "
            "is sent until they agree. Check the position in OpenAlgo and close it by hand",
        )

    def _report(self, trader: Trader, kill: _Kill) -> None:
        kill.reported = True
        if kill.broker == 0:
            left = "flat at the broker too"
        elif kill.broker is None:
            left = "its position book cannot be read: check the position in OpenAlgo"
        else:
            left = f"OpenAlgo's position book still shows {kill.broker} units: close them by hand"
        self._session.alert(
            AlertKind.ERROR,
            f"kill switch ({kill.reason}): {trader.name} is flat in {trader.symbol} "
            f"{trader.exchange}, {left}; its entries stay locked for the day",
        )

    def _cannot_act(self) -> None:
        if self._locked_alerted:
            return
        self._locked_alerted = True
        names = ", ".join(self._fired)
        self._session.alert(
            AlertKind.ERROR,
            f"the kill switch of {names} cannot act: OpenAlgo's mode changed and every order "
            "would go to the other book. Close the positions by hand in OpenAlgo",
        )


def _held(trader: Trader) -> _Held:
    """The strategy's position (`_Held`): flat with no tranche when it never opened."""
    position = call(trader.engine.position, INSTRUMENT, LIVE_PRODUCT)
    if position is None:
        return 0.0, ()
    tranches = tuple(
        (int(number), int(opened_at)) for number, _, _, opened_at in position["tranches"]
    )
    return float(position["quantity_units"]), tranches


def _net_units(answer: Mapping[str, Any], symbol: str, exchange: str) -> int | None:
    """The broker's net MIS units of an instrument in a position-book answer, signed; None
    when the answer cannot be read."""
    rows = answer.get("data") if answer.get("status") == "success" else None
    if not isinstance(rows, list):
        return None
    net = 0
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if (row.get("symbol"), row.get("exchange"), row.get("product")) != (
            symbol,
            exchange,
            LIVE_PRODUCT,
        ):
            continue
        try:
            net += round(float(row.get("quantity", 0)))
        except (TypeError, ValueError):
            return None
    return net


__all__ = [
    "ACCOUNT_LOSS",
    "KILL_CANCEL_WAIT_S",
    "KILL_DISAGREEMENTS_ALERTED",
    "KILL_READ_EVERY_S",
    "KILL_REASONS",
    "KILL_SWITCH",
    "KillSwitch",
    "exit_agreed",
]
