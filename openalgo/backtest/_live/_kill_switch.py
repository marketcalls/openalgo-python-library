"""The kill switch of a live session (16-session-design 5.4, 08-live-trading section 8;
unit 11): a strategy's entries locked for the day and its position exited, when the
engine and the broker agree on it.

`KillSwitch.fire(trader, reason)`, `reason` one of `KILL_REASONS` (the kernel's exit
reason, and the name of the entry lock):

1. The strategy's entries are locked with `reason` (a latch: nothing in the session
   lifts it) and its working entry and add orders are cancelled.
2. Once those cancels have ended, or `KILL_CANCEL_WAIT_S` after the fire, the position
   is exited through the kernel's exit owner (`request_exit` with `reason`: a risk exit,
   which the exit retry schedule asks for again when it is rejected), but only when a
   position-book read shows the broker holding at least the engine's units on the same
   side (`exit_agreed`). The engine's exit is of its whole position, so this is what
   keeps it from taking the broker's position through zero. With fewer units at the
   broker, none or the other side, or a book that cannot be read, nothing is sent and the
   book is read again (a book that lags a fill catches up); from the second such read in
   a row an ERROR alert says so, once, with what the session will still exit, and the
   strategy's book is then read every `KILL_READ_BACKOFF_S`. The event stream notes a
   disagreement when the broker's figure changes. The position is looked at again
   whenever it changes (an entry whose cancel was refused fills after the exit, say).
3. Once the engine is flat with nothing working, one more read, and an ERROR alert says
   whether the broker is flat too or what is left to close by hand.

**Which read counts.** A strategy's state is its position (units and tranches) and its
working orders. A read counts for it only when the read began at or after its state, as
the kill switch last saw it, began: the fire, a fill, its cancels ending, its exit
filling. Each read is judged once: the next judgement takes a read begun after it. Any
reader's read counts (the stops supervisor reads the book too).

**Asking for reads.** At most one at a time: the next is asked for once a read begun at
or after the last ask has come and `KILL_READ_EVERY_S` have passed since that ask. A read
the worker never answers is asked for again after `KILL_READ_LOST_AFTER_S`.

**Known gap until the reconciler (unit 12).** When the broker holds fewer units than the
engine, the engine still counts its own: the safe stop with `on_stop="flatten"` and the
MIS square-off exit all of them, which takes the broker's position through zero. The
disagreement alert says so. Adopting the broker's figure, when external trades explain
it, is the reconciler's (16-session-design 5.3).

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
from .._series import ist_time
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
#: The reasons the kill switch takes, each one of the kernel's exit reasons
#: (`RISK_EXIT_REASONS` of the binding): the account's daily loss limit is its only caller.
KILL_REASONS = (ACCOUNT_LOSS,)
#: Seconds the kill switch waits for the cancels of the strategy's entries before it
#: exits what the strategy holds: as the safe stop waits for its last words
#: (`_session_stop.SAFE_STOP_STATE_WAIT_S`).
KILL_CANCEL_WAIT_S = 5.0
#: Seconds between two asks for a position-book read: the stops supervisor's pace for a
#: held cancel (16-session-design 4.11).
KILL_READ_EVERY_S = 2.0
#: Seconds between two reads for a strategy whose disagreement was alerted: the user has
#: been told, and a book that stays apart is not read every few seconds all day.
KILL_READ_BACKOFF_S = 30.0
#: Seconds after which a read asked for and never answered counts as lost and is asked
#: for again: well past the orders client's own wait, the SDK's 120 s (`clients`), and a
#: call ahead of it on the worker. The worker answers every read (or the session stops),
#: so this only keeps a lost answer from silencing the kill switch.
KILL_READ_LOST_AFTER_S = 300.0
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
#: A strategy's state as the kill switch watches it: its position and its working orders.
_State = tuple[_Held, tuple[int, ...]]


@dataclass
class _Kill:
    """One fired strategy: why and when; its `state` as last seen and the instant from
    which a read counts for it (`since_ns`); the broker's units in the read that counts
    and is not judged yet, and when that read began (None for none); the position its last
    exit request was for; the reads in a row that disagreed and the broker's figure the
    event stream last noted; the instant before which no read is asked for it after its
    alert; what it has alerted."""

    reason: str
    fired_ns: int
    state: _State | None = None
    since_ns: int = 0
    broker: int | None = None
    read_ns: int | None = None
    exit_for: _Held | None = None
    disagreements: int = 0
    noted: int | None = None
    quiet_until_ns: int = 0
    alerted: bool = False
    reported: bool = False


@dataclass
class _ReadPace:
    """The kill switch's asks for a position-book read (module docstring): when it last
    asked, and whether a read begun since has come."""

    asked_ns: int | None = None
    answered: bool = True

    def due(self, now_ns: int) -> bool:
        """True when another read may be asked for at `now_ns`."""
        if self.asked_ns is None:
            return True
        wait_s = KILL_READ_EVERY_S if self.answered else KILL_READ_LOST_AFTER_S
        return now_ns >= self.asked_ns + _s(wait_s)

    def ask(self, now_ns: int) -> None:
        """A read was asked for at `now_ns`."""
        self.asked_ns, self.answered = now_ns, False

    def came(self, read_ns: int) -> None:
        """A read begun at `read_ns` came: it answers an ask made at or before then."""
        if self.asked_ns is not None and read_ns >= self.asked_ns:
            self.answered = True


class KillSwitch:
    """The kill switch of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._fired: dict[str, _Kill] = {}
        self._reads = _ReadPace()
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
        """A position-book read, whoever asked for it: it counts for each fired strategy
        whose state began at or before the read did (module docstring)."""
        self._reads.came(message.read_ns)
        for name, kill in self._fired.items():
            if message.read_ns < kill.since_ns:
                continue
            trader = self._session.trader_named(name)
            kill.broker = _net_units(message.positionbook, trader.symbol, trader.exchange)
            kill.read_ns = message.read_ns

    def duties(self, now_ns: int) -> None:
        """Step D: move each fired strategy on (module docstring), and ask for one
        position-book read when one of them needs it."""
        session = self._session
        if not self._fired:
            return
        if session.locked:
            self._cannot_act()
            return
        wanted = False
        for name, kill in self._fired.items():
            trader = session.trader_named(name)
            if trader.finished:
                continue
            state = _look(trader, kill, now_ns)
            wanted |= self._move(trader, kill, state, now_ns) and now_ns >= kill.quiet_until_ns
        if wanted and self._reads.due(now_ns):
            self._reads.ask(now_ns)
            session.hand_over([ReadPositions()])

    def _move(self, trader: Trader, kill: _Kill, state: _State, now_ns: int) -> bool:
        """One strategy's step in its `state`; True when it needs a position-book read."""
        held, working = state
        units = held[0]
        if units == 0:
            kill.exit_for = None
            if working or kill.reported:
                return False
            if kill.read_ns is None:
                return True
            self._report(trader, kill)
            return False
        kill.reported = False
        waiting = trader.entry_orders() and now_ns < kill.fired_ns + _s(KILL_CANCEL_WAIT_S)
        if waiting or held == kill.exit_for:
            return False
        if kill.read_ns is None:
            return True
        # Judged once: the next judgement takes a read begun after this one.
        kill.since_ns, kill.read_ns = kill.read_ns + 1, None
        if not exit_agreed(units, kill.broker):
            self._disagree(trader, kill, units, now_ns)
            return True
        kill.disagreements = 0
        kill.exit_for = held
        what = trader.request_exit(kill.reason)
        self._session.event("kill_exit", strategy=trader.name, units=units, outcome=what)
        return False

    def _disagree(self, trader: Trader, kill: _Kill, units: float, now_ns: int) -> None:
        kill.disagreements += 1
        if kill.disagreements == 1 or kill.broker != kill.noted:
            kill.noted = kill.broker
            self._session.event(
                "kill_disagrees", strategy=trader.name, units=units, broker_units=kill.broker
            )
        if kill.disagreements < KILL_DISAGREEMENTS_ALERTED:
            return
        kill.quiet_until_ns = now_ns + _s(KILL_READ_BACKOFF_S)
        if not kill.alerted:
            kill.alerted = True
            self._session.alert(AlertKind.ERROR, self._disagreement_words(trader, kill, units))

    def _disagreement_words(self, trader: Trader, kill: _Kill, units: float) -> str:
        """The ERROR alert of a lasting disagreement: what the books say, and what the
        session will still exit of the engine's units (module docstring)."""
        book = "cannot be read" if kill.broker is None else f"shows {kill.broker} units"
        exits = f"the square-off at {ist_time(trader.square_off_ns):%H:%M} IST"
        if self._session.parts.spec.arguments.on_stop == "flatten":
            exits = f'a Ctrl+C (on_stop="flatten") and {exits}'
        side = "sell" if units > 0 else "buy"
        left = "" if kill.broker is None else f", leaving {kill.broker - units:g} at the broker"
        return (
            f"kill switch ({kill.reason}): {trader.name} holds {units:g} units of "
            f"{trader.symbol} {trader.exchange} and OpenAlgo's position book {book}: no exit "
            f"is sent until they agree. The engine still counts its units: {exits} would "
            f"{side} all {abs(units):g} units{left}. Check the position in OpenAlgo"
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


def _look(trader: Trader, kill: _Kill, now_ns: int) -> _State:
    """The strategy's state at `now_ns`, noted: a new one restarts what counts as its
    read, and is looked at without the back-off of an earlier alert."""
    state = (_held(trader), tuple(sorted(trader.working())))
    if state != kill.state:
        kill.state, kill.since_ns = state, now_ns
        kill.broker = kill.read_ns = None
        kill.quiet_until_ns = 0
    return state


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


def _s(seconds: float) -> int:
    return round(seconds * _NS)


__all__ = [
    "ACCOUNT_LOSS",
    "KILL_CANCEL_WAIT_S",
    "KILL_DISAGREEMENTS_ALERTED",
    "KILL_READ_BACKOFF_S",
    "KILL_READ_EVERY_S",
    "KILL_READ_LOST_AFTER_S",
    "KILL_REASONS",
    "KillSwitch",
    "exit_agreed",
]
