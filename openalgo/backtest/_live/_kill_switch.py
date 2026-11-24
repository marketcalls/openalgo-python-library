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
   a row an ERROR alert says so, with what the session's other exits will do, once for each
   position, and the strategy's book is then read every `KILL_READ_BACKOFF_S`. The event
   stream notes a disagreement when the broker's figure changes. The position is looked
   at again whenever it changes (an entry whose cancel was refused fills after the exit,
   say).
3. Once the engine is flat with nothing working, an ERROR alert reports it: at the first
   read showing the broker flat too; when the book shows other units, or cannot be read,
   only once two reads in a row show the same (the units left to close by hand).

Which read counts for a strategy, and how often one is asked for, is `_kill_reads`'s.

**A shared instrument** (18-multi-strategy 4.9): the position book shows the net of every
strategy of the session that trades the instrument, so a member's exit is agreed only when
the broker holds at least their sum on its side (the group agreeing, or only an excess),
and a flat member is flat at the broker when the book shows what the others hold. A group
difference sends nothing: the reconciler's comparison of the group locks and alerts, and
shares out a difference other orders' trades explain (4.7), after which a read agrees and
each member's exit sells what it holds.

**No exit beyond the broker's units.** When the broker holds fewer units than the engine on
the same side, the reconciler's adoption of an explained difference (16-session-design
5.3) makes the two agree and the next read exits; an unexplained one is held by the
reconciler's exit guard, so no other owner (the safe stop's flatten, the MIS square-off,
a stop placed again) sends more than the broker holds either (5.4, "As built").

After the mode guard locked, the kill switch cannot act (every order would go to the
other book, 16-session-design 6.2): it alerts and sends nothing. The risk watch fires it
for the account's daily loss limit; the reconciler never does (decided 2026-11-16: a
mismatch locks entries and alerts): its interface to the kill switch is `fire`, `fired` and
`positions`, and every read it makes counts here. Times are UTC nanoseconds, quantities
units. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._series import ist_time
from ..errors import ConfigError
from ._broker_units import broker_units
from ._group_rules import and_join
from ._kill_reads import (
    KILL_READ_BACKOFF_S,
    KILL_READ_EVERY_S,
    KILL_READ_LOST_AFTER_S,
    Held,
    Kill,
    ReadPace,
    State,
    judged,
    look,
    seconds_ns,
)
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind
from ._worker_queue import ReadPositions
from .inbox import PositionRows

logger = logging.getLogger(LOGGER_NAME)

#: The kill switch fired by the account's daily loss limit (16-session-design 6.3).
ACCOUNT_LOSS = "account_loss"
#: The kill switch fired by the account's daily profit target (18-multi-strategy 2.5a).
ACCOUNT_PROFIT = "account_profit"
#: The reasons the kill switch takes, each one of the kernel's exit reasons
#: (`RISK_EXIT_REASONS` of the binding): the account's daily rules are its only callers.
KILL_REASONS = (ACCOUNT_LOSS, ACCOUNT_PROFIT)
#: Seconds the kill switch waits for the cancels of the strategy's entries before it
#: exits what the strategy holds: as the safe stop waits for its last words
#: (`_session_stop.SAFE_STOP_STATE_WAIT_S`).
KILL_CANCEL_WAIT_S = 5.0
#: Reads in a row that disagree with the engine before the kill switch alerts, or reports
#: units left at the broker: as two consecutive mismatches make a reconciliation mismatch
#: (08-live-trading section 7).
KILL_DISAGREEMENTS_ALERTED = 2


def exit_agreed(engine_units: float, broker_units: int | None) -> bool:
    """True when the engine's exit of its whole position (`engine_units`, signed) cannot
    take the broker's position (`broker_units`, signed; None for a book that could not be
    read) through zero: the broker holds at least as many units on the same side."""
    if broker_units is None or engine_units == 0:
        return False
    return engine_units * broker_units > 0 and abs(broker_units) >= abs(engine_units)


class KillSwitch:
    """The kill switch of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._fired: dict[str, Kill] = {}
        self._reads = ReadPace()
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
        self._fired[trader.name] = Kill(reason, session.now())
        if trader.lock(reason):
            session.event("lock", strategy=trader.name, reason=reason)
        for kernel_order in trader.entry_orders():
            trader.cancel(kernel_order)
        session.event("kill_switch", strategy=trader.name, reason=reason)
        logger.warning("%s: kill switch (%s)", trader.name, reason)
        return True

    def positions(self, message: PositionRows) -> None:
        """A position-book read, whoever asked for it: it counts for each fired strategy
        whose state began at or before the read did (module docstring), its broker's units
        those of its leg in its product (`_broker_units`)."""
        self._reads.came(message.read_ns)
        for name, kill in self._fired.items():
            if message.read_ns < kill.since_ns:
                continue
            trader = self._session.trader_named(name)
            kill.broker = broker_units(
                message.positionbook,
                message.holdings,
                trader.symbol,
                trader.exchange,
                trader.product,
            )
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
            state = look(trader, kill, now_ns)
            wanted |= self._move(trader, kill, state, now_ns) and now_ns >= kill.quiet_until_ns
        if wanted and self._reads.due(now_ns):
            self._reads.ask(now_ns)
            session.hand_over([ReadPositions()])

    def _move(self, trader: Trader, kill: Kill, state: State, now_ns: int) -> bool:
        """One strategy's step in its `state`; True when it needs a position-book read."""
        held, working = state
        units = held[0]
        if units == 0:
            kill.exit_for = None
            if working or kill.reported:
                return False
            return kill.read_ns is None or self._flat(trader, kill, kill.read_ns)
        kill.reported = False
        waiting = trader.entry_orders() and now_ns < kill.fired_ns + seconds_ns(KILL_CANCEL_WAIT_S)
        if waiting or held == kill.exit_for:
            return False
        if kill.read_ns is None:
            return True
        judged(kill, kill.read_ns)
        if not exit_agreed(units + self._others(trader), kill.broker):
            self._disagree(trader, kill, held, now_ns)
            return True
        kill.exit_for = held
        what = self._session.margin.request_exit(trader, kill.reason)
        self._session.event("kill_exit", strategy=trader.name, units=units, outcome=what)
        return False

    def _disagree(self, trader: Trader, kill: Kill, held: Held, now_ns: int) -> None:
        kill.disagreements += 1
        self._note(trader, kill, held[0])
        if kill.disagreements < KILL_DISAGREEMENTS_ALERTED:
            return
        kill.quiet_until_ns = now_ns + seconds_ns(KILL_READ_BACKOFF_S)
        # Once per position of the engine: a new one (an add or a fill) changes the units
        # and what a Ctrl+C or the square-off would sell, so it is alerted again.
        if kill.alerted_for != held:
            kill.alerted_for = held
            self._session.alert(AlertKind.ERROR, self._disagreement_words(trader, kill, held[0]))

    def _flat(self, trader: Trader, kill: Kill, read_ns: int) -> bool:
        """The flat strategy's read, begun at `read_ns`: reported when the broker is flat
        too, or once `KILL_DISAGREEMENTS_ALERTED` reads in a row show the same other figure
        (a book that lags the exit's fill catches up). True while another read is needed."""
        same = kill.disagreements > 0 and kill.broker == kill.noted
        judged(kill, read_ns)
        if kill.broker != self._others(trader):
            kill.disagreements = kill.disagreements + 1 if same else 1
            self._note(trader, kill, 0.0)
            if kill.disagreements < KILL_DISAGREEMENTS_ALERTED:
                return True
        self._report(trader, kill)
        return False

    def _others(self, trader: Trader) -> int:
        """The signed units the other strategies of `trader`'s instrument hold now (none
        for a strategy alone on it): the broker's net counts them too."""
        members = self._session.groups.members(trader)
        return sum(round(other.open_units()) for other in members if other is not trader)

    def _note(self, trader: Trader, kill: Kill, units: float) -> None:
        """Write a disagreement to the event stream: the first of a run, then each change of
        the broker's figure."""
        if kill.disagreements == 1 or kill.broker != kill.noted:
            kill.noted = kill.broker
            self._session.event(
                "kill_disagrees", strategy=trader.name, units=units, broker_units=kill.broker
            )

    def _disagreement_words(self, trader: Trader, kill: Kill, units: float) -> str:
        """The ERROR alert of a lasting disagreement: what the books say, and what the
        session's other exits will do (module docstring): with a book that cannot be read
        the engine still counts its own units; with one that shows fewer, no exit is sent
        beyond them (the reconciler's exit guard), and an explained difference is adopted
        (on a shared instrument shared out among the members, 18-multi-strategy 4.7)."""
        book = "cannot be read" if kill.broker is None else f"shows {kill.broker} units"
        others = self._others(trader)
        held = f"{units:g} units of {trader.symbol} {trader.exchange}"
        taken = "taken as the engine's"
        if self._session.groups.shared(trader):
            names = and_join(self._other_names(trader))
            held += f", {units + others:g} with {names}'s,"
            everyone = and_join([member.name for member in self._session.groups.members(trader)])
            taken = (
                f"shared out among {everyone}, the most recently opened position first, each "
                "taking its share as its own"
            )
        exits = f"the square-off at {ist_time(trader.square_off_ns):%H:%M} IST"
        if self._session.parts.spec.arguments.on_stop == "flatten":
            exits = f'a Ctrl+C (on_stop="flatten") and {exits}'
        side = "sell" if units > 0 else "buy"
        then = (
            f"{exits} would still {side} all {abs(units):g} units"
            if kill.broker is None
            else f"no exit of the session ({exits}) is sent for more than the broker holds: "
            f"trades of other orders that explain the difference are {taken}, else the exits "
            "wait for the books to agree"
        )
        return (
            f"kill switch ({kill.reason}): {trader.name} holds {held} and OpenAlgo's position "
            f"book {book}: no exit is sent until they agree, and {then}. Check the position "
            "in OpenAlgo"
        )

    def _report(self, trader: Trader, kill: Kill) -> None:
        """The ERROR alert of a flat strategy (module docstring); on a shared instrument
        the book's units are weighed against what the other strategies hold."""
        kill.reported = True
        others = self._others(trader)
        if kill.broker == 0 and others == 0:
            left = "flat at the broker too"
        elif kill.broker is None:
            left = "its position book cannot be read: check the position in OpenAlgo"
        elif kill.broker == others:
            names = and_join(self._other_names(trader))
            left = f"and OpenAlgo's position book shows only the {others} units of {names}"
        elif self._session.groups.shared(trader):
            names = and_join(self._other_names(trader))
            left = (
                f"OpenAlgo's position book still shows {kill.broker} units, {others} of them "
                f"{names}'s: close the rest by hand"
            )
        else:
            left = f"OpenAlgo's position book still shows {kill.broker} units: close them by hand"
        self._session.alert(
            AlertKind.ERROR,
            f"kill switch ({kill.reason}): {trader.name} is flat in {trader.symbol} "
            f"{trader.exchange}, {left}; its entries stay locked for the day",
        )

    def _other_names(self, trader: Trader) -> list[str]:
        """The other strategies of `trader`'s instrument, in call order."""
        return [m.name for m in self._session.groups.members(trader) if m is not trader]

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


__all__ = [
    "ACCOUNT_LOSS",
    "ACCOUNT_PROFIT",
    "KILL_CANCEL_WAIT_S",
    "KILL_DISAGREEMENTS_ALERTED",
    "KILL_READ_BACKOFF_S",
    "KILL_READ_EVERY_S",
    "KILL_READ_LOST_AFTER_S",
    "KILL_REASONS",
    "KillSwitch",
    "exit_agreed",
]
