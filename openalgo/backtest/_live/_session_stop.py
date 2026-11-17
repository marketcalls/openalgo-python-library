"""When a live session stops (16-session-design 3.3 to 3.5, 6.2; unit 8): the duties by
time that end it (`Control`). The common close is `_session_close`.

**Control** runs in the loop's step D once the session trades:

- **Ctrl+C, the safe stop (D29).** The first stop request locks every strategy's entries
  ("operator_stop") and cancels its working entries and adds (one decided in the same
  loop step is withdrawn before it is sent: `Trader.commands`). With `on_stop="flatten"`
  every open position is then exited (`request_exit`, "operator_stop"; the kernel answers
  "already_exiting" while an exit of it works, and a position whose exit failed every
  attempt of the retry schedule is asked once more). At each later step a position whose
  exit is not on the retry schedule (`_session_exits`) is asked again, so an entry that
  fills after the stop began is exited too. The session waits until every strategy is
  flat with nothing working, at most `SAFE_STOP_FLATTEN_S`; then each leg not flat is
  protected by state (`_stops_protect`: its broker-held stop placed again where it can
  be, and its exit asked for no more), and the session keeps running, with an ERROR
  alert and a console line naming each leg without a confirmed stop and what works
  toward its exit every `UNPROTECTED_REMINDER_EVERY_S`, until every leg is flat or
  protected, a second Ctrl+C, or the exchange's close plus `SESSION_END_GRACE_S`. For a
  leg whose exit failed every attempt the reminder names its stop and the orders working
  toward it, and says to end the session before any close by hand
  (`_stops_report.open_while_running`), since the engine may still send its exit or its
  stop. With `"keep"` every leg is protected by state at once, and the session waits at
  most `SAFE_STOP_STATE_WAIT_S` for the cancels and the stops' words (longer only to
  finish a step of protect by state's own, `Protection.may_end`), then leaves the
  positions open with their broker-held stops.
- **The second Ctrl+C** (3.3 step 4) asks for nothing more (no exit, no retry: the
  schedule is halted) and protects every leg by state. It waits at most
  `SAFE_STOP_STATE_WAIT_S` (bounded as with "keep") for each leg to be protected, and for
  the orders of a leg with no stop rule to end (with "flatten" every working order, the
  exits in flight among them; with "keep" the entries it cancelled), so a fill on the
  way is known, then prints one line per leg left open, naming its stop and an exit
  still in flight, and stops. The same lines end a stop with "keep" or a safe stop whose
  legs are all flat or protected; each is an ERROR alert too unless the leg's confirmed
  stop covers it with nothing else working (`_stops_report.protected`), which needs
  nothing of the operator. Two requests the loop sees at once act as the first
  followed by the second (decided for unit 10: how fast the keys were pressed must not
  change the outcome), so the flatten's exits are asked for and the second's bounded
  wait follows. After a guard lock nothing can be sent: a first Ctrl+C stops at once.
- **The end of the day.** Once every strategy is past its MIS square-off, flat and with
  nothing working, the session ends ("session_end"). From the square-off plus
  `SQUARE_OFF_VERIFY_AFTER_S` a strategy not flat gets the ERROR alert "still not flat
  after square-off", each minute while it lasts. A leg still open at the exchange's
  close plus `SESSION_END_GRACE_S` ends the session too, with an ERROR alert for every
  such leg, its stop or not. After a guard lock the session ends at that same time
  ("mode_breach").
- **A strategy's exception** (3.5): an ERROR alert, then the safe stop by `on_stop`. The
  operator's first Ctrl+C, before or after it, asks for the stop already under way: only
  a second one is the second Ctrl+C.
- The slices the plan's catch-up still owns are given up on at their last candle check.

The reminders and the lines of the legs left open are taken once the loop step that asked
for them has handed its commands to the order worker (`after_hand_over`, after the loop's
step E): an order the kernel made in that step (a target's exit, a stop placed again) is
then named as handed over, not as "not sent" beside "close it by hand".

Times are UTC nanoseconds, quantities units.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .._layout import NANOS_PER_SECOND
from ._session_loop import Session
from ._session_trader import StrategyFailed, Trader
from ._stops_protect import PROTECTED
from ._stops_report import left_open, open_while_running, protected
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the safe stop waits for every leg to be flat before it reminds (3.3 step 3).
SAFE_STOP_FLATTEN_S = 60.0
#: Seconds a stop with `on_stop="keep"`, or a second Ctrl+C, waits for the last words.
SAFE_STOP_STATE_WAIT_S = 5.0
#: Seconds between the reminders that a leg is still open during the safe stop.
UNPROTECTED_REMINDER_EVERY_S = 60.0
#: Seconds after the exchange's close at which the session ends whatever is open (3.4).
SESSION_END_GRACE_S = 300.0
#: Seconds after the square-off at which a strategy not flat is alerted, then each minute.
SQUARE_OFF_VERIFY_AFTER_S = 30.0
NOT_FLAT_REPEAT_S = 60.0
#: The entry lock, and the exit reason, of the safe stop (the kernel's RISK_EXIT_REASONS).
OPERATOR_STOP = "operator_stop"
#: The stop requests that make the second Ctrl+C (3.3 step 4).
SECOND_CTRL_C = 2
#: How a session ended (1.4 `stopped_by`).
SESSION_END, CTRL_C, MODE_BREACH, ERROR = "session_end", "ctrl_c", "mode_breach", "error"
#: What `request_exit` answers when it made or readied an exit (its EXIT_REQUESTS).
_EXIT_MADE = frozenset({"sent", "waits_for_cancels"})
#: What it answers while an exit of the position waits to be sent: it goes once the
#: orders on the position's closing side have ended.
_EXIT_WAITS = frozenset({"waits_for_cancels", "already_exiting"})

_NS = NANOS_PER_SECOND


class Control:
    """The duties by time that stop `session` (module docstring), by its `on_stop`
    ("flatten" or "keep"). `verdict` is set when the session should end."""

    def __init__(self, session: Session, on_stop: str) -> None:
        self._session = session
        self._on_stop = on_stop
        self.verdict: str | None = None
        self.failure: StrategyFailed | None = None
        self._stopping_ns: int | None = None
        self._second_ns: int | None = None
        self._remind_ns: int | None = None
        self._protecting = False
        # The lines due after the step's hand-over: the reminders, and the legs left open
        # (the value is `every_leg`).
        self._remind_due = False
        self._left_open_due: bool | None = None
        # The kernel's last answer to the safe stop's exit request, by strategy.
        self._asked: dict[str, str | None] = {}
        self._not_flat_ns = {
            t.name: t.square_off_ns + _s(SQUARE_OFF_VERIFY_AFTER_S) for t in session.traders
        }
        self._end_ns = max(t.close_ns for t in session.traders) + _s(SESSION_END_GRACE_S)

    @property
    def stopping(self) -> bool:
        """True once a safe stop has begun."""
        return self._stopping_ns is not None

    def strategy_failed(self, failed: StrategyFailed) -> None:
        """A strategy's code raised (3.5): an ERROR alert, then the safe stop (module
        docstring)."""
        if self.failure is None:
            self.failure = failed
        self._session.alert(
            AlertKind.ERROR,
            f"{failed.strategy} raised {type(failed.error).__name__}: {failed.error}. The "
            "session stops safely",
        )
        if not self.stopping:
            self._begin(self._session.now())

    def duties(self, now_ns: int) -> None:
        """Step D's stop and end-of-day duties."""
        session = self._session
        for trader in session.traders:
            for note in trader.candles.give_up(now_ns):
                session.warn(note)
        if self.verdict is None and session.trading:
            self._take_requests(now_ns)
        session.exits.duties(now_ns)
        if self.verdict is not None or not session.trading:
            return
        if self._second_ns is not None:
            self._second_due(now_ns)
        elif self.stopping:
            self._stop_due(now_ns)
        else:
            self._end_of_day(now_ns)

    def after_hand_over(self) -> None:
        """The lines the step's duties asked for, once the step has handed its commands to
        the order worker (module docstring): the reminders, then the legs left open."""
        if self._remind_due:
            self._remind_due = False
            self._remind()
        every_leg, self._left_open_due = self._left_open_due, None
        if every_leg is not None:
            self._left_open_lines(every_leg=every_leg)

    # ----- the safe stop --------------------------------------------------------------

    def _take_requests(self, now_ns: int) -> None:
        """The operator's stop requests: the first begins the safe stop, the second ends
        it; two seen at once act as the first followed by the second (module docstring).
        The retry schedule is halted before its duties run, so no retry is asked for in
        the step that saw the second."""
        session = self._session
        requested = session.stop.count
        if requested and not self.stopping:
            self._begin(now_ns)
        if self.verdict is None and requested >= SECOND_CTRL_C and self._second_ns is None:
            self._second_ns = now_ns
            session.exits.halt()
            session.event("second_stop")
            self._protect(session.traders, now_ns)

    def _protect(self, traders: Sequence[Trader], now_ns: int) -> None:
        """Protect `traders`' legs by state from now (3.3)."""
        self._protecting = True
        self._session.stops.protection.begin(traders, now_ns)

    def _begin(self, now_ns: int) -> None:
        session = self._session
        self._stopping_ns = now_ns
        self._remind_ns = now_ns + _s(SAFE_STOP_FLATTEN_S)
        session.deciding = False
        session.event("safe_stop", on_stop=self._on_stop)
        for trader in session.traders:
            trader.lock(OPERATOR_STOP)
            for kernel_order in trader.entry_orders():
                trader.cancel(kernel_order)
        if session.locked:
            self.verdict = MODE_BREACH
            return
        if self._on_stop == "flatten":
            self._ask_exits(retried_too=True)
        else:
            self._protect(session.traders, now_ns)

    def _ask_exits(self, *, retried_too: bool) -> None:
        """Ask the kernel to exit each open position. `retried_too` False skips one whose
        exit is on the retry schedule, which asks by its own times and gives up after its
        last attempt; the stop's first request asks for every position."""
        session = self._session
        for trader in session.traders:
            if trader.open_units() == 0 or session.stops.protection.holds(trader):
                continue
            if not retried_too and session.exits.owns(trader.name):
                continue
            what = trader.request_exit(OPERATOR_STOP)
            self._asked[trader.name] = what
            if what in _EXIT_MADE:
                session.event("exit_requested", strategy=trader.name, outcome=what)

    def _stop_due(self, now_ns: int) -> None:
        traders = self._session.traders
        if self._on_stop == "keep":
            if self._wait_over(now_ns, self._stopping_ns):
                self._left_open_due = False
                self.verdict = self._stopped_by()
            return
        if all(trader.flat or self._protected(trader) for trader in traders):
            self._left_open_due = False
            self.verdict = self._stopped_by()
            return
        if not self._session.locked:
            self._ask_exits(retried_too=False)
        if now_ns >= self._end_ns:
            self._left_open_due = True
            self.verdict = self._stopped_by()
        elif now_ns >= self._remind_ns:
            if not self._protecting:
                self._protect([trader for trader in traders if not trader.flat], now_ns)
            self._remind_ns = now_ns + _s(UNPROTECTED_REMINDER_EVERY_S)
            self._remind_due = True

    def _second_due(self, now_ns: int) -> None:
        """After the second Ctrl+C: once every leg is protected by state and the orders
        the stop waits on have ended, or `SAFE_STOP_STATE_WAIT_S` after it, list each leg
        left open and stop."""
        if not self._wait_over(now_ns, self._second_ns):
            return
        self._left_open_due = False
        self.verdict = self._stopped_by()

    def _wait_over(self, now_ns: int, since_ns: int) -> bool:
        """Whether the bounded wait begun at `since_ns` is over: nothing waits any more,
        or protect by state lets it end (`Protection.may_end`)."""
        session = self._session
        if not any(self._waiting(trader) for trader in session.traders):
            return True
        until_ns = since_ns + _s(SAFE_STOP_STATE_WAIT_S)
        return session.stops.protection.may_end(now_ns, until_ns)

    def _protected(self, trader: Trader) -> bool:
        """True once protect by state holds the leg and its stop covers it."""
        protection = self._session.stops.protection
        return protection.holds(trader) and protection.state(trader) == PROTECTED

    def _waiting(self, trader: Trader) -> bool:
        """Whether the stop still waits on `trader`'s leg: protect by state is not done,
        or orders of a leg it does not protect work on (with "flatten" every working
        order, the exits and the entries the stop cancels; with "keep" the entries it
        cancels, since the strategy's resting exits stay)."""
        protection = self._session.stops.protection
        if not protection.settled(trader):
            return True
        if self._on_stop == "keep":
            return bool(trader.entry_orders())
        return not self._protected(trader) and bool(trader.working())

    def _remind(self) -> None:
        session = self._session
        for trader in session.traders:
            units = trader.open_units()
            if units == 0 or self._protected(trader):
                continue
            line = f"{_leg(trader, units)}: not flat yet, {self._exit_state(trader)}"
            session.emit(line)
            session.alert(AlertKind.ERROR, f"safe stop: {line}")

    def _exit_state(self, trader: Trader) -> str:
        """What works toward the exit of `trader`'s position, as a reminder says it."""
        session = self._session
        stop = session.stops.leg(trader).stop
        closing = [kernel_order for kernel_order in trader.exit_orders() if kernel_order != stop]
        exits = [trader.order_name(kernel_order) for kernel_order in closing]
        if exits:
            return f"order {', '.join(exits)} working to close it"
        if trader.finished:
            return "the exchange's close ended its engine session, close it by hand"
        if session.locked:
            return "nothing can be sent after OpenAlgo's mode changed"
        if session.exits.retry_due(trader.name):
            return "its rejected exit is asked for again shortly"
        if session.exits.owns(trader.name):
            # Its stop was placed again (`_session_exits`): what to do follows from what
            # works at the broker, never a sale by hand beside a working stop (3.5) or
            # the running session, which may send the exit or the stop again (4.8, 4.10).
            note = open_while_running(session.stops, trader)
            return f"its exit failed every attempt; {note}"
        asked = self._asked.get(trader.name)
        # With no exit order working, the working orders left are entries: those on the
        # position's closing side are what a waiting exit waits for.
        cancels = [trader.order_name(kernel_order) for kernel_order in trader.entry_orders()]
        if cancels and asked in _EXIT_WAITS:
            return f"its exit waits for the cancel of order {', '.join(cancels)}"
        return f"no exit works (the engine answered {asked})"

    def _left_open_lines(self, *, every_leg: bool) -> None:
        """One console line per leg left open (3.3 step 4). An ERROR alert goes for each
        leg the operator must act on: one without a confirmed stop covering it, or with
        another order working; a leg its stop protects is information only. At the
        exchange's close (`every_leg`) a position still open is the operator's whatever
        its stop (3.4)."""
        session = self._session
        for trader in session.traders:
            units = trader.open_units()
            if units == 0:
                continue
            line = f"{_leg(trader, units)}: {left_open(session.stops, trader)}"
            session.emit(line)
            if every_leg or not protected(session.stops, trader):
                session.alert(AlertKind.ERROR, line)
            else:
                logger.info(line)

    def _stopped_by(self) -> str:
        if self.failure is not None:
            return ERROR
        return MODE_BREACH if self._session.locked else CTRL_C

    # ----- the end of the day ---------------------------------------------------------

    def _end_of_day(self, now_ns: int) -> None:
        session = self._session
        if session.locked:
            if now_ns >= self._end_ns:
                self.verdict = MODE_BREACH
            return
        traders = session.traders
        if all(now_ns >= t.square_off_ns and t.flat for t in traders):
            self.verdict = SESSION_END
            return
        if now_ns >= self._end_ns:
            self._left_open_due = True
            self.verdict = SESSION_END
            return
        for trader in traders:
            if now_ns >= self._not_flat_ns[trader.name] and not trader.flat:
                self._not_flat_ns[trader.name] = now_ns + _s(NOT_FLAT_REPEAT_S)
                self._not_flat(trader)

    def _not_flat(self, trader: Trader) -> None:
        exits = [trader.order_name(kernel_order) for kernel_order in trader.working()]
        self._session.alert(
            AlertKind.ERROR,
            f"still not flat after square-off: {trader.name} {trader.symbol} "
            f"{trader.exchange} {trader.open_units():g} units, working orders "
            f"{', '.join(exits) or 'none'}",
        )


def _leg(trader: Trader, units: float) -> str:
    """How a line names a strategy's leg: the strategy, the instrument, signed units."""
    return f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units"


def _s(seconds: float) -> int:
    return round(seconds * _NS)


__all__ = [
    "CTRL_C",
    "ERROR",
    "MODE_BREACH",
    "NOT_FLAT_REPEAT_S",
    "OPERATOR_STOP",
    "SAFE_STOP_FLATTEN_S",
    "SAFE_STOP_STATE_WAIT_S",
    "SECOND_CTRL_C",
    "SESSION_END",
    "SESSION_END_GRACE_S",
    "SQUARE_OFF_VERIFY_AFTER_S",
    "UNPROTECTED_REMINDER_EVERY_S",
    "Control",
]
