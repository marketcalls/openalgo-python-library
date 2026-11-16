"""When a live session stops (16-session-design 3.3 to 3.5, 6.2; unit 8): the duties by
time that end it (`Control`). The common close is `_session_close`.

**Control** runs in the loop's step D once the session trades:

- **Ctrl+C, the safe stop (D29).** The first stop request locks every strategy's entries
  ("operator_stop") and cancels its working entries and adds (one decided in the same
  loop step is withdrawn before it is sent: `Trader.commands`). With `on_stop="flatten"`
  each open position is then exited (`request_exit`, "operator_stop"), asked again at
  each step for a position whose exit is not on the retry schedule (`_session_exits`),
  so an entry that fills after the stop began is exited too (the kernel answers
  "already_exiting" while its exit works); the session waits until every strategy is
  flat with nothing working, at most `SAFE_STOP_FLATTEN_S`; after that it keeps running,
  with an ERROR alert and a console line naming each leg and what works toward its exit
  every `UNPROTECTED_REMINDER_EVERY_S`, until flat, a second Ctrl+C, or the exchange's
  close plus `SESSION_END_GRACE_S`. With `"keep"` the session waits at most
  `SAFE_STOP_STATE_WAIT_S` for the cancels and leaves the positions open. A second
  Ctrl+C prints one line per leg left open and stops (two requests made before the loop
  looked are a second Ctrl+C too). After a guard lock nothing can be sent: a Ctrl+C
  stops at once.
- **The end of the day.** Once every strategy is past its MIS square-off, flat and with
  nothing working, the session ends ("session_end"). From the square-off plus
  `SQUARE_OFF_VERIFY_AFTER_S` a strategy not flat gets the ERROR alert "still not flat
  after square-off", each minute while it lasts. A leg still open at the exchange's close
  plus `SESSION_END_GRACE_S` ends the session too, with an ERROR alert to close it by
  hand. After a guard lock the session ends at that same time ("mode_breach").
- **A strategy's exception** (3.5): an ERROR alert, then the safe stop by `on_stop`. The
  operator's first Ctrl+C, before or after it, asks for the stop already under way: only
  a second one is the second Ctrl+C.
- The slices the plan's catch-up still owns are given up on at their last candle check.

Times are UTC nanoseconds, quantities units.
"""

from __future__ import annotations

import logging

from .._layout import NANOS_PER_SECOND
from ._session_loop import Session
from ._session_trader import StrategyFailed, Trader
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
        self._remind_ns: int | None = None
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
        session.exits.duties(now_ns)
        if self.verdict is not None or not session.trading:
            return
        requested = session.stop.count
        if requested and not self.stopping:
            self._begin(now_ns)
            if self.verdict is not None:
                return
        if self.stopping and requested >= SECOND_CTRL_C:
            self._second()
            return
        if self.stopping:
            self._stop_due(now_ns)
        else:
            self._end_of_day(now_ns)

    # ----- the safe stop --------------------------------------------------------------

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
            self._ask_exits()

    def _ask_exits(self) -> None:
        """Ask the kernel to exit each open position whose exit is not on the retry
        schedule (which asks by its own times, and gives up after its last attempt)."""
        session = self._session
        for trader in session.traders:
            if trader.open_units() == 0 or session.exits.owns(trader.name):
                continue
            what = trader.request_exit(OPERATOR_STOP)
            self._asked[trader.name] = what
            if what in _EXIT_MADE:
                session.event("exit_requested", strategy=trader.name, outcome=what)

    def _stop_due(self, now_ns: int) -> None:
        traders = self._session.traders
        if self._on_stop == "keep":
            waiting = any(trader.entry_orders() for trader in traders)
            if not waiting or now_ns >= self._stopping_ns + _s(SAFE_STOP_STATE_WAIT_S):
                self._left_open()
                self.verdict = self._stopped_by()
            return
        if all(trader.flat for trader in traders):
            self.verdict = self._stopped_by()
            return
        if not self._session.locked:
            self._ask_exits()
        if now_ns >= self._end_ns:
            self._left_open()
            self.verdict = self._stopped_by()
        elif now_ns >= self._remind_ns:
            self._remind_ns = now_ns + _s(UNPROTECTED_REMINDER_EVERY_S)
            self._remind()

    def _second(self) -> None:
        self._left_open()
        self.verdict = self._stopped_by()

    def _remind(self) -> None:
        session = self._session
        for trader in session.traders:
            units = trader.open_units()
            if units == 0:
                continue
            line = (
                f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units: not flat "
                f"yet, {self._exit_state(trader)}"
            )
            session.emit(line)
            session.alert(AlertKind.ERROR, f"safe stop: {line}")

    def _exit_state(self, trader: Trader) -> str:
        """What works toward the exit of `trader`'s position, as a reminder says it."""
        session = self._session
        exits = [trader.order_name(kernel_order) for kernel_order in trader.exit_orders()]
        if exits:
            return f"order {', '.join(exits)} working to close it"
        if session.locked:
            return "nothing can be sent after OpenAlgo's mode changed"
        if session.exits.retry_due(trader.name):
            return "its rejected exit is asked for again shortly"
        if session.exits.owns(trader.name):
            return "its exit failed every attempt, close it by hand"
        return f"no exit works (the engine answered {self._asked.get(trader.name)})"

    def _left_open(self) -> None:
        session = self._session
        for line in _open_lines(session.traders, "no stop at the broker, close it by hand"):
            session.emit(line)
            session.alert(AlertKind.ERROR, line)

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
            self._left_open()
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


def _open_lines(traders: tuple[Trader, ...], what: str) -> list[str]:
    return [
        f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units: {what}"
        for trader in traders
        if (units := trader.open_units()) != 0
    ]


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
