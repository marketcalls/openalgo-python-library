"""How a live session stops (16-session-design 3.3 to 3.5, 6.2; unit 8): the duties by time
that end it (`Control`), and the common close (`close_session`).

**Control** runs in the loop's step D once the session trades:

- **Ctrl+C, the safe stop (D29).** The first stop request locks every strategy's entries
  ("operator_stop") and cancels its working entries and adds. With `on_stop="flatten"`
  each open position is then exited (`request_exit`, "operator_stop"), and the session
  waits until every strategy is flat with nothing working, at most `SAFE_STOP_FLATTEN_S`;
  after that it keeps running, with an ERROR alert and a console line naming each leg
  every `UNPROTECTED_REMINDER_EVERY_S`, until flat, a second Ctrl+C, or the exchange's
  close plus `SESSION_END_GRACE_S`. With `"keep"` the session waits at most
  `SAFE_STOP_STATE_WAIT_S` for the cancels and leaves the positions open. A second
  Ctrl+C prints one line per leg left open and stops. After a guard lock nothing can be
  sent: a Ctrl+C stops at once.
- **The end of the day.** Once every strategy is past its MIS square-off, flat and with
  nothing working, the session ends ("session_end"). From the square-off plus
  `SQUARE_OFF_VERIFY_AFTER_S` a strategy not flat gets the ERROR alert "still not flat
  after square-off", each minute while it lasts. A leg still open at the exchange's close
  plus `SESSION_END_GRACE_S` ends the session too, with an ERROR alert to close it by
  hand. After a guard lock the session ends at that same time ("mode_breach").
- **A strategy's exception** (3.5): an ERROR alert, then the safe stop by `on_stop`.
- The slices the plan's catch-up still owns are given up on at their last candle check.

**The common close** (3.3 step 5): `self.state` saved, the order worker stopped, each
kernel session stopped where it stands (`stop_session`, R20) with its last events
delivered, `on_stop` called and `finish` read, the DAILY_SUMMARY alert, then everything
the start opened closed in reverse (the feed, the data worker, journal L, the logs, the
alerts, the lock). A journal failure or another error closes the same way without the
flatten. What the reconciler (unit 12) adds is not done yet: no final book read.

Times are UTC nanoseconds, quantities units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .._driver import deliver
from .._engine import call
from .._layout import NANOS_PER_SECOND
from ._alerts_summary import SessionSummary, summary_text
from ._plan_catch_up import INSTRUMENT
from ._session_loop import Session
from ._session_trader import Skipped, StrategyFailed, Trader
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import LIVE_PRODUCT
from .bars import BarDrift

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
#: How a session ended (1.4 `stopped_by`).
SESSION_END, CTRL_C, MODE_BREACH, ERROR = "session_end", "ctrl_c", "mode_breach", "error"

_NS = NANOS_PER_SECOND


@dataclass(frozen=True)
class OpenPosition:
    """A position still open when the session ended: the strategy, the instrument, its
    signed units, and a note for the operator."""

    strategy: str
    symbol: str
    exchange: str
    units: float
    note: str


@dataclass(frozen=True)
class SessionOutcome:
    """What one run of a live session leaves (the public `SessionResult` is built from it
    in a later unit): the `mode` ("sandbox" or "live"), `broker`, `account`, IST `day`,
    `started_ns` and `ended_ns`, `stopped_by` ("session_end", "ctrl_c", "mode_breach",
    "error"), `restarts`, the `strategies` in call order, `outputs` (each kernel
    session's `finish()` tables by strategy), `summary` (the day's figures),
    `open_positions`, `mode_breach` (the guard's text or None), `warnings`,
    `skipped_decisions`, `drift` (the accepted candles' drift) and `paths` (journal,
    log, events)."""

    mode: str
    broker: str
    account: str
    day: Any
    started_ns: int
    ended_ns: int
    stopped_by: str
    restarts: int
    strategies: tuple[str, ...]
    outputs: dict[str, Any]
    summary: SessionSummary
    open_positions: tuple[OpenPosition, ...]
    mode_breach: str | None
    warnings: tuple[str, ...]
    skipped_decisions: tuple[Skipped, ...]
    drift: tuple[BarDrift, ...]
    paths: dict[str, str] = field(default_factory=dict)


class Control:
    """The duties by time that stop `session` (module docstring), by its `on_stop`
    ("flatten" or "keep"). `verdict` is set when the session should end."""

    def __init__(self, session: Session, on_stop: str) -> None:
        self._session = session
        self._on_stop = on_stop
        self.verdict: str | None = None
        self.failure: StrategyFailed | None = None
        self._stopping_ns: int | None = None
        self._stops_seen = 0
        self._remind_ns: int | None = None
        self._not_flat_ns = {
            t.name: t.square_off_ns + _s(SQUARE_OFF_VERIFY_AFTER_S) for t in session.traders
        }
        self._end_ns = max(t.close_ns for t in session.traders) + _s(SESSION_END_GRACE_S)

    @property
    def stopping(self) -> bool:
        """True once a safe stop has begun."""
        return self._stopping_ns is not None

    def strategy_failed(self, failed: StrategyFailed) -> None:
        """A strategy's code raised (3.5): an ERROR alert, then the safe stop."""
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
            for note in trader.give_up(now_ns):
                session.warn(note)
        session.exits.duties(now_ns)
        if self.verdict is not None:
            return
        requested = session.stop.count
        if requested and not self.stopping:
            self._stops_seen = requested
            self._begin(now_ns)
            if self.verdict is not None:
                return
        elif self.stopping and requested > self._stops_seen:
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
                call(trader.engine.cancel, kernel_order)
        if session.locked:
            self.verdict = MODE_BREACH
            return
        if self._on_stop == "flatten":
            for trader in session.traders:
                if trader.open_units():
                    what = call(trader.engine.request_exit, INSTRUMENT, LIVE_PRODUCT, OPERATOR_STOP)
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
        for line in _open_lines(session.traders, "not flat yet, the exit is retried"):
            session.emit(line)
            session.alert(AlertKind.ERROR, f"safe stop: {line}")

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
        exits = [trader.venue.order_id_of(order) or f"o{order}" for order in trader.working()]
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


def close_session(
    session: Session, stopped_by: str, *, journal_works: bool = True
) -> SessionOutcome:
    """The common close (module docstring); returns the session's outcome. Every part is
    closed whatever another does; a failure is logged, and the first is raised once all
    are closed (as `LiveError` from the stack)."""
    parts = session.parts
    open_positions = _open_positions(session)
    if journal_works:
        _save_states(session)
    parts.worker.close()
    outputs: dict[str, Any] = {}
    figures = [0.0, 0.0, 0]
    for trader in session.traders:
        outputs[trader.name] = _stop_engine(trader, figures)
    flat = not open_positions
    summary = SessionSummary(
        day=parts.spec.day,
        mode=parts.mode,
        broker=parts.broker,
        strategies=tuple(t.name for t in session.traders),
        closed_trades=int(figures[2]),
        gross_pnl=figures[0],
        charges=figures[1],
        slippage=None,
        rejected_orders=session.rejected,
        unprotected_seconds=0.0,
        reconcile_mismatches=session.mismatches,
        undelivered_alerts=parts.sender.undelivered,
        flat=flat,
    )
    session.alert(AlertKind.DAILY_SUMMARY, summary_text(summary))
    if not session.threads:
        # Under the test seam no sender thread posts: the queue goes now, as it would.
        parts.sender.deliver_queued()
    session.event("closed", stopped_by=stopped_by)
    ended_ns = session.now()
    paths = {
        "journal": str(parts.journal.path),
        "log": str(parts.logs.log_path),
        "events": str(parts.logs.events_path),
    }
    parts.stack.close()
    return SessionOutcome(
        mode=parts.mode.value,
        broker=parts.broker,
        account=parts.account,
        day=parts.spec.day,
        started_ns=session.started_ns,
        ended_ns=ended_ns,
        stopped_by=stopped_by,
        restarts=parts.journal_session.starts - 1,
        strategies=tuple(t.name for t in session.traders),
        outputs=outputs,
        summary=summary,
        open_positions=open_positions,
        mode_breach=None if session.breach is None else session.breach_text,
        warnings=tuple(session.warnings),
        skipped_decisions=tuple(s for t in session.traders for s in t.skipped),
        drift=tuple(d for t in session.traders for d in t.drifts),
        paths=paths,
    )


def _open_positions(session: Session) -> tuple[OpenPosition, ...]:
    note = (
        "known as of the mode breach"
        if session.locked
        else "no stop at the broker: close it by hand"
    )
    return tuple(
        OpenPosition(trader.name, trader.symbol, trader.exchange, units, note)
        for trader in session.traders
        if (units := trader.open_units()) != 0
    )


def _save_states(session: Session) -> None:
    for trader in session.traders:
        state = trader.state_changed()
        if state is not None:
            try:
                session.parts.journal.put_user_state(trader.name, state)
            # The close goes on: the error is the session's to report, not to stop it.
            except Exception:
                logger.exception("%s: self.state not saved at the close", trader.name)


def _stop_engine(trader: Trader, figures: list[float]) -> Any:
    """Stop a kernel session where it stands, deliver its last events, call `on_stop`,
    and return its `finish()` tables; adds its gross P&L, charges and closed trades to
    `figures`."""
    engine, part = trader.engine, trader.part
    _equity, _cash, realized, unrealized, costs, _margin = call(engine.account)
    kind, decided_at, rows, events, _timers = call(engine.stop_session)
    part.context.reveal(decided_at, rows)
    if not trader.failed:
        try:
            if events:
                deliver(part.strategy, part.context, events)
            part.strategy.on_stop()
        # The close goes on whatever the strategy does at its end; its error is logged.
        except Exception:
            logger.exception("%s raised in its last callbacks", trader.name)
    output = call(engine.finish)
    figures[0] += realized + unrealized
    figures[1] += costs
    figures[2] += len(output["trades"]["trade_id"])
    logger.info("%s stopped where it stood (%s)", trader.name, kind)
    return output


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
    "SESSION_END",
    "SESSION_END_GRACE_S",
    "SQUARE_OFF_VERIFY_AFTER_S",
    "UNPROTECTED_REMINDER_EVERY_S",
    "Control",
    "OpenPosition",
    "SessionOutcome",
    "close_session",
]
