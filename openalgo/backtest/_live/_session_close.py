"""The common close of a live session (16-session-design 3.3 step 5, 3.5, 1.4; unit 8),
and what a session leaves (`SessionOutcome`).

A final book read and reconciliation first, after a session end or a Ctrl+C (`reconcile`;
not after an error or a mode breach), `self.state` saved, the order worker stopped, each
kernel session stopped where it stands (`stop_session`, R20) with its last events
delivered, `on_stop` called and `finish` read,
the DAILY_SUMMARY alert, then everything the start opened closed in reverse (the feed,
the data worker, journal L, the logs, the alerts, the lock). A journal failure or another
error closes the same way without the flatten (journal L is not written after a journal
failure). A position left open is reported with its broker-held stop and what works
toward its exit (`_stops_report.left_open`), and the day's unprotected seconds
are the supervisor's windows; each comparison of the day that differed is in
`reconciliations`. The public `SessionResult` (unit 13) is built from the outcome.

Times are UTC nanoseconds, quantities units, amounts rupees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .._driver import deliver
from .._engine import call
from ..errors import JournalError
from ._alerts_summary import SessionSummary, summary_text
from ._reconcile_findings import Reconciliation
from ._session_loop import Session
from ._session_trader import Skipped, Trader
from ._stops_cover import Unprotected
from ._stops_held import Estimate
from ._stops_report import left_open, stop_state
from ._types import LOGGER_NAME, AlertKind
from .bars import BarDrift

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the common close waits for its final book read (3.3 step 5): the safe stop's
#: wait for its last words (`_session_stop.SAFE_STOP_STATE_WAIT_S`).
FINAL_READ_WAIT_S = 5.0
#: How a session ended that takes the final read: not after an error (3.5) or a mode
#: breach (6.2), when the books are not to be trusted or not to be read.
_FINAL_READ = frozenset({"session_end", "ctrl_c"})


@dataclass(frozen=True)
class OpenPosition:
    """A position still open when the session ended: the strategy, the instrument, its
    signed units, a note for the operator, its stop's broker order ID ("" for none, or one
    not bound) and whether that stop was confirmed working at the broker."""

    strategy: str
    symbol: str
    exchange: str
    units: float
    note: str
    stop_order_id: str = ""
    stop_confirmed: bool = False


@dataclass(frozen=True)
class SessionOutcome:
    """What one run of a live session leaves (the public `SessionResult` is built from it
    in a later unit): the `mode` ("sandbox" or "live"), `broker`, `account`, IST `day`,
    `started_ns` and `ended_ns`, `stopped_by` ("session_end", "ctrl_c", "mode_breach",
    "error"), `restarts`, the `strategies` in call order, `outputs` (each kernel
    session's `finish()` tables by strategy), `summary` (the day's figures),
    `open_positions`, `mode_breach` (the guard's text or None), `warnings`,
    `skipped_decisions`, `drift` (the accepted candles' drift), `paths` (journal, log,
    events), `unprotected` (each window a leg had no confirmed stop at the broker),
    `estimated` (the fills released at a price the books did not confirm) and
    `reconciliations` (each comparison with the broker's books that differed)."""

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
    unprotected: tuple[Unprotected, ...] = ()
    estimated: tuple[Estimate, ...] = ()
    reconciliations: tuple[Reconciliation, ...] = ()


def close_session(
    session: Session, stopped_by: str, *, journal_works: bool = True
) -> SessionOutcome:
    """The common close (module docstring); returns the session's outcome. Whatever stops
    a kernel session or the strategy's last callbacks raise, everything the start opened
    is closed before the error goes on; a part that fails to close is logged by the
    stack, which raises LiveError naming it once all are closed."""
    parts = session.parts
    try:
        if journal_works and stopped_by in _FINAL_READ:
            session.reconcile.final(FINAL_READ_WAIT_S)
        for request in session.reconcile.guard.left():
            logger.warning("a request held back by the exit guard was not sent: %r", request)
        open_positions = _open_positions(session)
        unprotected_s = session.stops.windows.seconds(session.now())
        if journal_works:
            _save_states(session)
        parts.worker.close()
        outputs = {trader.name: _stop_engine(trader) for trader in session.traders}
        summary = _summary(session, outputs, unprotected_s, flat=not open_positions)
        session.alert(AlertKind.DAILY_SUMMARY, summary_text(summary))
        if not session.threads:
            # Under the test seam no sender thread posts: the queue goes now, as it would.
            parts.sender.deliver_queued()
        session.event("closed", stopped_by=stopped_by)
        ended_ns = session.now()
    finally:
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
        outputs={name: output for name, (output, _) in outputs.items()},
        summary=summary,
        open_positions=open_positions,
        mode_breach=session.breach_text,
        warnings=tuple(session.warnings),
        skipped_decisions=tuple(s for t in session.traders for s in t.skipped),
        drift=tuple(d for t in session.traders for d in t.candles.drifts),
        paths={
            "journal": str(parts.journal.path),
            "log": str(parts.logs.log_path),
            "events": str(parts.logs.events_path),
        },
        unprotected=session.stops.windows.rows(),
        estimated=tuple(session.stops.held.estimates),
        reconciliations=tuple(session.reconcile.rows),
    )


def _summary(
    session: Session, outputs: dict[str, Any], unprotected_s: float, *, flat: bool
) -> SessionSummary:
    """The day's figures of this start: the trades the kernel sessions closed, their
    gross P&L and modelled charges (the broker's trade book carries none)."""
    parts = session.parts
    closed = sum(len(output["trades"]["trade_id"]) for output, _ in outputs.values())
    return SessionSummary(
        day=parts.spec.day,
        mode=parts.mode,
        broker=parts.broker,
        strategies=tuple(t.name for t in session.traders),
        closed_trades=closed,
        gross_pnl=sum(gross for _, (gross, _) in outputs.values()),
        charges=sum(charges for _, (_, charges) in outputs.values()),
        slippage=None,
        rejected_orders=session.rejected,
        unprotected_seconds=unprotected_s,
        reconcile_mismatches=session.mismatches,
        undelivered_alerts=parts.sender.undelivered,
        flat=flat,
    )


def _open_positions(session: Session) -> tuple[OpenPosition, ...]:
    stops = session.stops
    return tuple(
        OpenPosition(
            trader.name,
            trader.symbol,
            trader.exchange,
            units,
            "known as of the mode breach" if session.locked else left_open(stops, trader),
            *stop_state(stops, trader),
        )
        for trader in session.traders
        if (units := trader.open_units()) != 0
    )


def _save_states(session: Session) -> None:
    """Save each changed `self.state`. One the journal cannot take is a warning of the
    result: the close goes on, since what is open matters more than the saved state."""
    for trader in session.traders:
        state = trader.state_changed()
        if state is None:
            continue
        try:
            session.parts.journal.put_user_state(trader.name, state)
        except JournalError as error:
            session.warn(f"{trader.name}: self.state not saved at the close: {error}")


def _stop_engine(trader: Trader) -> tuple[Any, tuple[float, float]]:
    """Stop a kernel session where it stands, deliver its last events and call
    `on_stop` (unless the strategy failed); its `finish()` tables, and its (gross P&L,
    modelled charges) in rupees."""
    engine, part = trader.engine, trader.part
    _equity, _cash, realized, unrealized, costs, _margin = call(engine.account)
    if not trader.finished:
        _kind, decided_at, rows, events, _timers = call(engine.stop_session)
        part.context.reveal(decided_at, rows)
        if not trader.failed:
            try:
                if events:
                    deliver(part.strategy, part.context, events)
                part.strategy.on_stop()
            # The close goes on whatever the strategy does at its end: its error is logged
            # with its traceback, and what is open is closed below.
            except Exception:
                logger.exception("%s raised in its last callbacks", trader.name)
        logger.info("%s stopped where it stood", trader.name)
    return call(engine.finish), (realized + unrealized, costs)


__all__ = ["OpenPosition", "SessionOutcome", "close_session"]
