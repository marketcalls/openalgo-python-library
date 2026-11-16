"""The common close of a live session (16-session-design 3.3 step 5, 3.5, 1.4; unit 8),
and what a session leaves (`SessionOutcome`).

`self.state` saved, the order worker stopped, each kernel session stopped where it stands
(`stop_session`, R20) with its last events delivered, `on_stop` called and `finish` read,
the DAILY_SUMMARY alert, then everything the start opened closed in reverse (the feed,
the data worker, journal L, the logs, the alerts, the lock). A journal failure or another
error closes the same way without the flatten (journal L is not written after a journal
failure). The reconciler (unit 12) adds a final book read; the public `SessionResult`
(unit 13) is built from the outcome.

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
from ._session_loop import Session
from ._session_trader import Skipped, Trader
from ._types import LOGGER_NAME, AlertKind
from .bars import BarDrift

logger = logging.getLogger(LOGGER_NAME)


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


def close_session(
    session: Session, stopped_by: str, *, journal_works: bool = True
) -> SessionOutcome:
    """The common close (module docstring); returns the session's outcome. Whatever stops
    a kernel session or the strategy's last callbacks raise, everything the start opened
    is closed before the error goes on; a part that fails to close is logged by the
    stack, which raises LiveError naming it once all are closed."""
    parts = session.parts
    try:
        open_positions = _open_positions(session)
        if journal_works:
            _save_states(session)
        parts.worker.close()
        outputs = {trader.name: _stop_engine(trader) for trader in session.traders}
        summary = _summary(session, outputs, flat=not open_positions)
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
    )


def _summary(session: Session, outputs: dict[str, Any], *, flat: bool) -> SessionSummary:
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
        unprotected_seconds=0.0,
        reconcile_mismatches=session.mismatches,
        undelivered_alerts=parts.sender.undelivered,
        flat=flat,
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
