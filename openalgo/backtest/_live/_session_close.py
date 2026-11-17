"""The common close of a live session (16-session-design 3.3 step 5, 3.5, 1.4; unit 8),
and what a session leaves (`SessionOutcome`).

The last lines are printed before the close begins, and from then on the session sends
nothing they did not name (3.3 step 5 "As built"): the exit guard sends none of the
requests it holds and no new one but what only takes exposure away, the reconciler's own
safe corrections among them (`ExitGuard.end`), the order worker withdraws what it put
back to send later (`OrderWorker.withdraw_put_back`: OpenAlgo silent, an unbound hold),
and no price reaches the engine or the stops supervisor (`_stops_after`). Then a final
book read and reconciliation, after a session end or a Ctrl+C (`reconcile`; not after an
error or a mode breach), which takes in the books and the order news only, and waits
within its seconds for the answers to what it handed over (a stop fitted to a sale by
hand it adopts among them); an ERROR alert for each leg whose last comparison saw a
difference with OpenAlgo's books no later read will confirm, for each leg whose units or
stop changed after the lines, for each leg left open that the exit guard still holds
(OpenAlgo's books never agreed with the engine), and for each leg left flat with an order
still working, naming what the guard held back; `self.state` saved; the order worker
stopped; each kernel session stopped where it stands (`stop_session`, R20) with its last
events delivered, `on_stop` called and `finish` read; the DAILY_SUMMARY alert; then
everything the start opened closed in reverse (the feed, the data worker, journal L, the
logs, the alerts, the lock). A journal failure or another error closes the same way
without the flatten (journal L is not written after a journal failure). A position left
open is reported with its broker-held stop and what works toward its exit
(`_stops_report.left_open`), after what happened to it following the last lines when
anything did (`_stops_after.AfterLines.note`), and the day's unprotected seconds are the
supervisor's windows; each comparison of the day that differed is in `reconciliations`.
A leg whose difference with OpenAlgo's books no later read will confirm (16-session-design
5.4 "As built", round 7) is an open position at OpenAlgo's units with no confirmed stop,
flat or not in the engine, a mismatch of the summary and named on its Reconcile and End
state lines; OpenAlgo's units are those the read showed, moved by the engine's own fills
since (`_reconcile_close`), and a leg they leave flat is no open position (round 8). A leg
no read judged after a fill of its own orders landed during a read is named on those lines
as not confirmed, with the engine's units, and has one alert to check OpenAlgo's position
book (round 9). The public `SessionResult` (unit 13) is built from the outcome.

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
from ._reconcile_close import (
    NOT_CONFIRMED_ALERT,
    differing,
    late_alert,
    late_summary,
    not_confirmed_alert,
    not_confirmed_summary,
)
from ._reconcile_final import final_read
from ._reconcile_findings import Reconciliation
from ._session_loop import Session
from ._session_trader import Skipped, Trader
from ._stops_cover import Unprotected
from ._stops_held import Estimate
from ._stops_report import stop_state
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
    signed units (OpenAlgo's at the last read when they differed from the engine's and no
    later read confirmed it, moved by the engine's own fills since: `_reconcile_close`), a
    note for the operator, its stop's broker order ID ("" for none, or one not bound) and
    whether that stop was confirmed working at the broker for no more units than the
    position holds (one larger, which a trigger would take through zero, is not, nor any
    while OpenAlgo's books show other units than the engine's:
    `_stops_report.stop_state`)."""

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
        session.stops.after.end(session.traders)
        session.reconcile.end()
        parts.worker.withdraw_put_back()
        if journal_works and stopped_by in _FINAL_READ:
            final_read(session.reconcile, session, FINAL_READ_WAIT_S)
        for request in session.reconcile.guard.left():
            logger.warning("a request held back by the exit guard was not sent: %r", request)
        session.stops.after.alert(session)
        _alert_not_confirmed(session)
        open_positions = _open_positions(session)
        # Named while the kernel sessions run: a finished one has no units to read.
        named = _named_legs(session)
        unprotected_s = session.stops.windows.seconds(session.now())
        if journal_works:
            _save_states(session)
        parts.worker.close()
        outputs = {trader.name: _stop_engine(trader) for trader in session.traders}
        summary = _summary(session, outputs, unprotected_s, named, not open_positions)
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


def _named_legs(session: Session) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The legs the summary names at the close, in its words: each whose last comparison
    was late and saw a difference (`_reconcile_close.late_summary`) or whose difference
    with OpenAlgo's books no later read confirms (`_reconcile_close.Differing.summary`),
    then each with none that no read judged after a fill of its own orders landed during a
    read (`_reconcile_close.not_confirmed_summary`)."""
    differs, unsure = [], []
    for trader in session.traders:
        late = session.reconcile.late(trader.name)
        seen = differing(session.reconcile, trader)
        if late is not None:
            differs.append(late_summary(trader, late))
        elif seen is not None:
            differs.append(seen.summary(trader))
        elif session.reconcile.unjudged(trader.name):
            unsure.append(not_confirmed_summary(trader, round(trader.open_units())))
    return tuple(differs), tuple(unsure)


def _alert_not_confirmed(session: Session) -> None:
    """One alert per leg no read judged after a fill of its own orders landed during a
    read, to check OpenAlgo's position book (`_reconcile_close.not_confirmed_alert`), and
    one ERROR alert per leg whose last comparison was late and saw a difference, with that
    read's figures (`_reconcile_close.late_alert`)."""
    for trader in session.traders:
        late = session.reconcile.late(trader.name)
        if late is not None:
            session.alert(AlertKind.ERROR, late_alert(trader, late))
        elif session.reconcile.unjudged(trader.name):
            differs = differing(session.reconcile, trader)
            text = not_confirmed_alert(trader, round(trader.open_units()), differs)
            session.alert(NOT_CONFIRMED_ALERT, text)


def _summary(
    session: Session,
    outputs: dict[str, Any],
    unprotected_s: float,
    legs: tuple[tuple[str, ...], tuple[str, ...]],
    flat: bool,
) -> SessionSummary:
    """The day's figures of this start: the trades the kernel sessions closed, their
    gross P&L and modelled charges (the broker's trade book carries none), and the legs
    `_named_legs` gives, each differing one a mismatch too."""
    differs, unsure = legs
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
        reconcile_mismatches=session.mismatches + len(differs),
        undelivered_alerts=parts.sender.undelivered,
        flat=flat,
        differing=differs,
        unconfirmed=unsure,
    )


def _open_positions(session: Session) -> tuple[OpenPosition, ...]:
    """Each leg the engine holds open, and each whose difference with OpenAlgo's books no
    later read confirms, at OpenAlgo's units then; none they show flat (module
    docstring)."""
    stops, positions = session.stops, []
    for trader in session.traders:
        differs = differing(session.reconcile, trader)
        units = trader.open_units() if differs is None else differs.broker_units
        if units == 0:
            continue
        note = (
            "known as of the mode breach" if session.locked else stops.after.note(session, trader)
        )
        positions.append(
            OpenPosition(
                trader.name, trader.symbol, trader.exchange, units, note, *stop_state(stops, trader)
            )
        )
    return tuple(positions)


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
