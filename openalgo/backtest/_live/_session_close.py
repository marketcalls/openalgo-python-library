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
still working, naming what the guard held back; what a restart of the day needs saved
to journal L, now that the final fills are in (each strategy's state with its figures and
its `self.state`, the session's counts with the mismatches the close names:
`_session_figures`); the order worker stopped; each kernel session stopped where it
stands (`stop_session`, R20) with its last events delivered, `on_stop` called and
`finish` read; the DAILY_SUMMARY alert; then
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
since (`_reconcile_close`), and a leg they leave flat is no open position (round 8; the
listing is `_session_open`'s). A leg no read judged after a fill of its own orders landed
during a read is named on those lines as not confirmed, with the engine's units, and has
one alert to check OpenAlgo's position book (round 9). What the close names besides the
legs it holds is `_session_named`'s: an instrument several strategies share has one row of
its own after the legs (when OpenAlgo's net is a position or differs from the members'
sum), OpenAlgo's net against the members' sum, and one line of the DAILY_SUMMARY, and its
group's difference is named and counted once (18-multi-strategy 4.11). The instrument's
row holds no position of its own: the day is flat when no row holds units. The public
`SessionResult` (unit 13) is built from the outcome.

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
from ._plan_catch_up import StrategyPlan
from ._reconcile_final import final_read
from ._reconcile_findings import Reconciliation
from ._session_loop import Session
from ._session_named import (
    NamedLegs,
    SharedPosition,
    alert_not_confirmed,
    named_legs,
    shared_positions,
)
from ._session_open import OpenPosition, open_positions
from ._session_records import FillRecord, LockRecord, OrderRecord, Restored, order_records
from ._session_saved import save, save_session
from ._session_slippage import Slippage, strategy_slippage
from ._session_trader import Skipped, Trader
from ._stops_cover import Unprotected
from ._stops_held import Estimate
from ._types import LOGGER_NAME, AlertKind
from .bars import BarDrift

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the common close waits for its final book read (3.3 step 5): the safe stop's
#: wait for its last words (`_session_stop.SAFE_STOP_STATE_WAIT_S`).
FINAL_READ_WAIT_S = 5.0
#: How a session ended that takes the final read: not after an error (3.5) or a mode
#: breach (6.2), when the books are not to be trusted or not to be read.
_FINAL_READ = frozenset({"session_end", "ctrl_c"})
#: How the close warns of a strategy's state it could not save, and what a restart does.
NOT_SAVED = "its state and the day's figures were not saved at the close"
RESUMES_FROM_LAST_SAVE = "a restart today resumes it from its last save"


@dataclass(frozen=True)
class SessionOutcome:
    """What one run of a live session leaves (the public `SessionResult` is built from it,
    `_result_build`): the `mode` ("sandbox" or "live"), `broker`, `account`, IST `day`,
    `started_ns` and `ended_ns`, `stopped_by` ("session_end", "ctrl_c", "mode_breach",
    "error"), `restarts`, the `strategies` in call order, `outputs` (each kernel
    session's `finish()` tables by strategy), `summary` (the day's figures across its
    restarts, `_session_figures`), `open_positions`, `mode_breach` (the guard's text or
    None), `warnings`, `skipped_decisions`, `drift` (the accepted candles' drift), `paths`
    (journal, log, events, decisions), `unprotected` (each window a leg had no confirmed stop at the
    broker), `estimated` (the fills released at a price the books did not confirm),
    `reconciliations` (each comparison with the broker's books that differed), and the
    records of `_session_records`: `orders`, `fills` and `locks`; `plans`, each
    strategy's part of the session plan by name (its settings, strategy and run context,
    for its result), `timer_warnings`, its timers' notes, read before its kernel
    session finished, and `restored`, each restored strategy's restore at a restart."""

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
    orders: tuple[OrderRecord, ...] = ()
    fills: tuple[FillRecord, ...] = ()
    locks: tuple[LockRecord, ...] = ()
    plans: dict[str, StrategyPlan] = field(default_factory=dict)
    timer_warnings: dict[str, tuple[str, ...]] = field(default_factory=dict)
    restored: dict[str, Restored] = field(default_factory=dict)


def close_session(
    session: Session, stopped_by: str, *, journal_works: bool = True
) -> SessionOutcome:
    """The common close (module docstring); returns the session's outcome. Whatever stops
    a kernel session or the strategy's last callbacks raise, everything the start opened
    is closed before the error goes on; a part that fails to close is logged by the
    stack, which raises LiveError naming it once all are closed."""
    parts = session.parts
    try:
        _settle(session, stopped_by, journal_works=journal_works)
        # Named while the kernel sessions run: a finished one has no units to read.
        named = _named(session)
        unprotected_s = session.stops.windows.seconds(session.now())
        if journal_works:
            _save_day(session, unprotected_s, named.legs.new_mismatches)
        parts.worker.close()
        # Read while each kernel session runs: a finished one counts its timers no more.
        timers = {t.name: tuple(t.part.context.timers.warnings()) for t in session.traders}
        slipped = {t.name: strategy_slippage(session, t) for t in session.traders}
        stopped = {trader.name: _stop_engine(trader) for trader in session.traders}
        outputs = {name: output for name, (output, _) in stopped.items()}
        orders = order_records(session, outputs)
        summary = _summary(session, (stopped, slipped), unprotected_s, named)
        session.alert(AlertKind.DAILY_SUMMARY, summary_text(summary))
        if not session.threads:
            # Under the test seam no sender thread posts: the queue goes now, as it would.
            parts.sender.deliver_queued()
        session.event("closed", stopped_by=stopped_by)
        outcome = _outcome(session, stopped_by, (outputs, orders, timers), summary, named)
    finally:
        parts.stack.close()
    return outcome


@dataclass(frozen=True)
class _Named:
    """What the close names, read while the kernel sessions run: the `open_positions`,
    the legs the summary names (`legs`, `_session_named.NamedLegs`) and each shared
    instrument (`shared`)."""

    open_positions: tuple[OpenPosition, ...]
    legs: NamedLegs
    shared: tuple[SharedPosition, ...]

    @property
    def flat(self) -> bool:
        """True when no row holds units: an instrument's row listed for a flat net its
        members' sum differs from holds none (module docstring)."""
        return not any(position.units for position in self.open_positions)


def _settle(session: Session, stopped_by: str, *, journal_works: bool) -> None:
    """The close's steps before it names what is left (module docstring): the stops
    supervisor's and the reconciler's ends, what the order worker put back withdrawn, the
    final read after a session end or a Ctrl+C, a log line per request the exit guard held
    back, and the alerts of the legs that changed after the last lines or that no read
    confirmed."""
    session.stops.after.end(session.traders)
    session.reconcile.end()
    session.parts.worker.withdraw_put_back()
    if journal_works and stopped_by in _FINAL_READ:
        final_read(session.reconcile, session, FINAL_READ_WAIT_S)
    for request in session.reconcile.guard.left():
        logger.warning("a request held back by the exit guard was not sent: %r", request)
    session.stops.after.alert(session)
    alert_not_confirmed(session)


def _named(session: Session) -> _Named:
    """What the close names (`_Named`): each shared instrument, then the open positions
    with its row, then the legs the summary names."""
    shared = shared_positions(session)
    return _Named(open_positions(session, shared), named_legs(session), shared)


def _outcome(
    session: Session,
    stopped_by: str,
    records: tuple[dict[str, Any], tuple[OrderRecord, ...], dict[str, tuple[str, ...]]],
    summary: SessionSummary,
    named: _Named,
) -> SessionOutcome:
    """The session's outcome, ended now: `records` are each kernel session's `finish()`
    tables, the orders and each strategy's timers' notes, by strategy."""
    parts = session.parts
    outputs, orders, timers = records
    return SessionOutcome(
        mode=parts.mode.value,
        broker=parts.broker,
        account=parts.account,
        day=parts.spec.day,
        started_ns=session.started_ns,
        ended_ns=session.now(),
        stopped_by=stopped_by,
        restarts=parts.journal_session.starts - 1,
        strategies=tuple(t.name for t in session.traders),
        outputs=outputs,
        summary=summary,
        open_positions=named.open_positions,
        mode_breach=session.breach_text,
        warnings=tuple(session.warnings),
        skipped_decisions=tuple(s for t in session.traders for s in t.skipped),
        drift=tuple(d for t in session.traders for d in t.candles.drifts),
        paths={
            "journal": str(parts.journal.path),
            "log": str(parts.logs.log_path),
            "events": str(parts.logs.events_path),
            "decisions": str(parts.logs.decisions_path),
        },
        unprotected=session.stops.windows.rows(),
        estimated=tuple(session.stops.held.estimates),
        reconciliations=tuple(session.reconcile.rows),
        orders=orders,
        fills=tuple(session.fills),
        locks=tuple(lock for trader in session.traders for lock in trader.records.locks),
        plans={trader.name: trader.part for trader in session.traders},
        timer_warnings=timers,
        restored={t.name: t.records.restored for t in session.traders if t.records.restored},
    )


def _summary(
    session: Session,
    stopped: tuple[dict[str, Any], dict[str, Slippage]],
    unprotected_s: float,
    named: _Named,
) -> SessionSummary:
    """The day's figures across its restarts (`_session_figures`): the trades the kernel
    sessions closed, their gross P&L and modelled charges (the broker's trade book carries
    none) from `stopped`'s outputs, the slippage of their MARKET fills against the
    backtest's (`stopped`'s slippages, `_session_slippage`), the session's counts, the legs
    `named` gives (`_session_named.named_legs`), each differing one a mismatch too unless
    the day counted it already, and a line per shared instrument it gives; the strategies
    earlier starts ran and this one does not, the day not flat while a row of `named` holds
    units or one of them held a position at its last save."""
    parts = session.parts
    figures = session.figures
    outputs, slipped = stopped
    legs = named.legs
    traded, counted = figures.day(session, outputs, unprotected_s, slippages=slipped)
    return SessionSummary(
        day=parts.spec.day,
        mode=parts.mode,
        broker=parts.broker,
        strategies=tuple(t.name for t in session.traders),
        closed_trades=traded.closed_trades,
        gross_pnl=traded.gross_pnl,
        charges=traded.charges,
        slippage=traded.summary_slippage,
        slippage_not_compared=traded.slippage_not_compared,
        rejected_orders=counted.rejected_orders,
        unprotected_seconds=counted.unprotected_seconds,
        reconcile_mismatches=counted.reconcile_mismatches + legs.new_mismatches,
        undelivered_alerts=counted.undelivered_alerts,
        flat=named.flat and not figures.earlier_open,
        differing=legs.differs,
        unconfirmed=legs.unsure,
        earlier=tuple(strategy.words() for strategy in figures.earlier),
        shared=tuple(instrument.line() for instrument in named.shared),
    )


def _save_day(session: Session, unprotected_s: float, named: int) -> None:
    """Save to journal L what a restart of the day needs, as the loop's step F does, now
    that the final fills are in: each running strategy's state with its figures, and its
    `self.state` when it changed; then the session's counts with what the close adds,
    `named` mismatches and `unprotected_s` seconds unprotected. A save the journal cannot
    take is a warning of the result: the close goes on, since what is open matters more
    than what is saved."""
    for trader in session.traders:
        try:
            save(session, trader)
        except JournalError as error:
            session.warn(f"{trader.name}: {NOT_SAVED} ({error}): {RESUMES_FROM_LAST_SAVE}")
        state = trader.state_changed()
        if state is None:
            continue
        try:
            session.parts.journal.put_user_state(trader.name, state)
        except JournalError as error:
            session.warn(f"{trader.name}: self.state not saved at the close: {error}")
    try:
        save_session(session, unprotected_s=unprotected_s, named=named)
    except JournalError as error:
        session.warn(f"the session's counts of the day were not saved at the close: {error}")


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


__all__ = ["SessionOutcome", "close_session"]
