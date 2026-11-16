"""A live session from start to stop (16-session-design 1.2, 2, 3; unit 8): `run_session`,
what `bt.live` runs (the public verb comes with the session's result, unit 13).

Threads (2.1). The calling thread runs the session loop, which owns the kernel sessions
and the strategies; the order worker sends the orders and reads the books on its own
thread with the orders and guard clients; the data worker reads quotes and candles on its
own thread with the data client; the alerts post from their own thread; the feed's
callbacks only queue. Under the test seam (`threads=False`) the workers' steps and the
alerts' posts run inline in each loop step and every wait is the injected `sleep`, so a
test drives a whole session on a manual clock (2.11).

The sequence:

1. **Start** (`_session_start.open_session`, 3.1 steps 1 to 11): arguments, clients, mode,
   alerts, locks, logs, preflight (waiting for the broker's login before the open), the
   order worker and the journal, the plan and its catch-up. A failure closes what was
   opened and is raised; a Ctrl+C up to here is a plain KeyboardInterrupt.
2. From here Ctrl+C (SIGINT, and SIGBREAK on Windows or SIGTERM elsewhere) only sets the
   stop flag: the handlers are installed for the session and restored at its end
   (`signals=True`, the main thread only).
3. **The start phase** (`_session_start.begin_trading`, steps 12 to 15): the data worker,
   the banner and the countdown while the loop turns, the feed, then trading.
4. **The loop** (`_session_loop.Session.step`) until `Control` (`_session_stop`) gives its
   verdict: the end of the day, Ctrl+C's safe stop, a mode breach.
5. **The close** (`_session_close.close_session`): the kernel sessions stopped where they
   stand, the summary, everything closed. Returns a `SessionOutcome`.

Errors mid-session (3.5): a strategy's exception stops the session safely by `on_stop`,
then is raised again (a failure during that safe stop takes the emergency stop below, and
the strategy's exception is still the one raised); a journal failure (the worker's or
journal L's) stops sending at once and closes without the flatten, then is raised again;
any other error of the loop locks every strategy's entries, protects each leg by state
while the journal works (its broker-held stop placed again where it can be, for at most
`SAFE_STOP_STATE_WAIT_S`, with no slice released and no strategy called), sends an ERROR
alert naming each open leg, its stop, the orders working toward it and what to do
(`_stops_report.left_open`: a confirmed stop must be cancelled before any sale by hand,
an order working means the order book first, and only a leg with neither is to be closed
by hand), and closes the same way. A stop asked for during
the countdown stops at once (nothing was sent); a feed that does not connect in time
closes the session and raises LiveError.

Times are UTC nanoseconds, amounts rupees.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..errors import JournalError, LiveError
from ..strategy import Strategy
from ._session_close import SessionOutcome, close_session
from ._session_loop import LOOP_WAIT_MAX_S, Session
from ._session_news import WorkerFailedError
from ._session_parts import SessionStopped, StopFlag
from ._session_start import StartSettings, begin_trading, open_session
from ._session_stop import CTRL_C, ERROR, SAFE_STOP_STATE_WAIT_S, Control
from ._session_trader import StrategyFailed, Trader
from ._stops_protect import protect_and_wait
from ._stops_report import left_open
from ._types import LOGGER_NAME, AlertKind, Now, Sleep
from .clients import SessionClients
from .feedlink import FeedSource

logger = logging.getLogger(LOGGER_NAME)

#: The entry lock of an emergency stop (3.5).
EMERGENCY_LOCK = "emergency"


def run_session(
    strategy: type[Strategy] | Sequence[type[Strategy]],
    client: Any,
    *,
    clients: SessionClients | None = None,
    feed: FeedSource | None = None,
    now: Now | None = None,
    sleep: Sleep | None = None,
    threads: bool = True,
    signals: bool = True,
    stop: StopFlag | None = None,
    emit: Callable[[str], None] | None = None,
    wait_max_s: float = LOOP_WAIT_MAX_S,
    **arguments: Any,
) -> SessionOutcome:
    """Run a live session of `strategy` (a `bt.Strategy` subclass or a list of them)
    through `client` (an `openalgo.api`), with `bt.live`'s keyword `arguments`
    (`capital`, `max_daily_loss`, `alerts_to`, `on_stop` and the rest), until it stops;
    see the module docstring.

    The test seam: `clients` (the session's five clients, else derived from `client`),
    `feed` (the feed source, else `clients.feed`), `now` and `sleep` (else the wall clock
    and `time.sleep`), `threads` False, `signals` False (no handlers; `stop` is then the
    only Ctrl+C), `stop` (a `StopFlag`), `emit` (the console, else stderr) and
    `wait_max_s`.

    Raises ConfigError and NotImplementedError for unusable arguments, PreflightError
    for a failed check, LiveError when OpenAlgo does not report its mode, a lock is
    held, the handlers cannot be installed off the main thread, the order worker does
    not start, a restart of the day finds its orders, or the feed does not connect;
    JournalError for the journal; and the strategy's own exception after the safe stop.
    """
    settings = StartSettings(
        now=time.time_ns if now is None else now,
        sleep=time.sleep if sleep is None else sleep,
        threads=threads,
        emit=_console if emit is None else emit,
        stop=StopFlag() if stop is None else stop,
        wait_max_s=wait_max_s,
        clients=clients,
        feed=feed,
    )
    if signals and threading.current_thread() is not threading.main_thread():
        raise LiveError(
            "bt.live runs on the main thread: Ctrl+C reaches only the main thread, and it "
            "is how a session is stopped safely"
        )
    session = open_session(strategy, client, settings, arguments)
    handlers = _stop_handlers(settings.stop) if signals else contextlib.nullcontext()
    with handlers:
        return _run(session, settings, session.parts.spec.arguments.on_stop)


def _run(session: Session, settings: StartSettings, on_stop: str) -> SessionOutcome:
    control = Control(session, on_stop)
    session.control = control
    source = settings.feed if settings.feed is not None else session.parts.clients.feed
    try:
        begin_trading(session, source)
    except SessionStopped:
        return close_session(session, CTRL_C)
    except BaseException as error:
        failure = _cause(error)
        _abort(session, failure, journal_works=not isinstance(failure, JournalError))
        if failure is error:
            raise
        raise failure from error
    return _loop(session, control)


def _loop(session: Session, control: Control) -> SessionOutcome:
    try:
        while control.verdict is None:
            try:
                session.step()
            except StrategyFailed as failed:
                control.strategy_failed(failed)
    except BaseException as error:
        failure = _cause(error)
        _abort(session, failure, journal_works=not isinstance(failure, JournalError))
        # A failure during the safe stop of a strategy's exception: the strategy's error
        # stopped the session first and is the one raised; the failure is in the log and
        # the alert, and chained as the error's context.
        if control.failure is not None:
            raise control.failure.error  # noqa: B904
        if failure is error:
            raise
        raise failure from error
    outcome = close_session(session, control.verdict)
    if control.failure is not None:
        raise control.failure.error
    return outcome


def _cause(error: BaseException) -> BaseException:
    """The error that stopped the session: a strategy's or the order worker's own."""
    if isinstance(error, StrategyFailed | WorkerFailedError):
        return error.error
    return error


def _abort(session: Session, error: BaseException, *, journal_works: bool) -> None:
    """The emergency stop (3.5): every strategy's entries are locked, each leg is
    protected by state while the journal works, an ERROR alert lists each open leg with
    its left-open note (its stop, the orders working toward it, and what to do: only a
    leg with no stop and nothing working is to be closed by hand), and the session closes
    without the flatten. A failure of the close is logged: the error that stopped the
    session is the one raised."""
    logger.error("the session stops on %s: %s", type(error).__name__, error)
    try:
        for trader in session.traders:
            trader.lock(EMERGENCY_LOCK)
        if journal_works and not session.locked:
            _protect(session)
        legs = [leg for trader in session.traders if (leg := _open_leg(session, trader))]
        session.alert(
            AlertKind.ERROR,
            f"the session stops on an error ({type(error).__name__}: {error}). Open: "
            f"{'. '.join(legs) if legs else 'nothing'}",
        )
        close_session(session, ERROR, journal_works=journal_works)
    # The close of a failed session must not hide the failure that stopped it.
    except Exception:
        logger.exception("the session's close after an error failed")
        session.parts.stack.close()


def _open_leg(session: Session, trader: Trader) -> str:
    """The emergency alert's words for `trader`'s leg: "" when flat, else the leg and its
    left-open note. The alert is the operator's only notice, so a leg whose kernel calls
    fail is still listed, as unknown, with the order book to check (open decision 15)."""
    try:
        units = trader.open_units()
        if units == 0:
            return ""
        return f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units: " + left_open(
            session.stops, trader
        )
    # Whatever the kernel's state, the alert must go out.
    except Exception as failure:
        logger.exception("%s: its leg cannot be read for the emergency alert", trader.name)
        return (
            f"{trader.name} {trader.symbol} {trader.exchange}: its position and stop cannot "
            f"be read ({type(failure).__name__}), check the order book and the positions"
        )


def _protect(session: Session) -> None:
    """Protect each leg by state; a failure there (the kernel's own, say) is logged, and
    the alert then names the legs it left without a stop."""
    try:
        protect_and_wait(session, SAFE_STOP_STATE_WAIT_S)
    # The emergency stop goes on whatever fails here: what is open is alerted below.
    except Exception:
        logger.exception("protect by state failed in the emergency stop")


@contextlib.contextmanager
def _stop_handlers(stop: StopFlag) -> Iterator[None]:
    """Ctrl+C (and a service shutdown) sets the stop flag for the session; the previous
    handlers come back at its end."""
    names = ["SIGINT", "SIGBREAK" if sys.platform == "win32" else "SIGTERM"]
    kept: dict[int, Any] = {}
    try:
        for name in names:
            number = getattr(signal, name, None)
            if number is not None:
                kept[number] = signal.signal(number, lambda _number, _frame: stop.request())
        yield
    finally:
        for number, previous in kept.items():
            signal.signal(number, previous)


def _console(line: str) -> None:
    """The operator's console: standard error, so the session's lines never mix with a
    program's own output."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


__all__ = ["EMERGENCY_LOCK", "SessionOutcome", "StopFlag", "run_session"]
