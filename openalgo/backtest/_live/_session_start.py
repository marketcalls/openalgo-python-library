"""The start of a live session (16-session-design 3.1 and 3.2; unit 8): `open_session`
runs steps 1 to 11 and returns the loop, `begin_trading` runs steps 12 to 15 on it.

1. The arguments and the strategies (`plan.specify`): ConfigError or NotImplementedError.
2. The session's clients (`clients.derive`), unless given (the test seam).
3. The mode: the guard reads OpenAlgo's analyzer switch, and the session adopts it.
4. The alerts, on their own client, shared by every thread.
5. `ping` names the broker (the default account name); the account and server locks.
6. The day's logs under the account's folder.
7. The preflight. Started before the open, a preflight that fails only on the broker's
   login is run again every `LOGIN_RETRY_EVERY_S`, its alert test reported again rather
   than sent, with an ERROR alert at `LOGIN_ALERT_AT` IST; at the open it fails. The
   guard's banner line is read now.
8. The order worker starts journal handle W, the journal's session of the day and the
   order path (on its own thread, or inline under the test seam), and the main thread
   opens journal handle L. A restart of the day with any order in the journal is
   refused: resuming one is built with the reconciler (unit 12).
11. The plan (`plan.build`): warm-up from broker history, each strategy's kernel session,
   live series and `init`, its saved `self.state` merged, the catch-up to the present.
   Each kernel session's entries are locked ("starting") and its sizing equity capped by
   the funds available. Steps 9 and 10 (unbound intents and the restart comparison)
   belong to a restart.

A failure up to step 11 closes what was opened, in reverse, and is raised. Then
`begin_trading`: 12. the data worker starts and the loop turns (the start phase);
13. the banner, a SESSION alert with its first lines, and the countdown, during which the
loop keeps turning; 14. the feed connects and the session watches it; 15. entries are
unlocked. A stop requested during 13 or 14 raises `SessionStopped`.

Times are UTC nanoseconds, amounts rupees.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._engine import call
from .._layout import NANOS_PER_SECOND
from ..calendar import default_calendar
from ..errors import LiveError
from ..strategy import Strategy
from ._alerts_logs import configure_session_logs
from ._journal_file import DEFAULT_BUSY_TIMEOUT_S
from ._preflight_lock import live_root
from ._session_banner import BANNER_ALERT_LINES, banner_lines
from ._session_loop import Session
from ._session_parts import SessionParts, StopFlag
from ._session_trader import Trader, ist_instant
from ._shared_alerts import SharedAlerts
from ._types import LOGGER_NAME, AlertKind, Mode, Now, Sleep
from .alerts import Alerts
from .clients import SessionClients, derive
from .data_worker import DataWorker
from .feedlink import FeedLink, FeedSource
from .guard import ModeGuard
from .inbox import Inbox, WorkerFailed, WorkerReady
from .journal import JOURNAL_FILE_NAME, Journal
from .plan import SessionSpec, build, specify
from .preflight import COUNTDOWN_SECONDS, AccountLock, PreflightPlan, check_preflight, countdown
from .venue_map import VenueMap
from .worker import JournalTarget, OrderWorker

logger = logging.getLogger(LOGGER_NAME)

#: Seconds between two preflights while the broker's login is awaited before the open
#: (16-session-design 3.2).
LOGIN_RETRY_EVERY_S = 60.0
#: The IST time from which a broker still not logged in is alerted (08-live-trading 7).
LOGIN_ALERT_AT = "09:00"
#: Seconds the main thread waits for the order worker's startup (3.1 step 8).
WORKER_READY_TIMEOUT_S = 30.0
#: The product of every order of this release (Tier 1).
PRODUCT = "MIS"
#: The entry lock of the start phase (3.1 steps 11 to 15).
STARTING_LOCK = "starting"
#: The folder of the day's logs, under the account's folder.
LOGS_FOLDER = "logs"

_NS = NANOS_PER_SECOND


@dataclasses.dataclass(frozen=True)
class StartSettings:
    """How a session runs: its clock (`now`, UTC nanoseconds) and wait (`sleep`),
    `threads` False for the test seam, `emit` the operator's console, `stop` its
    Ctrl+C, `wait_max_s` the loop's longest wait, and the test seam's `clients` and
    `feed` (None: derived from the user's client)."""

    now: Now
    sleep: Sleep
    threads: bool
    emit: Callable[[str], None]
    stop: StopFlag
    wait_max_s: float
    clients: SessionClients | None = None
    feed: FeedSource | None = None


def open_session(
    strategy: type[Strategy] | Sequence[type[Strategy]],
    client: Any,
    settings: StartSettings,
    arguments: Mapping[str, Any],
) -> Session:
    """Steps 1 to 11 (module docstring): the loop of a session ready to start trading.
    Raises what each step raises, after closing what the steps before it opened."""
    now = settings.now
    spec = specify(strategy, clock=now, **arguments)
    stack = contextlib.ExitStack()
    try:
        return _open(spec, client, settings, stack)
    except BaseException:
        stack.close()
        raise


def _open(
    spec: SessionSpec, client: Any, settings: StartSettings, stack: contextlib.ExitStack
) -> Session:
    now, options = settings.now, spec.arguments
    clients = settings.clients
    if clients is None:
        clients = stack.enter_context(derive(client))
    shared = SharedAlerts()
    guard = ModeGuard(clients.guard, now=now, alerts=shared)
    mode = guard.start()
    names = tuple(each.name for each in spec.strategies)
    sender = stack.enter_context(
        Alerts(
            clients.alerts,
            username=options.alerts_to,
            mode=mode,
            now=now,
            strategies=names,
            start_sender=settings.threads,
        )
    )
    shared.attach(sender)
    broker = _broker(clients.data, options.account)
    account = spec.account_name(broker)
    stack.enter_context(
        AccountLock(account, server_url=clients.orders.base_url, folder=options.live_dir)
    )
    folder = live_root(account, options.live_dir) / account
    logs = stack.enter_context(configure_session_logs(folder / LOGS_FOLDER, spec.day, now=now))
    # The alerts close before the logs (closing them twice is safe), so what they say at
    # their end is in the day's log.
    stack.callback(sender.close)
    report = _preflight(spec, clients, mode, shared, settings)
    banner = guard.banner(report.broker)
    inbox = Inbox(now)
    target = JournalTarget(
        folder / JOURNAL_FILE_NAME, account, report.broker, mode, spec.day, DEFAULT_BUSY_TIMEOUT_S
    )
    worker = OrderWorker(
        clients.orders,
        guard=guard,
        alerts=shared,
        inbox=inbox,
        journal=target,
        now=now,
        sleep=settings.sleep,
    )
    stack.callback(worker.close)
    journal_session = _start_worker(worker, inbox, settings)
    journal = stack.enter_context(
        Journal.open(
            target.path,
            account=account,
            broker=report.broker,
            mode=mode,
            now=now,
            busy_timeout_s=DEFAULT_BUSY_TIMEOUT_S,
        )
    )
    snapshot = journal.snapshot(day=spec.day)
    _refuse_restart(journal_session.starts, snapshot, journal.queued(day=spec.day))
    saved = {name: kept.state for name, kept in snapshot.user_states.items()}
    plan = build(spec, clients.data, facts=report.instruments, clock=now, user_states=saved)
    parts = SessionParts(
        spec=spec,
        plan=plan,
        mode=mode,
        broker=report.broker,
        account=account,
        banner=banner,
        report=report,
        clients=clients,
        guard=guard,
        alerts=shared,
        sender=sender,
        logs=logs,
        journal=journal,
        journal_session=journal_session,
        worker=worker,
        inbox=inbox,
        stack=stack,
    )
    session = Session(
        parts,
        now=now,
        sleep=settings.sleep,
        threads=settings.threads,
        stop=settings.stop,
        emit=settings.emit,
        wait_max_s=settings.wait_max_s,
    )
    session.traders = _traders(session, report.funds_available)
    for note in (*report.warnings, *plan.warnings):
        session.warn(note)
    return session


def _broker(client: Any, account: str | None) -> str:
    """The broker `ping` names (3.1 step 5). Raises LiveError when OpenAlgo does not
    answer it."""
    answer = client.ping()
    if not isinstance(answer, Mapping):
        answer = {"message": repr(answer)}
    data = answer.get("data")
    broker = data.get("broker") if isinstance(data, Mapping) else None
    if answer.get("status") == "success" and isinstance(broker, str) and broker.strip():
        return broker.strip()
    words = answer.get("message")
    raise LiveError(
        f"OpenAlgo did not answer ping with its broker ({words}): check that OpenAlgo is "
        "running, the API key is valid and the broker is logged in, then start again"
        + ("" if account is None else f" (account {account!r})")
    )


def _preflight(
    spec: SessionSpec, clients: SessionClients, mode: Mode, alerts: Any, settings: StartSettings
) -> Any:
    """Step 7, with the wait for the broker's login before the open (3.2). Raises
    PreflightError as the preflight does."""
    now, options = settings.now, spec.arguments
    plan = PreflightPlan(
        instruments=spec.instruments,
        intervals=spec.intervals,
        product=PRODUCT,
        capital=spec.capital,
        max_daily_loss=spec.max_daily_loss,
        account_max_daily_loss=options.account_max_daily_loss,
        mode=mode,
        alerts=alerts,
    )
    calendar = default_calendar()
    opens = [calendar.session(exchange, spec.day) for _, exchange in spec.instruments]
    first_open = min((bounds[0] for bounds in opens if bounds is not None), default=0)
    alert_ns, alerted = ist_instant(spec.day, LOGIN_ALERT_AT), False
    outcome = check_preflight(clients.data, plan, now=now)
    while outcome.waits_for_login and now() < first_open:
        if now() >= alert_ns and not alerted:
            alerted = True
            alerts.send(
                AlertKind.ERROR,
                "the broker is not logged in to OpenAlgo yet: "
                + "; ".join(outcome.failures)
                + ". Log in before the open; the session checks again each minute",
            )
        settings.sleep(LOGIN_RETRY_EVERY_S)
        plan = dataclasses.replace(plan, tested_alerts=outcome.check("alerts"))
        outcome = check_preflight(clients.data, plan, now=now)
    return outcome.report()


def _start_worker(worker: OrderWorker, inbox: Inbox, settings: StartSettings) -> Any:
    """Start the order worker and wait for its `WorkerReady` (3.1 step 8). Raises its
    startup's error, or LiveError when it is not ready in `WORKER_READY_TIMEOUT_S`."""
    if not settings.threads:
        return worker.start_up()
    worker.start()
    deadline_ns = settings.now() + round(WORKER_READY_TIMEOUT_S * _NS)
    while settings.now() < deadline_ns:
        envelope = inbox.take(wait_s=0.1)
        if envelope is None:
            continue
        message = envelope.message
        if isinstance(message, WorkerReady):
            return message.session
        if isinstance(message, WorkerFailed):
            raise message.error
        logger.warning("message before the order worker was ready, dropped: %r", message)
    raise LiveError(
        f"the order worker did not start within {WORKER_READY_TIMEOUT_S:g} s: the journal "
        "may be locked by another program"
    )


def _refuse_restart(starts: int, snapshot: Any, queued: Sequence[Any]) -> None:
    """A restart of the day is refused once the journal holds an order of the day:
    resuming it needs the kernel's saved state and the reconciler. A start after one that
    sent nothing is a fresh start."""
    sent = snapshot.bound_orders or snapshot.unbound or queued
    if starts > 1 and sent:
        raise LiveError(
            f"this is start {starts} of today's session in the journal, which holds the "
            "orders of an earlier start: resuming a session after a restart is not "
            "supported yet. Check the positions and orders in OpenAlgo and close them by "
            "hand, or run the session under another account name (account=)"
        )


def _traders(session: Session, funds_available: float) -> tuple[Trader, ...]:
    """Each strategy's `Trader`, its venue map numbered with the session's request IDs,
    its entries locked until trading begins and its sizing capped by the funds."""
    parts = session.parts
    calendar, day = default_calendar(), parts.spec.day
    numbers = itertools.count(1)
    traders = []
    for part in parts.plan.strategies:
        venue = VenueMap(
            part.engine,
            strategy=part.spec.name,
            instruments=part.setup.instruments,
            session=parts.journal_session,
            request_ids=numbers,
            alerts=parts.alerts,
            now=session.now,
        )
        times = calendar.intraday_times(part.spec.exchange, day)
        square_off = part.spec.plan.square_off
        square_off_ns = times.square_off if square_off is None else ist_instant(day, square_off)
        trader = Trader(
            part,
            venue,
            square_off_ns=square_off_ns,
            close_ns=times.session_close,
            now=session.now,
            event=session.event,
        )
        trader.lock(STARTING_LOCK)
        call(part.engine.cap_sizing_equity, float(funds_available))
        traders.append(trader)
    return tuple(traders)


def begin_trading(session: Session, source: FeedSource) -> None:
    """Steps 12 to 15 (module docstring). Raises SessionStopped on a stop requested
    during the countdown or the feed's connect, and LiveError when the feed does not
    connect in time."""
    parts = session.parts
    data = DataWorker(parts.clients.data, parts.inbox, day=parts.spec.day, now=session.now)
    session.data = data
    parts.stack.callback(data.close)
    if session.threads:
        data.start()
    lines = banner_lines(session)
    for line in lines:
        session.emit(line)
    session.alert(AlertKind.SESSION, "\n".join(lines[:BANNER_ALERT_LINES]))
    countdown(COUNTDOWN_SECONDS, emit=session.emit, sleep=session.pause)
    instruments = [{"symbol": t.symbol, "exchange": t.exchange} for t in session.traders]
    link = FeedLink(
        source,
        instruments=instruments,
        inbox=parts.inbox,
        updates=parts.worker.updates,
        now=session.now,
        sleep=session.pause,
        threads=session.threads,
    )
    parts.stack.callback(link.close)
    # Set before the connect: the loop turns while the feed connects, and every feed
    # event it takes goes to the link, which counts those from before it subscribed.
    session.link = link
    link.connect()
    session.watch(link)
    for trader in session.traders:
        trader.unlock(STARTING_LOCK)
    session.trading = True
    session.event("trading")


__all__ = [
    "LOGIN_ALERT_AT",
    "LOGIN_RETRY_EVERY_S",
    "LOGS_FOLDER",
    "PRODUCT",
    "STARTING_LOCK",
    "WORKER_READY_TIMEOUT_S",
    "StartSettings",
    "begin_trading",
    "open_session",
]
