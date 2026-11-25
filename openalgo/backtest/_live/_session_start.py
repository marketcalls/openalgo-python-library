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
   guard's banner line is read now. The later days on which OpenAlgo's holidays and the
   package calendar disagree are the event `CALENDAR_DISAGREES` once the session exists.
8. The order worker starts journal handle W, the journal's session of the day and the
   order path (on its own thread, or inline under the test seam), and the main thread
   opens journal handle L (`_session_worker`). A strategy a kept stop GTT refusal names,
   or one that holds a position or a GTT from an earlier day, is refused now, before
   anything is sent (`_preflight_carry`: PreflightError).
9. and 10. Unbound intents of the day resolved first, then the books read and each
   strategy's journal position judged against the broker's (`_restart_books`): a broker
   position on a traded symbol that the journal does not hold, or a journal position the
   broker does not hold that nothing explains, refuses the start (LiveError).
11. The plan (`plan.build`): warm-up from broker history, each strategy's kernel session,
   live series and `init`, its saved `self.state` merged, the catch-up to the present.
   Each kernel session's entries are locked ("starting") and its sizing equity capped by
   the funds available (`_session_traders`); each strategy's version and run manifest
   written to journal L (`_session_versions`); a restart's saved state restored and
   checked (`_session_restore.restore`).

A failure up to step 11 closes what was opened, in reverse, and is raised. Then
`begin_trading`: 12. the data worker starts and the loop turns (the start phase): with a
position resumed, the loop turns until its stop is confirmed, at most
`RESTART_PROTECT_WAIT_S`, before anything else; 13. the banner, a SESSION alert with its
first lines, and the countdown, during which the loop keeps turning; 14. the feed
connects and the session watches it; 15. entries are unlocked. A stop requested during
13 or 14 raises `SessionStopped`.

Times are UTC nanoseconds, amounts rupees.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..calendar import default_calendar
from ..errors import LiveError
from ..strategy import Strategy
from ._alerts_logs import SessionLogs, configure_session_logs
from ._journal_file import DEFAULT_BUSY_TIMEOUT_S
from ._preflight_carry import refuse_carried, refuse_kept
from ._preflight_lock import live_root
from ._preflight_plan import WARNING
from ._restart_books import judge_start, read_start
from ._session_banner import BANNER_ALERT_LINES, banner_lines
from ._session_figures import DayFigures
from ._session_loop import Session
from ._session_parts import SessionParts, StopFlag
from ._session_rearm import warn_at_start
from ._session_restore import RESTART_PROTECT_WAIT_S, protect_restored, restore
from ._session_trader import ist_instant
from ._session_traders import build_traders
from ._session_versions import record_versions
from ._session_worker import start_worker
from ._shared_alerts import SharedAlerts
from ._types import AlertKind, Mode, Now, Sleep
from .alerts import Alerts
from .clients import SessionClients, derive
from .data_worker import DataWorker
from .feedlink import FeedLink, FeedSource
from .guard import ModeGuard
from .inbox import Inbox
from .journal import JOURNAL_FILE_NAME
from .plan import SessionSpec, build, specify
from .preflight import COUNTDOWN_SECONDS, AccountLock, PreflightPlan, check_preflight, countdown
from .worker import JournalTarget

#: Seconds between two preflights while the broker's login is awaited before the open
#: (16-session-design 3.2).
LOGIN_RETRY_EVERY_S = 60.0
#: The IST time from which a broker still not logged in is alerted (08-live-trading 7).
LOGIN_ALERT_AT = "09:00"
#: The entry lock of the start phase (3.1 steps 11 to 15).
STARTING_LOCK = "starting"
#: The folder of the day's logs, under the account's folder.
LOGS_FOLDER = "logs"
#: The event naming the later days OpenAlgo's holidays and the package calendar disagree
#: on (18-multi-strategy 12.2; `_preflight_calendar`).
CALENDAR_DISAGREES = "calendar_disagrees"


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


@dataclasses.dataclass(frozen=True)
class _Opened:
    """What steps 2 to 6 opened: the session's `clients`, the `shared` alerts and their
    `sender`, the mode `guard` and the `mode` it adopted, the `account` and its `folder`,
    and the day's `logs`."""

    clients: SessionClients
    shared: SharedAlerts
    sender: Alerts
    guard: ModeGuard
    mode: Mode
    account: str
    folder: pathlib.Path
    logs: SessionLogs


def _open(
    spec: SessionSpec, client: Any, settings: StartSettings, stack: contextlib.ExitStack
) -> Session:
    """Steps 2 to 11 (module docstring) for the checked `spec`, what each step opens closed
    by `stack`."""
    opened = _open_account(spec, client, settings, stack)
    report = _preflight(spec, opened.clients, opened.mode, opened.shared, settings)
    banner = opened.guard.banner(report.broker)
    inbox = Inbox(settings.now)
    path = opened.folder / JOURNAL_FILE_NAME
    target = JournalTarget(
        path, opened.account, report.broker, opened.mode, spec.day, DEFAULT_BUSY_TIMEOUT_S
    )
    started = start_worker(
        opened.clients.orders,
        guard=opened.guard,
        alerts=opened.shared,
        inbox=inbox,
        target=target,
        now=settings.now,
        sleep=settings.sleep,
        threads=settings.threads,
        orders_per_second=spec.arguments.max_orders_per_second,
        read_holdings=spec.reads_holdings,
        gtt_legs=spec.gtt_legs,
        stack=stack,
    )
    journal = started.journal
    refuse_kept(journal.kept_refusals(), spec.positional)
    names = tuple(each.name for each in spec.strategies)
    refuse_carried(journal.day_closes(), journal.open_gtts(), names, spec.day)
    start = read_start(started.worker, inbox, settings, journal, spec.day, opened.shared)
    restart = judge_start(spec, journal, start, opened.shared)
    snapshot = journal.snapshot(day=spec.day)
    saved = {name: kept.state for name, kept in snapshot.user_states.items()}
    data = opened.clients.data
    plan = build(spec, data, facts=report.instruments, clock=settings.now, user_states=saved)
    parts = SessionParts(
        spec=spec,
        plan=plan,
        mode=opened.mode,
        broker=report.broker,
        account=opened.account,
        banner=banner,
        report=report,
        clients=opened.clients,
        guard=opened.guard,
        alerts=opened.shared,
        sender=opened.sender,
        logs=opened.logs,
        journal=journal,
        journal_session=started.journal_session,
        worker=started.worker,
        inbox=inbox,
        stack=stack,
    )
    return _ready(parts, settings, start, restart, snapshot)


def _open_account(
    spec: SessionSpec, client: Any, settings: StartSettings, stack: contextlib.ExitStack
) -> _Opened:
    """Steps 2 to 6 (module docstring): the clients, the mode, the alerts, the account and
    server locks and the day's logs, each closed by `stack`. Raises what each step
    raises."""
    now, options = settings.now, spec.arguments
    clients = settings.clients
    if clients is None:
        clients = stack.enter_context(derive(client))
    # Every alert is echoed on the operator's console, with or without a Telegram user.
    shared = SharedAlerts(echo=settings.emit)
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
    return _Opened(clients, shared, sender, guard, mode, account, folder, logs)


def _ready(
    parts: SessionParts, settings: StartSettings, start: Any, restart: Any, snapshot: Any
) -> Session:
    """The end of step 11 (module docstring): the loop over `parts`, each strategy's kernel
    session made and its entries locked ("starting"), the start's audit records, the day's
    figures from the journal's `snapshot`, what the worker posted during the `start`'s
    reads, the `restart`'s saved state restored, and the start's warnings. Raises
    JournalError when the journal cannot write the records, and what the restore raises."""
    session = Session(
        parts,
        now=settings.now,
        sleep=settings.sleep,
        threads=settings.threads,
        stop=settings.stop,
        emit=settings.emit,
        wait_max_s=settings.wait_max_s,
    )
    report = parts.report
    session.traders = build_traders(session, report.funds_available)
    for trader in session.traders:
        trader.lock(STARTING_LOCK)
    _audit_records(session, report)
    names = tuple(each.name for each in parts.spec.strategies)
    session.figures = DayFigures.at_start(snapshot.rms_states, names)
    for envelope in start.deferred:
        # What the worker posted during the start's reads, for the loop to take.
        parts.inbox.put(envelope.message)
    restore(session, restart, settings.sleep)
    # The preflight marks its warnings with a prefix of its own; the console adds one.
    for note in (*(note.removeprefix(WARNING) for note in report.warnings), *parts.plan.warnings):
        session.warn(note)
    session.risk.warn_at_start()
    return session


def _audit_records(session: Session, report: Any) -> None:
    """The start's audit records (18-multi-strategy 12): each strategy's version and run
    manifest written to journal L (`_session_versions`), and the event naming the later
    days on which OpenAlgo's holidays and the package calendar disagree (the preflight's
    `report`). Raises JournalError when the journal cannot write them."""
    session.versions = record_versions(session)
    if report.calendar_disagrees:
        session.event(CALENDAR_DISAGREES, days=list(report.calendar_disagrees))


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
        product=spec.product,
        capital=spec.capital,
        max_daily_loss=spec.max_daily_loss,
        account_max_daily_loss=options.account_max_daily_loss,
        mode=mode,
        alerts=alerts,
        contracts=[i for each in spec.strategies for i in each.instruments if i.expiry is not None],
        gtt=bool(spec.gtt_legs),
        positional_stops=spec.positional_stops,
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
    if any(trader.open_units() for trader in session.traders):
        protect_restored(session, RESTART_PROTECT_WAIT_S)
    warn_at_start(session)
    lines = banner_lines(session)
    for line in lines:
        session.emit(line)
    session.banner_shown = True
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
    "CALENDAR_DISAGREES",
    "LOGIN_ALERT_AT",
    "LOGIN_RETRY_EVERY_S",
    "LOGS_FOLDER",
    "STARTING_LOCK",
    "StartSettings",
    "begin_trading",
    "open_session",
]
