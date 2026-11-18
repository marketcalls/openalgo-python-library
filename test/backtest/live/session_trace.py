"""What the session invariants are checked over (16-session-design 7.1, unit 14): every change
of every order at the fake OpenAlgo, recorded as it happens, and what each session run on
that fake left.

- **The order changes** (`OrderChange`): each change of an order of either book, at the
  moment the fake makes it (its own book, not the order stream, which a fault may hold or
  lose, and which never carries a sandbox modify), with the client that placed the order
  and the net position of its instrument in that book right after it. Recorded from the
  fake's construction on, so every position is the sum of the fills before it.
- **The runs** (`Run`): each session that opened on the fake, in order: the session
  object, the names of its clients (`FakeOpenAlgo.client`), its event stream and its
  alerts as it sent them, where the fake's log stood when it opened, when its mode guard
  locked and when it closed, each strategy's kernel state as its close stopped it
  (`Closing`), and the outcome its close returned (None for a run that crashed or whose
  close failed).

`trace_sessions` installs the recording for one fake and registers the invariants to run
when the test ends (`session_invariants.assert_session_invariants`). The hooks wrap the
session's start, its close, the kernel stop of each strategy at the close, the event
stream's writer, the shared alerts' sender and the mode guard's lock, once per test
(monkeypatch undoes them), and hand each run to the trace of the fake whose clients it
uses. Nothing here changes what a session does.

A test whose session cannot keep one of the checks says which and why (`SessionTrace.waive`);
one whose row leaves a leg without a confirmed stop names the leg
(`SessionTrace.leave_unprotected`).

Times are UTC nanoseconds, quantities signed units, prices rupees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fake_openalgo import FakeOpenAlgo, client_name

from openalgo import api
from openalgo.backtest._engine import call
from openalgo.backtest._live import _session_close as close_module
from openalgo.backtest._live import session as session_module
from openalgo.backtest._live._alerts_logs import SessionLogs
from openalgo.backtest._live._shared_alerts import SharedAlerts
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.guard import ModeGuard

#: The session's client roles (`clients.SessionClients`).
ROLES = ("orders", "guard", "data", "alerts", "feed")

#: The checks of 7.1 (`session_invariants`), by the name a waiver gives.
ONE_STOP = "one stop per leg"
WITHIN = "exits within the position"
OTHER_SIDE = "never the other side"
NO_ORPHAN = "no orphan"
PROTECTED = "protected"
BOOKS_AGREE = "books agree"
ANSWERED = "every modify and cancel answered"
GUARD_LOCK = "nothing after a guard lock"
PRICED = "priced fills"
CHECKS = frozenset(
    {ONE_STOP, WITHIN, OTHER_SIDE, NO_ORPHAN, PROTECTED, BOOKS_AGREE, ANSWERED, GUARD_LOCK, PRICED}
)
#: The rows 7.1 lists as leaving a leg without a confirmed stop at the end.
LISTED_ROWS = frozenset({"F35", "F36", "F60", "F77 variant", "F80"})


@dataclass(frozen=True)
class OrderChange:
    """An order as it stood right after one change at the fake (module docstring): the
    change's place in the sequence, its time, the book's mode, the order's terms and
    state, the units and price of the fill the change made (0 for none), and the net
    position of the order's instrument in that book after it."""

    seq: int
    ts_ns: int
    mode: Mode
    order_id: str
    client: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    price_type: str
    status: str
    quantity: int
    filled: int
    fill_units: int
    fill_price: float
    position: int

    @property
    def working(self) -> bool:
        """True while the order can still fill."""
        return self.status in ("open", "trigger pending")

    @property
    def remaining(self) -> int:
        """The units not filled yet."""
        return self.quantity - self.filled

    @property
    def instrument(self) -> tuple[Mode, str, str]:
        return (self.mode, self.symbol, self.exchange)


@dataclass
class Closing:
    """A strategy's kernel state as the close stopped its kernel session: its signed units,
    whether its leg wants a broker-held stop (a stop level is set), its working orders'
    broker order IDs (those bound) and how many work with none bound yet (sent with no
    answer, or not sent), and the units its stop covers at the broker as the stops
    supervisor confirmed it working (`StopsSupervisor.covered`: 0 for none)."""

    strategy: str
    symbol: str
    exchange: str
    units: float
    wants_stop: bool
    working_order_ids: frozenset[str]
    stop_covers: int
    unbound_working: int = 0


@dataclass
class Run:
    """One session run on the fake (module docstring). `ended_at` is how many order changes
    the fake had made when its close returned (None while it has not); `log_from` and
    `log_to` are how many entries the fake's log held when the run opened and when its
    close returned; `locked_at` how many it held when the run's mode guard locked (None:
    it never did). `alerts` are the (kind, text) of each alert the run sent."""

    session: Any
    clients: frozenset[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[tuple[Any, str]] = field(default_factory=list)
    closing: dict[str, Closing] = field(default_factory=dict)
    outcome: Any = None
    ended_at: int | None = None
    log_from: int = 0
    log_to: int | None = None
    locked_at: int | None = None

    @property
    def mode(self) -> Mode:
        return self.session.parts.mode


class SessionTrace:
    """The recording of one fake's order changes and session runs (module docstring)."""

    def __init__(self, fake: FakeOpenAlgo) -> None:
        self.fake = fake
        self.changes: list[OrderChange] = []
        self.runs: list[Run] = []
        #: The strategies whose leg the test leaves without a confirmed stop, with the
        #: listed row or the decided end that leaves it so.
        self.unprotected: dict[str, str] = {}
        #: The checks the test's session cannot keep, with why.
        self.waived: dict[str, str] = {}
        #: Whether the invariants were checked already (`check_now`).
        self.checked = False
        self._positions: dict[tuple[Mode, str, str], int] = {}
        self._last: Any = None
        venue = fake._venue
        publish = venue._publish

        def published(book: Any, change: Any) -> None:
            # A change the book's modify returned is recorded already (`_record_modifies`).
            if change is not self._last:
                self._record(book.mode, change)
            publish(book, change)

        venue._publish = published
        for book in venue.books.values():
            self._record_modifies(book)

    def leave_unprotected(self, strategy: str, row: str) -> None:
        """The test (one of the rows 7.1 lists: F35, F36, F60, the F77 variant, F80)
        asserts `strategy`'s leg unprotected in `open_positions` and the ERROR alert that
        names it: the end's protection invariant takes that instead of a stop. Raises
        ValueError for a row 7.1 does not list (`left_unprotected_by_decision`)."""
        if row not in LISTED_ROWS:
            raise ValueError(f"7.1 lists {sorted(LISTED_ROWS)}, not {row!r}")
        self.unprotected[strategy] = row

    def left_unprotected_by_decision(self, strategy: str, decision: str) -> None:
        """As `leave_unprotected`, for a decided end of 16-session-design that leaves the
        leg without a confirmed stop (`decision` names the paragraph); each is listed in
        FAULT_ROWS.md for the lead to add to 7.1's list."""
        self.unprotected[strategy] = decision

    def waive(self, check: str, reason: str) -> None:
        """Do not assert `check` (one of `CHECKS`) for this test, for `reason` (said in the
        test too); every other check still runs. Raises ValueError for an unknown check."""
        if check not in CHECKS:
            raise ValueError(f"no check {check!r}: use one of {sorted(CHECKS)}")
        self.waived[check] = reason

    def check_now(self, check: Callable[[SessionTrace, FakeOpenAlgo], None]) -> None:
        """Run `check` over this trace now, inside the test, and not again when it ends:
        for a test marked as a known break of an invariant."""
        self.checked = True
        check(self, self.fake)

    def session_clients(self) -> frozenset[str]:
        """The names of every client a run of this trace used."""
        return frozenset(name for run in self.runs for name in run.clients)

    def _record_modifies(self, book: Any) -> None:
        """Record each modify of `book` as the book makes it: the sandbox's is never
        published (`Venue.modify`), and the broker's is published right after."""
        modify = book.modify

        def modified(order: Any, **terms: Any) -> Any:
            change = modify(order, **terms)
            self._record(book.mode, change)
            self._last = change
            return change

        book.modify = modified

    def _record(self, mode: Mode, change: Any) -> None:
        order, fill = change.order, change.fill
        key = (mode, order.symbol, order.exchange)
        units = 0 if fill is None else fill.quantity
        if units:
            signed = units if order.action == "BUY" else -units
            self._positions[key] = self._positions.get(key, 0) + signed
        self.changes.append(
            OrderChange(
                len(self.changes),
                self.fake.clock.now(),
                mode,
                order.order_id,
                order.client,
                order.strategy,
                order.symbol,
                order.exchange,
                order.action,
                order.price_type,
                order.status,
                order.quantity,
                order.filled,
                units,
                0.0 if fill is None else fill.price,
                self._positions.get(key, 0),
            )
        )


def trace_sessions(
    fake: FakeOpenAlgo, monkeypatch: Any, check: Callable[..., None]
) -> SessionTrace:
    """Record `fake`'s order changes and session runs (module docstring), and run
    `check(trace, fake)` when the test ends (unless the test ran it, `check_now`)."""
    trace = SessionTrace(fake)
    hooks = getattr(session_module.open_session, "hooks", None)
    if hooks is None:
        hooks = _Hooks()
        hooks.install(monkeypatch)
    hooks.traces.append(trace)
    fake.at_test_end(lambda: None if trace.checked else check(trace, fake))
    return trace


def closing_of(session: Any, trader: Any) -> Closing | None:
    """`trader`'s `Closing` read from its running kernel session and `session`'s stops
    supervisor, None when the kernel cannot be read (a test's hook broke it)."""
    try:
        units = trader.open_units()
        state = call(trader.engine.live_state)
        working = set(trader.working())
        bound = {kernel: placed.order_id for kernel, placed in trader.venue.placed()}
        covers = int(session.stops.covered(trader))
    # Any failure of the kernel: the end checks then take what the fake shows.
    except Exception:
        return None
    wants = any(leg.get("stop") is not None for leg in state.get("legs", []))
    order_ids = frozenset(bound[k] for k in working if bound.get(k))
    unbound = len(working) - len(order_ids)
    return Closing(
        trader.name, trader.symbol, trader.exchange, units, wants, order_ids, covers, unbound
    )


class _Hooks:
    """The hooks of one test (module docstring): each run goes to the trace of the fake
    whose clients it uses. What a part sends before its run is known (the start's alerts,
    a guard that locks in the start) waits on the part (`traced`) until the run opens."""

    def __init__(self) -> None:
        self.traces: list[SessionTrace] = []
        self._runs: dict[int, Run] = {}
        self._traces: dict[int, SessionTrace] = {}
        self._open = session_module.open_session
        self._close = session_module.close_session
        self._stop_engine = close_module._stop_engine
        self._write = SessionLogs.event
        self._send = SharedAlerts.send
        self._lock = ModeGuard._lock

    def install(self, monkeypatch: Any) -> None:
        """Wrap the session's start, close, kernel stops, event writer, alerts and guard
        lock for the test."""
        hooks = self

        def opened(*args: Any, **kwargs: Any) -> Any:
            return hooks.opened(*args, **kwargs)

        def written(logs: Any, kind: str, **fields: Any) -> None:
            hooks.written(logs, kind, **fields)

        def sent(alerts: Any, kind: Any, text: str, **kwargs: Any) -> None:
            hooks.sent(alerts, kind, text, **kwargs)

        def locked(guard: Any, *args: Any, **kwargs: Any) -> None:
            hooks.locked(guard, *args, **kwargs)

        opened.hooks = self
        monkeypatch.setattr(session_module, "open_session", opened)
        monkeypatch.setattr(session_module, "close_session", self.closed)
        monkeypatch.setattr(close_module, "_stop_engine", self.stopped)
        monkeypatch.setattr(SessionLogs, "event", written)
        monkeypatch.setattr(SharedAlerts, "send", sent)
        monkeypatch.setattr(ModeGuard, "_lock", locked)

    def opened(self, *args: Any, **kwargs: Any) -> Any:
        session = self._open(*args, **kwargs)
        parts = session.parts
        clients = [_unwrapped(getattr(parts.clients, role, None)) for role in ROLES]
        made = [client for client in clients if client is not None]
        trace = next((t for t in self.traces if any(t.fake.owns(c) for c in made)), None)
        if trace is not None:
            names = frozenset(client_name(client) for client in made)
            run = Run(
                session,
                names,
                list(getattr(parts.logs, "traced", [])),
                list(getattr(parts.alerts, "traced", [])),
                log_from=getattr(parts.logs, "traced_log_from", len(trace.fake.log.entries)),
                locked_at=getattr(parts.guard, "traced_lock", {}).get(id(trace)),
            )
            trace.runs.append(run)
            self._runs[id(parts.logs)] = run
            self._runs[id(parts.alerts)] = run
            self._runs[id(parts.guard)] = run
            self._traces[id(run)] = trace
        return session

    def closed(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        outcome = self._close(session, *args, **kwargs)
        run = self._runs.get(id(session.parts.logs))
        if run is not None:
            run.outcome = outcome
            trace = self._traces[id(run)]
            run.ended_at = len(trace.changes)
            run.log_to = len(trace.fake.log.entries)
        return outcome

    def stopped(self, trader: Any) -> Any:
        for run in {id(run): run for run in self._runs.values()}.values():
            if any(t is trader for t in run.session.traders):
                closing = closing_of(run.session, trader)
                if closing is not None:
                    run.closing[trader.name] = closing
        return self._stop_engine(trader)

    def written(self, logs: Any, kind: str, **fields: Any) -> None:
        self._write(logs, kind, **fields)
        line = {"ts_ns": logs._now(), "kind": kind, **fields}
        run = self._runs.get(id(logs))
        if run is not None:
            run.events.append(line)
            return
        logs.__dict__.setdefault("traced", []).append(line)
        if "traced_log_from" not in logs.__dict__ and self.traces:
            # The log as the session's first event found it: its run's calls come after.
            logs.traced_log_from = min(len(t.fake.log.entries) for t in self.traces)

    def sent(self, alerts: Any, kind: Any, text: str, **kwargs: Any) -> None:
        run = self._runs.get(id(alerts))
        if run is not None:
            run.alerts.append((kind, text))
        else:
            alerts.__dict__.setdefault("traced", []).append((kind, text))
        self._send(alerts, kind, text, **kwargs)

    def locked(self, guard: Any, *args: Any, **kwargs: Any) -> None:
        self._lock(guard, *args, **kwargs)
        run = self._runs.get(id(guard))
        if run is not None:
            trace = self._traces[id(run)]
            run.locked_at = len(trace.fake.log.entries)
        else:
            guard.traced_lock = {id(t): len(t.fake.log.entries) for t in self.traces}


def _unwrapped(client: Any) -> api | None:
    """The fake's client inside a test's client double (one that keeps it as `_client`)."""
    for _ in range(5):
        if client is None or isinstance(client, api):
            return client
        client = getattr(client, "_client", None)
    return None


__all__ = [
    "ANSWERED",
    "BOOKS_AGREE",
    "CHECKS",
    "GUARD_LOCK",
    "LISTED_ROWS",
    "NO_ORPHAN",
    "ONE_STOP",
    "OTHER_SIDE",
    "PRICED",
    "PROTECTED",
    "WITHIN",
    "Closing",
    "OrderChange",
    "Run",
    "SessionTrace",
    "closing_of",
    "trace_sessions",
]
