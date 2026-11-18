"""What the session invariants are checked over (16-session-design 7.1, unit 14): every change
of every order at the fake OpenAlgo, recorded as it happens, and what each session run on
that fake left.

- **The order changes** (`OrderChange`): each change of an order of either book, at the
  moment the fake makes it (its own book, not the order stream, which a fault may hold or
  lose), with the client that placed the order and the net position of its instrument in
  that book right after it. Recorded from the fake's construction on, so every position
  is the sum of the fills before it.
- **The runs** (`Run`): each session that opened on the fake, in order: the session
  object, the names of its clients (`FakeOpenAlgo.client`), its event stream as it wrote
  it, each strategy's kernel state as its close stopped it (`Closing`), and the outcome
  its close returned (None for a run that crashed or whose close failed).

`trace_sessions` installs the recording for one fake and registers the invariants to run
when the test ends (`session_invariants.assert_session_invariants`). The hooks wrap the
session's start, its close, the kernel stop of each strategy at the close and the event
stream's writer, once per test (monkeypatch undoes them), and hand each run to the trace
of the fake whose clients it uses. Nothing here changes what a session does.

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
from openalgo.backtest._live._types import Mode

#: The session's client roles (`clients.SessionClients`).
ROLES = ("orders", "guard", "data", "alerts", "feed")


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
    """One session run on the fake (module docstring); `ended_at` is how many order
    changes the fake had made when its close returned (None while it has not)."""

    session: Any
    clients: frozenset[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    closing: dict[str, Closing] = field(default_factory=dict)
    outcome: Any = None
    ended_at: int | None = None

    @property
    def mode(self) -> Mode:
        return self.session.parts.mode


class SessionTrace:
    """The recording of one fake's order changes and session runs (module docstring)."""

    def __init__(self, fake: FakeOpenAlgo) -> None:
        self.fake = fake
        self.changes: list[OrderChange] = []
        self.runs: list[Run] = []
        #: The strategies a listed row leaves unprotected at the end, with the row.
        self.unprotected: dict[str, str] = {}
        #: Why the test opts out of the invariants ("" while it does not).
        self.opted_out = ""
        self._positions: dict[tuple[Mode, str, str], int] = {}
        venue = fake._venue
        publish = venue._publish

        def recorded(book: Any, change: Any) -> None:
            self._record(book.mode, change)
            publish(book, change)

        venue._publish = recorded

    def leave_unprotected(self, strategy: str, row: str) -> None:
        """The test (one of the rows 7.1 lists: F35, F36, F60, the F77 variant, F80)
        asserts `strategy`'s leg unprotected in `open_positions` and the ERROR alert that
        names it: the end's protection invariant takes that instead of a stop."""
        self.unprotected[strategy] = row

    def opt_out(self, reason: str) -> None:
        """Check no invariant for this test, for `reason` (said in the test too)."""
        self.opted_out = reason

    def session_clients(self) -> frozenset[str]:
        """The names of every client a run of this trace used."""
        return frozenset(name for run in self.runs for name in run.clients)

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
    `check(trace, fake)` when the test ends."""
    trace = SessionTrace(fake)
    hooks = getattr(session_module.open_session, "hooks", None)
    if hooks is None:
        hooks = _Hooks()
        hooks.install(monkeypatch)
    hooks.traces.append(trace)
    fake.at_test_end(lambda: check(trace, fake))
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
    whose clients it uses."""

    def __init__(self) -> None:
        self.traces: list[SessionTrace] = []
        self._runs: dict[int, Run] = {}
        self._traces: dict[int, SessionTrace] = {}
        self._open = session_module.open_session
        self._close = session_module.close_session
        self._stop_engine = close_module._stop_engine
        self._write = SessionLogs.event

    def install(self, monkeypatch: Any) -> None:
        """Wrap the session's start, close, kernel stops and event writer for the test."""
        hooks = self

        def opened(*args: Any, **kwargs: Any) -> Any:
            return hooks.opened(*args, **kwargs)

        def written(logs: Any, kind: str, **fields: Any) -> None:
            hooks.written(logs, kind, **fields)

        opened.hooks = self
        monkeypatch.setattr(session_module, "open_session", opened)
        monkeypatch.setattr(session_module, "close_session", self.closed)
        monkeypatch.setattr(close_module, "_stop_engine", self.stopped)
        monkeypatch.setattr(SessionLogs, "event", written)

    def opened(self, *args: Any, **kwargs: Any) -> Any:
        session = self._open(*args, **kwargs)
        clients = [_unwrapped(getattr(session.parts.clients, role, None)) for role in ROLES]
        made = [client for client in clients if client is not None]
        trace = next((t for t in self.traces if any(t.fake.owns(c) for c in made)), None)
        if trace is not None:
            names = frozenset(client_name(client) for client in made)
            run = Run(session, names, list(getattr(session.parts.logs, "traced", [])))
            trace.runs.append(run)
            self._runs[id(session.parts.logs)] = run
            self._traces[id(run)] = trace
        return session

    def closed(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        outcome = self._close(session, *args, **kwargs)
        run = self._runs.get(id(session.parts.logs))
        if run is not None:
            run.outcome = outcome
            run.ended_at = len(self._traces[id(run)].changes)
        return outcome

    def stopped(self, trader: Any) -> Any:
        for run in self._runs.values():
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
        else:
            logs.__dict__.setdefault("traced", []).append(line)


def _unwrapped(client: Any) -> api | None:
    """The fake's client inside a test's client double (one that keeps it as `_client`)."""
    for _ in range(5):
        if client is None or isinstance(client, api):
            return client
        client = getattr(client, "_client", None)
    return None


__all__ = ["Closing", "OrderChange", "Run", "SessionTrace", "closing_of", "trace_sessions"]
