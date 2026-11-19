"""The hooks that hand each session run on the fake OpenAlgo to its trace (`session_trace`,
16-session-design 7.1, unit 14).

`trace_sessions` installs the recording for one fake and registers the invariants to run
when the test ends (`session_invariants.assert_session_invariants`). The hooks wrap the
session's start, its close, the kernel stop of each strategy at the close, the event
stream's writer, the shared alerts' sender and the mode guard's lock, once per test
(monkeypatch undoes them), and hand each run to the trace of the fake whose clients it
uses. Nothing here changes what a session does.

Where the fake's log stood when the mode guard locked is taken as the lock begins: the
call that showed the change is in the log by then, and anything the session's clients
call from then on, while the lock sends its MODE_BREACH alert included, is after the lock
(unit 14 round 2, finding 3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fake_openalgo import FakeOpenAlgo, client_name
from session_trace import Run, SessionTrace, closing_of

from openalgo import api
from openalgo.backtest._live import _session_close as close_module
from openalgo.backtest._live import session as session_module
from openalgo.backtest._live._alerts_logs import SessionLogs
from openalgo.backtest._live._shared_alerts import SharedAlerts
from openalgo.backtest._live.guard import ModeGuard

#: The session's client roles (`clients.SessionClients`).
ROLES = ("orders", "guard", "data", "alerts", "feed")
#: How many client doubles deep a test may wrap the fake's client (`_unwrapped`).
WRAPPED_AT_MOST = 5


def trace_sessions(
    fake: FakeOpenAlgo, monkeypatch: Any, check: Callable[..., None]
) -> SessionTrace:
    """Record `fake`'s order changes and session runs (module docstring), and run
    `check(trace, fake)` when the test ends (unless the test ran it, `check_now`)."""
    trace = SessionTrace(fake)
    hooks = getattr(session_module.open_session, "hooks", None)
    if hooks is None:
        hooks = Hooks()
        hooks.install(monkeypatch)
    hooks.traces.append(trace)
    fake.at_test_end(lambda: None if trace.checked else check(trace, fake))
    return trace


class Hooks:
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
        """Note where the fake's log stands as the lock begins (module docstring), then
        lock; a guard locks once, so the first note stands."""
        run = self._runs.get(id(guard))
        if run is not None:
            if run.locked_at is None:
                run.locked_at = len(self._traces[id(run)].fake.log.entries)
        elif "traced_lock" not in guard.__dict__:
            guard.traced_lock = {id(t): len(t.fake.log.entries) for t in self.traces}
        self._lock(guard, *args, **kwargs)


def _unwrapped(client: Any) -> api | None:
    """The fake's client inside a test's client double (one that keeps it as `_client`)."""
    for _ in range(WRAPPED_AT_MOST):
        if client is None or isinstance(client, api):
            return client
        client = getattr(client, "_client", None)
    return None


__all__ = ["ROLES", "Hooks", "trace_sessions"]
