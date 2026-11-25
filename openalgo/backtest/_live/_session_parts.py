"""What the start of a live session hands its loop (16-session-design 3.1; unit 8): the
parts the start built (`SessionParts`), the session's Ctrl+C (`StopFlag`), and the signal
that a stop was asked for during the start (`SessionStopped`). Times are UTC
nanoseconds.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from ._alerts_logs import SessionLogs
from ._carry_start import CarryAtStart
from ._journal_reads import JournalSession
from ._preflight_plan import PreflightReport
from ._shared_alerts import SharedAlerts
from ._types import Mode
from .alerts import Alerts
from .clients import SessionClients
from .guard import ModeGuard
from .inbox import Inbox
from .journal import Journal
from .plan import SessionPlan, SessionSpec
from .worker import OrderWorker


class SessionStopped(Exception):
    """The stop flag was set during a wait of the start (the countdown, the feed's
    connect): the session stops safely from there."""


class StopFlag:
    """Ctrl+C for a session: `request` counts a request (the signal handler calls it, as
    a test does); the loop reads `count`. No lock: a signal handler runs on the main
    thread between two of its instructions, and only it writes."""

    def __init__(self) -> None:
        self._count = 0

    def request(self) -> None:
        """Ask the session to stop (a second call is the second Ctrl+C)."""
        self._count += 1

    @property
    def count(self) -> int:
        """How many stops were asked for."""
        return self._count


@dataclass(frozen=True)
class SessionParts:
    """What the start sequence built (16-session-design 3.1 steps 1 to 11), owned by the
    loop: the resolved `spec` and the built `plan`, the `mode` adopted, the `broker` and
    `account`, the guard's `banner` line and the preflight's `report`, the `clients`,
    the `guard`, the shared `alerts` and their `sender`, the session `logs`, journal
    handle L (`journal`) and its `journal_session`, the order `worker`, the loop's
    `inbox`, `stack`, which closes what was opened, in reverse, and `carry`, what the
    session continues from earlier days (`_carry_start`; None for a part built without
    one, which continues nothing)."""

    spec: SessionSpec
    plan: SessionPlan
    mode: Mode
    broker: str
    account: str
    banner: str
    report: PreflightReport
    clients: SessionClients
    guard: ModeGuard
    alerts: SharedAlerts
    sender: Alerts
    logs: SessionLogs
    journal: Journal
    journal_session: JournalSession
    worker: OrderWorker
    inbox: Inbox
    stack: contextlib.ExitStack
    carry: CarryAtStart | None = None


__all__ = ["SessionParts", "SessionStopped", "StopFlag"]
