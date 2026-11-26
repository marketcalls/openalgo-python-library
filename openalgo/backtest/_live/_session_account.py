"""Steps 2 to 6 of a live session's start (16-session-design 3.1; unit 8), split from
`_session_start` so each module keeps one job:

2. The session's clients (`clients.derive`), unless given (the test seam).
3. The mode: the guard reads OpenAlgo's analyzer switch, and the session adopts it.
4. The alerts, on their own client, shared by every thread.
5. `ping` names the broker (the default account name); the account and server locks.
6. The day's logs under the account's folder (`LOGS_FOLDER`).

What each step opens is closed by the start's exit stack. Used by the main thread at the
start.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..errors import LiveError
from ._alerts_logs import SessionLogs, configure_session_logs
from ._preflight_lock import live_root
from ._shared_alerts import SharedAlerts
from ._types import Mode
from .alerts import Alerts
from .clients import SessionClients, derive
from .guard import ModeGuard
from .plan import SessionSpec
from .preflight import AccountLock

if TYPE_CHECKING:
    from ._session_start import StartSettings

#: The folder of the day's logs, under the account's folder.
LOGS_FOLDER = "logs"


@dataclasses.dataclass(frozen=True)
class Opened:
    """What steps 2 to 6 opened: the session's `clients`, the `shared` alerts and their
    `sender`, the mode `guard` and the `mode` it adopted, the `account` and its `folder`,
    the day's `logs`, and the `broker` `ping` named."""

    clients: SessionClients
    shared: SharedAlerts
    sender: Alerts
    guard: ModeGuard
    mode: Mode
    account: str
    folder: pathlib.Path
    logs: SessionLogs
    broker: str


def open_account(
    spec: SessionSpec, client: Any, settings: StartSettings, stack: contextlib.ExitStack
) -> Opened:
    """Steps 2 to 6 (module docstring) for the checked `spec`, through the user's
    `client` with the start's `settings`, each part closed by `stack`. Raises what each
    step raises."""
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
    return Opened(clients, shared, sender, guard, mode, account, folder, logs, broker)


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


__all__ = ["LOGS_FOLDER", "Opened", "open_account"]
