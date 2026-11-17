"""One lock around a live session's alerts for the threads that send them
(16-session-design 2.8).

`Alerts.send` is not documented thread-safe (it keeps the repeat window, the sequence of
the alerts and the queue's bookkeeping), and a session sends from more than one thread:
the loop, and the order worker through its order path and its mode guard. Every module
is handed the one `SharedAlerts`, which implements `AlertsLike` and calls the attached
alerts' `send` under one lock.

The mode guard is made before the alerts can be (the alerts name the mode the guard
reads, 3.1 steps 3 and 4), so the shared alerts start with nothing attached: until
`attach`, an alert goes to the log only, as a warning.

Every alert is also echoed on the operator's console (`echo`), whether or not it goes to
Telegram, so a session without an OpenAlgo username is not silent once it trades: one
line (or more, for the daily summary) starting with the alert's kind in capitals
(`echo_line`). A SESSION alert is not echoed: it carries the start banner's first lines,
printed already; nor is a repeated condition the alerts hold back within their repeat
window. A caller that prints its own line for an alert passes `echo=False`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..errors import ConfigError, LiveError
from ._types import LOGGER_NAME, AlertKind, AlertsLike

logger = logging.getLogger(LOGGER_NAME)

#: The alert kinds the console does not echo (module docstring).
NOT_ECHOED = frozenset({AlertKind.SESSION})


def echo_line(kind: AlertKind, text: str) -> str:
    """An alert as the console shows it: "ENTRY: BUY 10 SBIN NSE at 500.00 (...)"."""
    return f"{kind.name.replace('_', ' ')}: {text}"


class SharedAlerts:
    """The session's alerts for every thread (implements `AlertsLike`): `send` forwards
    to the attached alerts under one lock, so two threads never run inside them at once,
    and echoes the alert on the console through `echo` (a callable taking one line; None
    for no echo). `attach` once, from the main thread, when the alerts are entered."""

    def __init__(self, *, echo: Callable[[str], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._alerts: AlertsLike | None = None
        self._echo = echo

    @property
    def attached(self) -> bool:
        """True once `attach` has given the session's alerts."""
        with self._lock:
            return self._alerts is not None

    def attach(self, alerts: AlertsLike) -> None:
        """Forward every later alert to `alerts` (the session's `alerts.Alerts`).

        Raises ConfigError for an object without `send` and `test`, and LiveError when
        alerts are attached already: a session has one set of alerts, and a second would
        split its repeat window and its counts.
        """
        if not all(callable(getattr(alerts, name, None)) for name in ("send", "test")):
            raise ConfigError(
                f"the shared alerts forward to an alerts.Alerts, which has send and test; "
                f"got {type(alerts).__name__}"
            )
        with self._lock:
            if self._alerts is not None:
                raise LiveError(
                    "the session's alerts are attached already: attach them once, when the "
                    "alerts are entered"
                )
            self._alerts = alerts

    def send(self, kind: AlertKind, text: str, *, echo: bool = True) -> None:
        """Send an alert through the attached alerts, under the lock; before `attach`, log
        it as a warning instead. Then echo it on the console unless `echo` is False or its
        kind is not echoed (module docstring). Never raises, as the attached alerts never
        do: an echo that fails is logged."""
        with self._lock:
            alerts = self._alerts
            if alerts is not None:
                suppressed = _suppressed(alerts)
                alerts.send(kind, text)
                # A repeated condition the alerts held back is not printed again either.
                echo = echo and _suppressed(alerts) == suppressed
            else:
                name = kind.value if isinstance(kind, AlertKind) else repr(kind)
                logger.warning(
                    "alert %s before the session's alerts were set up, so it goes to the log "
                    "only: %s",
                    name,
                    text,
                )
            if echo and self._echo is not None and kind not in NOT_ECHOED:
                self._echoed(kind, text)

    def _echoed(self, kind: AlertKind, text: str) -> None:
        shown = kind if isinstance(kind, AlertKind) else AlertKind.ERROR
        try:
            self._echo(echo_line(shown, text))
        # The console must never stop an alert or the thread that sent it.
        except Exception:
            logger.exception("an alert could not be printed on the console")

    def test(self) -> bool:
        """The attached alerts' test alert: True only when it was delivered; False, and a
        warning in the log, before `attach`.

        Not taken under the lock: the test waits on OpenAlgo for up to the alerts
        client's timeout, a send from another thread must not wait behind it, and
        `Alerts.test` touches only what is safe from any thread (its own client, and its
        sender's note of the failure, which has a lock of its own).
        """
        with self._lock:
            alerts = self._alerts
        if alerts is None:
            logger.warning("no test alert sent: the session's alerts are not set up yet")
            return False
        return alerts.test()


def _suppressed(alerts: AlertsLike) -> int:
    """How many repeated conditions `alerts` held back so far (`Alerts.suppressed`; 0 for
    alerts that hold none back)."""
    return int(getattr(alerts, "suppressed", 0))


__all__ = ["NOT_ECHOED", "SharedAlerts", "echo_line"]
