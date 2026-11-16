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
"""

from __future__ import annotations

import logging
import threading

from ..errors import ConfigError, LiveError
from ._types import LOGGER_NAME, AlertKind, AlertsLike

logger = logging.getLogger(LOGGER_NAME)


class SharedAlerts:
    """The session's alerts for every thread (implements `AlertsLike`): `send` forwards
    to the attached alerts under one lock, so two threads never run inside them at once.
    `attach` once, from the main thread, when the alerts are entered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alerts: AlertsLike | None = None

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

    def send(self, kind: AlertKind, text: str) -> None:
        """Send an alert through the attached alerts, under the lock; before `attach`, log
        it as a warning instead. Never raises, as the attached alerts never do."""
        with self._lock:
            alerts = self._alerts
            if alerts is not None:
                alerts.send(kind, text)
                return
        name = kind.value if isinstance(kind, AlertKind) else repr(kind)
        logger.warning(
            "alert %s before the session's alerts were set up, so it goes to the log only: %s",
            name,
            text,
        )

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


__all__ = ["SharedAlerts"]
