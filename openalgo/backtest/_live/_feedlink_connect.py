"""The feed's first `connect()`, run off the loop's thread, for `feedlink`
(16-session-design 2.9 and 3.3).

The SDK's `FeedAPI.connect()` waits in real time for the socket to open and then for the
server to accept the API key (5 s each by default: `Timing` in `openalgo/_feed_state.py`),
and nothing but `disconnect()` ends that wait early. On the loop's thread it would stop
the loop for up to 10 s at the start: no tick, no order news, no protective work at a
restart and no Ctrl+C. So the link runs it on a short-lived thread of its own and waits
through the session's sleep, which keeps the loop turning and sees the stop.

A stop while it runs disconnects the feed, which ends the SDK's wait at its next look.
The SDK's `connect()` re-arms a feed that `disconnect()` stopped, so a `connect()` that
only began after that disconnect would connect anew: when the link has stopped by the
time `connect()` returns, this thread disconnects the feed itself.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from ._types import LOGGER_NAME

#: The name of the thread that runs the feed's first `connect()`.
CONNECT_THREAD_NAME = "openalgo-live-feed-connect"

logger = logging.getLogger(LOGGER_NAME)


class FirstConnect:
    """One call of the feed's `connect()`.

    `source` is the link's feed; `stopped()` tells whether the link has stopped it, and
    is asked once `connect()` has returned. Call `start()` to run the call on a thread of
    its own (a daemon: it ends within the feed's own wait times), or `run()` to run it on
    the calling thread, for a feed whose `connect()` never waits; one of them, once.
    """

    def __init__(self, source: Any, *, stopped: Callable[[], bool]) -> None:
        self._source = source
        self._stopped = stopped
        self._returned = threading.Event()
        self._connected = False
        self._error: Exception | None = None

    @property
    def returned(self) -> bool:
        """Whether `connect()` has returned or raised."""
        return self._returned.is_set()

    @property
    def connected(self) -> bool:
        """Whether `connect()` returned True: the feed connected and authenticated."""
        return self._returned.is_set() and self._connected

    @property
    def error(self) -> Exception | None:
        """What `connect()` raised, for the link's thread to raise; else None."""
        return self._error if self._returned.is_set() else None

    def start(self) -> None:
        """Run `connect()` on a thread of its own."""
        threading.Thread(target=self.run, name=CONNECT_THREAD_NAME, daemon=True).start()

    def run(self) -> None:
        """Run `connect()` here; then, when the link stopped meanwhile, disconnect."""
        try:
            self._connected = bool(self._source.connect())
        # Handed to the link's thread through `error`: this thread has no caller.
        except Exception as error:
            self._error = error
        finally:
            self._returned.set()
        if self._stopped():
            self._disconnect()

    def _disconnect(self) -> None:
        try:
            self._source.disconnect()
        # Nobody waits on this thread: what the disconnect raised is logged.
        except Exception:
            logger.exception("the feed's disconnect after a stopped start failed")


__all__ = ["CONNECT_THREAD_NAME", "FirstConnect"]
