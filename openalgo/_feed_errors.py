"""Error reporting of the feed: the error hooks, or the log when there is none
(private to openalgo.feed)."""

import logging
import threading
from collections.abc import Callable
from typing import Any

from ._feed_hooks import HookRegistry

# The documented logger name of the feed, whichever module writes to it.
logger = logging.getLogger("openalgo.feed")

AUTHENTICATION_CODE = "AUTHENTICATION_ERROR"
# Callbacks that run at tick rate. Only their failures are rate limited in the log.
TICK_CALLBACKS = frozenset({"ltp", "quote", "depth"})


class FeedServerError(Exception):
    """The server rejected a request. code is the server's error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def server_rejection(frame: Any) -> tuple[str, FeedServerError, dict[str, Any]] | None:
    """
    Tell whether a decoded server frame is a rejection.

    Returns None when it is not. Otherwise (kind, error, details): kind is
    "authentication" when the server refused the API key and "server" for any
    other rejection, and details holds "rejected", the entries the server did
    not accept, for a subscribe or unsubscribe reply.
    """
    if not isinstance(frame, dict):
        return None
    frame_type = frame.get("type")
    text = str(frame.get("message") or "Unknown error")
    if frame_type == "auth":
        if frame.get("status") == "success":
            return None
        code = frame.get("code") or AUTHENTICATION_CODE
        return "authentication", FeedServerError(code, text), {}

    rejected = []
    if frame_type == "subscribe":
        entries = frame.get("subscriptions") or []
        rejected = [entry for entry in entries if entry.get("status") == "error"]
    elif frame_type == "unsubscribe":
        rejected = frame.get("failed") or []
    if rejected:
        names = ", ".join(f"{entry.get('exchange')}:{entry.get('symbol')}" for entry in rejected)
        error = FeedServerError(f"{frame_type.upper()}_REJECTED", f"not accepted: {names}")
        return "server", error, {"rejected": rejected}

    if frame.get("status") != "error":
        return None
    code = frame.get("code") or "SERVER_ERROR"
    kind = "authentication" if code == AUTHENTICATION_CODE else "server"
    return kind, FeedServerError(code, text), {}


def _describe(context: dict[str, Any]) -> str:
    """Say in words where an error came from, for the log."""
    kind = context.get("kind")
    if kind == "callback":
        return f"unhandled exception in the {context.get('callback')} callback"
    if kind == "hook":
        return f"unhandled exception in an {context.get('callback')} hook"
    if kind == "authentication":
        return "the server rejected the API key"
    if kind == "server":
        return "the server reported an error"
    return "a message from the server could not be processed"


class ErrorReporter:
    """
    Takes every error the feed catches to the error hooks, or to the
    "openalgo.feed" logger when no error hook is registered, so that none is
    silent.

    In the log, a failure of a tick callback (LTP, quote, depth) is written
    once and then only counted for window() seconds, per callback and
    exception type; the count is written when the window ends. Everything
    else (order updates, hooks, server rejections) is always written.
    """

    def __init__(
        self,
        hooks: HookRegistry,
        window: Callable[[], float],
        timer: Callable[..., Any] = threading.Timer,
    ) -> None:
        """timer(seconds, function, args=...) schedules the end of a window."""
        self._hooks = hooks
        self._window = window
        self._timer = timer
        self._lock = threading.Lock()
        # Open windows: (callback, exception type) -> failures counted, not written.
        self._skipped: dict[tuple[Any, type], int] = {}

    def report(self, error: BaseException, **context: Any) -> None:
        """Hand error and its context to the error hooks, or write it to the log."""
        if self._hooks.has("error"):
            self._hooks.post("error", error, context)
            self._hooks.drain()
        else:
            self.log(error, context)

    def log(self, error: BaseException, context: dict[str, Any]) -> None:
        """Write error to the log, subject to the rate limit for tick callbacks."""
        callback = context.get("callback")
        rate_limited = context.get("kind") == "callback" and callback in TICK_CALLBACKS
        if rate_limited and not self._open_window((callback, type(error))):
            return
        traceback = error.__traceback__
        logger.error(
            "OpenAlgo feed: %s: %s: %s",
            _describe(context),
            type(error).__name__,
            error,
            exc_info=(type(error), error, traceback) if traceback else None,
        )

    def _open_window(self, key: tuple[Any, type]) -> bool:
        """
        True when this failure is to be written: it then opens a window in
        which the same failure is only counted. False while one is open.
        """
        with self._lock:
            if key in self._skipped:
                self._skipped[key] += 1
                return False
            self._skipped[key] = 0
        seconds = self._window()
        timer = self._timer(seconds, self._close_window, args=(key, seconds))
        timer.daemon = True
        timer.start()
        return True

    def _close_window(self, key: tuple[Any, type], seconds: float) -> None:
        """The window is over: write how many failures it left out."""
        with self._lock:
            skipped = self._skipped.pop(key, 0)
        if skipped:
            callback, error_type = key
            logger.error(
                "OpenAlgo feed: %d more %s in the %s callback in the last %s s were not logged",
                skipped,
                error_type.__name__,
                callback,
                seconds,
            )
