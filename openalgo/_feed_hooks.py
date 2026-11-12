"""Hook listeners of the feed and their ordered delivery (private to openalgo.feed)."""

import threading
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any


def fresh_copy(value: Any) -> Any:
    """Copy dicts, lists and tuples all the way down. Any other object is shared."""
    if isinstance(value, dict):
        return {key: fresh_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fresh_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(fresh_copy(item) for item in value)
    return value


class HookRegistry:
    """
    The listeners of each event, and the one path on which events reach them.

    post() queues an event and drain() delivers what is queued. Only one
    thread delivers at a time: a drain() that finds another thread delivering
    returns at once, and that thread delivers the new event after the one it
    is busy with. Events therefore reach the listeners one at a time and in
    the order they were posted, also when a listener causes the next event
    itself or another thread posts while a listener runs.

    No lock is held while a listener runs.
    """

    def __init__(
        self, events: Iterable[str], on_failure: Callable[[str, BaseException, tuple], None]
    ) -> None:
        """on_failure(event, exception, arguments) is told when a listener raises."""
        self._lock = threading.Lock()
        self._listeners: dict[str, tuple[Callable, ...]] = {event: () for event in events}
        # (event, arguments, still_current, unheard): see post().
        self._queue: deque[tuple[str, tuple, Callable[[], bool] | None, Callable | None]] = deque()
        self._delivering = False
        self._on_failure = on_failure

    def add(self, event: str, callback: Callable) -> Callable:
        """Register callback for event. Returns it, so add() works as a decorator."""
        if not callable(callback):
            raise TypeError(f"on_{event} expects a callable, got {type(callback).__name__}")
        with self._lock:
            self._listeners[event] += (callback,)
        return callback

    def remove(self, callback: Callable) -> bool:
        """Unregister callback from every event. Returns True if it was registered."""
        removed = False
        with self._lock:
            for event, listeners in self._listeners.items():
                kept = tuple(listener for listener in listeners if listener != callback)
                removed = removed or len(kept) != len(listeners)
                self._listeners[event] = kept
        return removed

    def post(
        self,
        event: str,
        *args: Any,
        still_current: Callable[[], bool] | None = None,
        unheard: Callable[..., None] | None = None,
    ) -> None:
        """
        Queue event with the arguments its listeners receive.

        still_current, when given, is asked before each listener: once it
        answers False the listeners that have not been called yet are skipped,
        because what the event announces is already over.

        unheard, when given, receives the arguments instead when no listener
        is registered for event: at once when there is none now, or on the
        delivering thread when the last one was removed before the event's turn.
        """
        with self._lock:
            heard = bool(self._listeners[event])
            if heard:
                self._queue.append((event, args, still_current, unheard))
        if not heard and unheard is not None:
            unheard(*args)

    def drain(self) -> None:
        """Deliver the queued events, unless another thread is already doing so."""
        with self._lock:
            if self._delivering:
                return
            self._delivering = True
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        self._delivering = False
                        return
                    event, args, still_current, unheard = self._queue.popleft()
                    listeners = self._listeners[event]
                if listeners or unheard is None:
                    self._deliver(event, args, still_current, listeners)
                else:
                    unheard(*args)
        except BaseException:
            # A listener raised something like KeyboardInterrupt: let it
            # through, but do not leave the queue blocked for good.
            with self._lock:
                self._delivering = False
            raise

    def _deliver(
        self,
        event: str,
        args: tuple,
        still_current: Callable[[], bool] | None,
        listeners: tuple[Callable, ...],
    ) -> None:
        for listener in listeners:
            if still_current is not None and not still_current():
                return
            try:
                # Every listener gets its own copy: one cannot alter what the
                # next one receives.
                listener(*fresh_copy(args))
            except Exception as exc:
                self._on_failure(event, exc, args)
