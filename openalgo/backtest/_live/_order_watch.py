"""The order path's view of whether OpenAlgo answers (`orders.OrderPath` takes it in; split
from `orders` so each module keeps its size): since when OpenAlgo has been silent after a
client timeout or a dropped connection, when it last answered, the probe that asks it, and
the answers of calls made beside the order path on its client (the GTT calls,
20-positional-live 3.1), which count as the order path's own (`_order_answers.ServerWatch`).

Times are UTC nanoseconds. Used by the thread that owns the order path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._answers import server_answer
from ._order_answers import ServerWatch
from ._types import GuardLike, Now


class WatchedPath:
    """The server watch of an order path (module docstring): the path gives its client,
    guard, clock and `ServerWatch`."""

    _client: Any
    _guard: GuardLike
    _now: Now
    _watch: ServerWatch

    @property
    def silent_since_ns(self) -> int | None:
        """Since when (UTC nanoseconds) OpenAlgo has not answered a call after a client
        timeout or a dropped connection (nothing is sent then); None while it answers."""
        return self._watch.silent_since_ns

    @property
    def asked_ns(self) -> int | None:
        """The time (UTC nanoseconds) of the order path's last answer from OpenAlgo, as
        `ServerWatch` notes it; None before the first."""
        return self._watch.asked_ns

    def note_answer(self, answer: Mapping[str, Any]) -> None:
        """Take in the answer (`server_answer`'s) to a call made beside the order path on
        its client, a GTT call: a client timeout or a dropped connection holds every send
        until OpenAlgo answers again, as after the order path's own calls."""
        self._watch.note(answer, self._now())

    def probe(self) -> bool:
        """Ask OpenAlgo whether it answers again with one `orderbook` read, checked by the
        guard: OpenAlgo's own JSON (a success or its own error) lifts a silent server's
        hold on sending; a timeout, a dropped connection or a gateway's page does not.
        Nothing is read while the guard is locked. True when OpenAlgo is not silent after."""
        if not self._guard.locked:
            answer = server_answer(self._client.orderbook())
            self._watch.note(answer, self._now())
            self._guard.check_response(answer)
        return not self._watch.silent


__all__ = ["WatchedPath"]
