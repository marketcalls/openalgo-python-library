"""The order worker's GTT requests (20-positional-live 3; the lead's rulings N3b-2, N3b-4
and N3b-5), split from `_worker_send` so each module keeps one job.

- A `GttRequest` goes to the GTT sender (`_gtt_send`).
- `ReadGtts` asks the GTT follower for a read of the GTT book; the close's (`at_close`)
  reads every GTT leg's units with it and is posted as `GttBookSeen`.
- `BindGttChild` binds a fired GTT's child in the journal and has the order path follow
  it (`_order_children`), its word that came before the binding handed over now.
- `reference` gives the follower each GTT leg's units of a reconciliation read, the
  start's among them (made after the order path's restore), as `GttFollower.reference`
  asks: it takes a settled read and ignores one that is not.

Times are UTC nanoseconds, quantities signed whole units. Used by the worker's thread only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..errors import LiveError
from ._broker_units import broker_units
from ._gtt_reads import LegKey
from ._gtt_send import GttSender
from ._gtt_types import BindGttChild, GttRequest, ReadGtts
from ._order_children import child_intent
from ._worker_news import NewsHandover
from ._worker_stops import StopBook
from .inbox import Message
from .journal import Journal
from .orders import OrderPath

#: The requests this module takes.
GTT_REQUESTS = (GttRequest, ReadGtts, BindGttChild)


class WorkerGtts:
    """The GTT requests of the order worker (module docstring): `gtts` the GTT sender with
    its follower, `path` the order path, `journal` handle W, `news` and `stops` the
    worker's, `post` puts a message on the loop's inbox, `now` gives UTC nanoseconds, and
    `legs` are the (symbol, exchange, product) of the session's CNC and NRML legs."""

    def __init__(
        self,
        gtts: GttSender,
        *,
        path: OrderPath,
        journal: Journal,
        news: NewsHandover,
        stops: StopBook,
        post: Callable[[Message], None],
        now: Callable[[], int],
        legs: Sequence[LegKey],
    ) -> None:
        self._gtts, self._path, self._journal = gtts, path, journal
        self._news, self._stops, self._post, self._now = news, stops, post, now
        self.legs = tuple(legs)

    def execute(self, request: GttRequest | ReadGtts | BindGttChild) -> None:
        """Execute one of the GTT requests (module docstring). Raises JournalError when the
        journal fails, and LiveError for a child whose GTT the journal does not hold."""
        if isinstance(request, GttRequest):
            self._gtts.execute(request)
        elif isinstance(request, ReadGtts) and request.at_close:
            self._post(self._gtts.follower.close_read(set(self.legs)))
        elif isinstance(request, ReadGtts):
            self._gtts.follower.read_now()
        else:
            self._bind_child(request)

    def reference(self, positionbook: Any, holdings: Any, read_ns: int) -> None:
        """Give the GTT follower each GTT leg's units of a reconciliation read begun at
        `read_ns` (UTC ns) (module docstring). A leg whose units the read cannot give is
        left out."""
        for symbol, exchange, product in self.legs:
            units = broker_units(positionbook, holdings, symbol, exchange, product)
            if units is not None:
                self._gtts.follower.reference((symbol, exchange, product), units, read_ns)

    def _bind_child(self, request: BindGttChild) -> None:
        self._gtts.follower.bind_child(request.seq, request.order_id, request.kernel_order)
        gtt = self._journal.gtt(request.seq)
        if gtt is None:
            raise LiveError(f"GTT {request.seq} is not in the journal: its child is not followed")
        intent = child_intent(gtt, request.order_id, self._now())
        news = self._path.follow_child(request.order_id, intent)
        self._news.taken(1)
        self._news.hand_over(news, self._stops)


__all__ = ["GTT_REQUESTS", "WorkerGtts"]
