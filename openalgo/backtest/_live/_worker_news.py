"""How the order worker hands an order's news to the loop (16-session-design 2.4 step 1,
4.1, 4.2), split from `worker` so each module keeps one job.

- The raw order updates the feed's order callback queued are drained, each normalized (one
  that cannot be read is logged and counted, never raised) and taken by the order path,
  and its news is posted as `inbox.OrderNews`. A fill of an order the order path does not
  follow is posted as `inbox.OtherFill`, so the loop can ask for a comparison soon (5.1).
- Posting news: a terminal word settles its order (`_worker_settle`) and lets its requests
  be taken again, and a STOP intent's terminal word clears its leg's journal stop first
  (`_worker_stops`), so the news says whether it did.
- In hand: every order word the worker takes in is counted from before it leaves the raw
  FIFO, or before the busy mark of the call that returned it clears (a poll, a read, a
  resolution, a held word released), until it is posted or turns out to be no news.
  `pending`, the worker's `raw_pending`, is the FIFO's length plus that count. The loop
  reads `busy_since_ns`, then `raw_pending`, then its own inbox: each signal is raised
  before the one ahead of it clears, so read in that order the three never all show the
  worker idle while a word it took is unposted, and a stop's confirmation cannot slip past
  the deferral of 4.2 while it is journaled.

Used by the worker's thread, but for `pending`, which any thread reads.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import LiveError
from ._order_updates import normalize_update
from ._types import LOGGER_NAME, OrderUpdate, Purpose
from ._worker_queue import WorkerQueue, order_key
from ._worker_settle import Settlements
from .inbox import Message, OrderNews, OtherFill, RawOrderUpdate, RawUpdates
from .orders import OrderPath

if TYPE_CHECKING:
    # For the type checker only: the stop book hands its reads' news over through here.
    from ._worker_stops import StopBook

logger = logging.getLogger(LOGGER_NAME)


class NewsHandover:
    """The worker's handover of order news to the loop (module docstring). `updates` is
    the raw FIFO the feed's order callback fills, `settle` and `queue` are the worker's,
    and `post` puts a message on the loop's inbox."""

    def __init__(
        self,
        updates: RawUpdates,
        *,
        settle: Settlements,
        queue: WorkerQueue,
        post: Callable[[Message], None],
    ) -> None:
        self._updates = updates
        self._settle = settle
        self._queue = queue
        self._post = post
        # Written by the worker's thread only. An int is replaced whole, so a reader on
        # another thread sees the count before or after a change, never a torn one.
        self._in_hand = 0
        #: Raw order updates that could not be read, logged and dropped.
        self.unreadable = 0

    @property
    def pending(self) -> int:
        """The order words the loop has not been given: those waiting in the FIFO, then
        those the worker holds. The FIFO is read first, since a word leaves it only once
        it is counted in hand."""
        return len(self._updates) + self._in_hand

    def drain(self, path: OrderPath, stops: StopBook) -> None:
        """Hand over the raw updates waiting: each normalized, taken by `path`, and its
        news posted. Raises JournalError when the journal fails, and LiveError for news
        of an order with no intent in the journal."""
        # One is counted before the FIFO is emptied, so a word is never in neither place.
        self._in_hand += 1
        batch = self._updates.drain()
        self._in_hand += len(batch) - 1
        for raw in batch:
            self.hand_over(self._news_of(raw, path), stops)

    def taken(self, count: int) -> None:
        """`count` order words a call returned are in hand: called before the worker's
        busy mark for the call clears."""
        self._in_hand += count

    def handed(self, count: int = 1) -> None:
        """`count` order words in hand were posted, or were no news."""
        self._in_hand -= count

    def hand_over(self, news: OrderUpdate | None, stops: StopBook) -> None:
        """Post a word in hand as `OrderNews` with its intent, or let it go when it is no
        news (None); either way it is no longer in hand. Its terminal news settles the
        order and, for a STOP intent, clears its leg's journal stop first. Raises
        LiveError for an order with no intent in the journal, and JournalError when the
        stop cannot be cleared."""
        if news is not None:
            intent = stops.intent_of(news.order_id)
            cleared = False
            if news.status.terminal:
                waiting = self._settle.ended(news.order_id)
                if waiting is not None:
                    self._queue.release(order_key(waiting.request))
                if intent.purpose is Purpose.STOP:
                    cleared = stops.ended(intent, news.order_id)
            self._post(OrderNews(news, intent.intent_id, cleared))
        self.handed()

    def _news_of(self, raw: RawOrderUpdate, path: OrderPath) -> OrderUpdate | None:
        try:
            update = normalize_update(raw.message, ts_recv_ns=raw.arrived_ns, source="stream")
        except LiveError as error:
            self.unreadable += 1
            logger.warning("an order update cannot be read, dropped: %s", error)
            return None
        news = path.on_update(update)
        # A fill of an order the order path does not follow (another client's, or an
        # unbound intent's, kept aside): the loop asks for a comparison soon (5.1).
        if news is None and update.filled_quantity > 0 and path.last_known(update.order_id) is None:
            self._post(OtherFill(update.symbol, update.exchange, update.ts_recv_ns))
        return news


__all__ = ["NewsHandover"]
