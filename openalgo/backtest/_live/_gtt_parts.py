"""The order worker's GTT parts, built at its start-up on journal handle W and the order
path (20-positional-live 3; `worker` calls `gtt_parts`): the reads (`_gtt_reads`), the calls
(`_gtt_calls`, counted in the order rate's window), the follower (`_gtt_follow`), which
takes back the journal's GTTs and hears of every order the journal does not know, and the
sender (`_gtt_send`), which holds the follower.

Times are UTC nanoseconds. Used by the worker's thread only.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from ._gtt_calls import GttCalls
from ._gtt_follow import GttFollower
from ._gtt_reads import GttReader
from ._gtt_send import GttSender
from ._order_rate import RateBudget
from ._types import AlertsLike, GuardLike, Now, Sleep
from ._worker_news import NewsHandover
from ._worker_queue import WorkerQueue
from .inbox import Message
from .journal import Journal
from .orders import OrderPath


def gtt_parts(
    client: object,
    *,
    journal: Journal,
    path: OrderPath,
    guard: GuardLike,
    queue: WorkerQueue,
    budget: RateBudget,
    news: NewsHandover,
    post: Callable[[Message], None],
    alerts: AlertsLike,
    now: Now,
    sleep: Sleep,
    calling: Callable[[], AbstractContextManager[None]],
    market_open: Callable[[int], bool] | None,
) -> GttSender:
    """The GTT sender of an order worker and the follower it holds (module docstring),
    through `client` (the orders client). Raises JournalError when the journal's GTTs
    cannot be read."""
    reader = GttReader(
        client,
        guard=guard,
        note=path.note_answer,
        own=path.own_fills,
        now=now,
        calling=calling,
    )
    follower = GttFollower(
        reader,
        journal=journal,
        post=post,
        alerts=alerts,
        now=now,
        market_open=market_open,
        busy=lambda: path.silent_since_ns is not None,
    )
    follower.restore()
    news.unknown_order = follower.order_seen
    calls = GttCalls(client, path=path, guard=guard, budget=budget, sleep=sleep, now=now)
    return GttSender(
        calls,
        reader,
        follower,
        journal=journal,
        queue=queue,
        path=path,
        guard=guard,
        post=post,
        now=now,
    )


__all__ = ["gtt_parts"]
