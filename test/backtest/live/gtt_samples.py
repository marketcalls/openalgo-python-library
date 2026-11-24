"""An order worker sending GTT requests on the fake OpenAlgo, shared by the GTT tests
(`test_worker_gtt_*.py`, `test_gtt_follow*.py`; 20-positional-live 3, the lead's rulings
N3a-1 and N3a-2).

The worker is `worker_samples`' (the fake's real SDK clients, a real mode guard, a real
journal in the test's folder), here on any trading day and in the mode the fake is in,
reading holdings with its books (a session with a CNC strategy). Requests are built as
unit N3b will build them from the kernel's GTT commands: by default a long CNC leg's stop
on SBIN, sell 10 when SBIN trades at 790.00 or below, the child's limit 780.00, while SBIN
trades at 800.00. Nothing here contacts a server.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
from collections.abc import Callable
from typing import Any

from fake_openalgo import BROKER, FakeOpenAlgo
from worker_samples import ACCOUNT, DAY, STRATEGY, WorkerRig

from openalgo.backtest._live._gtt_types import (
    Direction,
    GttAction,
    GttAnswer,
    GttOutcome,
    GttRequest,
)
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._venue_requests import PlaceRequest, Priority
from openalgo.backtest._live.inbox import GttNews
from openalgo.backtest._live.worker import JournalTarget, OrderWorker

NS = 10**9
#: SBIN's price when the stop is placed, its trigger and the child's limit, rupees.
PRICE, TRIGGER, LIMIT = 800.0, 790.0, 780.0
UNITS = 10
#: The fake broker's first trigger ID on Monday 2026-11-16, and its next.
FIRST_TRIGGER, SECOND_TRIGGER = "261116000001", "261116000002"
#: The trading day after `DAY`.
NEXT_DAY = dt.date(2026, 11, 17)
#: A row of the user's own GTT the engine cannot read: a broker wrote it with no leg.
UNREADABLE_ROW = {"trigger_id": "990001", "status": "active", "symbol": "TCS", "legs": []}


def broker_with_sbin(fake: FakeOpenAlgo, price: float = PRICE) -> FakeOpenAlgo:
    """`fake` knowing SBIN on NSE, last traded at `price`."""
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", price)
    return fake


def bought(fake: FakeOpenAlgo, units: int = UNITS, product: str = "CNC") -> str:
    """Another client of the account buys `units` SBIN at the market in `product` (the
    leg the GTT protects); its order ID."""
    answer = fake.client().placeorder(
        strategy=STRATEGY,
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product=product,
        quantity=units,
    )
    return answer["orderid"]


def with_rows(rig: Any, monkeypatch: Any, change: Callable[[list[Any]], list[Any]]) -> None:
    """The GTT book's answers to `rig`'s worker, each changed by `change(rows)`."""
    client = rig.orders._client
    original = client.gttorderbook

    def changed(**fields: Any) -> Any:
        answer = original(**fields)
        return {**answer, "data": change(list(answer["data"]))}

    monkeypatch.setattr(client, "gttorderbook", changed)


class GttRig(WorkerRig):
    """`worker_samples.WorkerRig` on trading `day`, in the fake's mode, reading holdings;
    `market_open(now_ns)` as the worker takes it (None: no read on the clock), and
    `orders_per_second` its order rate (the worker's default for None)."""

    def __init__(
        self,
        fake: FakeOpenAlgo,
        folder: Any,
        *,
        day: dt.date = DAY,
        market_open: Any = None,
        orders_per_second: int | None = None,
        start: bool = True,
    ) -> None:
        self.day = day
        self.market_open = market_open
        self.orders_per_second = orders_per_second
        self._requests = itertools.count(1)
        super().__init__(fake, folder, start=start, read_holdings=True)

    def _worker(self) -> OrderWorker:
        mode = self.fake.mode
        target = JournalTarget(self.folder / "journal.sqlite", ACCOUNT, BROKER, mode, self.day)
        rate = (
            {} if self.orders_per_second is None else {"orders_per_second": self.orders_per_second}
        )
        worker = OrderWorker(
            self.orders,
            guard=self.guard,
            alerts=self.alerts,
            inbox=self.inbox,
            journal=target,
            now=self.clock.now,
            sleep=self.clock.sleep,
            read_holdings=True,
            market_open=self.market_open,
            **rate,
        )
        self.orders.worker = worker
        return worker

    def gtt(
        self,
        action: GttAction = GttAction.PLACE,
        *,
        seq: int | None = None,
        side: str = "SELL",
        trigger: float = TRIGGER,
        limit: float = LIMIT,
        units: int = UNITS,
        direction: Direction = Direction.BELOW,
        product: str = "CNC",
        purpose: Purpose = Purpose.STOP,
        gtt_id: int = 1,
        priority: Priority = Priority.PROTECT,
    ) -> GttRequest:
        """A GTT request of `STRATEGY` on SBIN NSE (module docstring's stop by default)."""
        return GttRequest(
            request_id=next(self._requests),
            action=action,
            purpose=purpose,
            strategy=STRATEGY,
            gtt_id=gtt_id,
            symbol="SBIN",
            exchange="NSE",
            product=product,
            side=side,
            trigger_price=trigger,
            limit_price=limit,
            quantity=units,
            direction=direction,
            priority=priority,
            seq=seq,
        )

    def own_exit(
        self, units: int, product: str = "CNC", *, price_type: str = "MARKET", price: float = 0.0
    ) -> PlaceRequest:
        """The session's own exit of the leg, as the venue map sends a strategy's exit:
        sell `units` SBIN in `product`, at the market or at a limit of `price` rupees."""
        fields = {"price_type": price_type, "price": price}
        request = self.place(Purpose.EXIT, action="SELL", quantity=units, **fields)
        intent = dataclasses.replace(request.intent, product=product)
        return dataclasses.replace(request, intent=intent)

    def send(self, request: Any) -> list[Any]:
        """Hand the worker `request`, step once, and return the GTT news it posted."""
        return gtt_news(self.run(request))

    def step_for(self, seconds: float, every_s: float = 1.0) -> list[Any]:
        """Step the worker every `every_s` seconds for `seconds`, moving the fake's clock;
        the GTT news posted meanwhile."""
        news: list[Any] = []
        for _ in range(round(seconds / every_s)):
            self.clock.advance(every_s)
            self.worker.step()
            news += gtt_news(self.messages())
        return news


def gtt_news(messages: list[Any]) -> list[GttAnswer | GttOutcome]:
    """The GTT answers and outcomes among `messages`, in order."""
    return [message.news for message in messages if isinstance(message, GttNews)]


def answers(news: list[Any]) -> list[GttAnswer]:
    return [item for item in news if isinstance(item, GttAnswer)]


def outcomes(news: list[Any]) -> list[GttOutcome]:
    return [item for item in news if isinstance(item, GttOutcome)]


def calls(fake: FakeOpenAlgo, *endpoints: str) -> list[str]:
    """The endpoints the fake answered among `endpoints` (every one when none), in order."""
    return [e.what for e in fake.log.calls() if not endpoints or e.what in endpoints]


__all__ = [
    "FIRST_TRIGGER",
    "LIMIT",
    "NEXT_DAY",
    "NS",
    "PRICE",
    "SECOND_TRIGGER",
    "TRIGGER",
    "UNITS",
    "UNREADABLE_ROW",
    "GttRig",
    "answers",
    "bought",
    "broker_with_sbin",
    "calls",
    "gtt_news",
    "outcomes",
    "with_rows",
]
