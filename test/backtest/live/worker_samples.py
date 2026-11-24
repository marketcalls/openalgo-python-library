"""An order worker on the fake OpenAlgo, shared by the worker's tests (`test_worker*.py`).

The worker runs with the real SDK clients of the fake, a real mode guard and a real
journal in the test's folder; its order updates come from the fake's direct feed, on the
test's thread, and the test drives `step()` on the fake's manual clock. Requests are
built here as the venue map builds them. Nothing here contacts a server.
"""

from __future__ import annotations

import datetime as dt
import itertools
import sqlite3
from typing import Any

from fake_openalgo import BROKER, FakeOpenAlgo

from openalgo.backtest._live._types import AlertKind, Mode, OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import (
    CancelRequest,
    ModifyRequest,
    OrderTerms,
    PlaceRequest,
    place_priority,
)
from openalgo.backtest._live.guard import ModeGuard
from openalgo.backtest._live.inbox import Inbox, RawOrderUpdate
from openalgo.backtest._live.journal import Journal
from openalgo.backtest._live.worker import JournalTarget, OrderWorker
from openalgo.backtest.errors import JournalError

#: The fake's trading day (its fixture starts on 2026-11-16 at 10:15 IST).
DAY = dt.date(2026, 11, 16)
STRATEGY = "ema_cross"
ACCOUNT = "trader1"
PRICE = 812.0


def disk_failure(*args: Any, **kwargs: Any) -> Any:
    """Stands in for a journal write that meets a failing disk: the JournalError the
    journal raises for SQLite's own error."""
    raise JournalError("cannot record the order update: disk I/O error") from (
        sqlite3.OperationalError("disk I/O error")
    )


def watch_handover(worker: OrderWorker, monkeypatch: Any) -> list[tuple[int | None, int]]:
    """Note the worker's `busy_since_ns` and `raw_pending` whenever it journals an order
    word or looks up the intent of news it is about to post: moments when that news is not
    posted yet. The list fills as the worker steps."""
    seen: list[tuple[int | None, int]] = []
    for name in ("record_update", "intent_of"):
        original = getattr(Journal, name)

        def spy(self: Journal, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            seen.append((worker.busy_since_ns, worker.raw_pending))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Journal, name, spy)
    return seen


def looks_idle(seen: list[tuple[int | None, int]]) -> list[tuple[int | None, int]]:
    """The moments of `watch_handover` at which the worker looked idle to the loop."""
    return [(busy, waiting) for busy, waiting in seen if busy is None and waiting == 0]


class Alerts:
    """Keeps every alert sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))

    def test(self) -> bool:
        return True

    def kinds(self) -> list[AlertKind]:
        return [kind for kind, _ in self.sent]


class Recording:
    """The fake's orders client, noting the worker's `busy_since_ns` at each call."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.worker: OrderWorker | None = None
        self.busy_during: list[tuple[str, int | None]] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._client, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            assert self.worker is not None
            self.busy_during.append((name, self.worker.busy_since_ns))
            return method(*args, **kwargs)

        return call


class WorkerRig:
    """An order worker of `fake`, its journal at `folder`, started up on the test's
    thread unless `start` is False, reading the holdings with every book read when
    `read_holdings` (a session with a CNC strategy); `starts` counts the restarts made
    with `restart`."""

    def __init__(
        self, fake: FakeOpenAlgo, folder: Any, *, start: bool = True, read_holdings: bool = False
    ) -> None:
        self.fake = fake
        self.clock = fake.clock
        self.folder = folder
        self.read_holdings = read_holdings
        self.alerts = Alerts()
        self.inbox = Inbox(self.clock.now)
        self.guard = ModeGuard(fake.client(), now=self.clock.now, alerts=self.alerts)
        self.guard.start()
        self.orders = Recording(fake.client())
        self.feed = fake.direct_feed()
        self.feed.connect()
        self.feed.subscribe_orders(self._on_order)
        self._numbers = itertools.count(1)
        self.starts = 0
        self.worker = self._worker()
        if start:
            self.start_up()

    def _worker(self) -> OrderWorker:
        target = JournalTarget(self.folder / "journal.sqlite", ACCOUNT, BROKER, Mode.LIVE, DAY)
        worker = OrderWorker(
            self.orders,
            guard=self.guard,
            alerts=self.alerts,
            inbox=self.inbox,
            journal=target,
            now=self.clock.now,
            sleep=self.clock.sleep,
            read_holdings=self.read_holdings,
        )
        self.orders.worker = worker
        return worker

    def start_up(self) -> None:
        self.worker.start_up()
        self.starts += 1

    def restart(self) -> None:
        """Close the worker and start a new one on the same journal (a restart)."""
        self.worker.close()
        self.worker = self._worker()
        self.start_up()

    def _on_order(self, message: dict[str, Any]) -> None:
        self.worker.updates.put(RawOrderUpdate(message, self.clock.now()))

    def messages(self) -> list[Any]:
        """Every message the worker posted since the last call, in order."""
        taken = []
        while (envelope := self.inbox.take()) is not None:
            taken.append(envelope.message)
        return taken

    def of(self, kind: type, messages: list[Any]) -> list[Any]:
        return [message for message in messages if isinstance(message, kind)]

    def run(self, request: Any) -> list[Any]:
        """Put `request`, step once and return what was posted."""
        self.worker.put(request)
        self.worker.step()
        return self.messages()

    def journal(self) -> Journal:
        """A second handle on the worker's journal (as the loop's L), on this thread."""
        return Journal.open(
            self.folder / "journal.sqlite",
            account=ACCOUNT,
            broker=BROKER,
            mode=Mode.LIVE,
            now=self.clock.now,
        )

    def intent(
        self,
        purpose: Purpose = Purpose.ENTRY,
        *,
        action: str = "BUY",
        quantity: int = 10,
        price_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
        created_ns: int | None = None,
        symbol: str = "SBIN",
    ) -> OrderIntent:
        """An intent of a new kernel order, as the venue map writes it."""
        kernel_order = next(self._numbers)
        return OrderIntent(
            intent_id=f"20261116-live-{self.starts}-{STRATEGY}-o{kernel_order}",
            strategy=STRATEGY,
            symbol=symbol,
            exchange="NSE",
            action=action,
            quantity=quantity,
            price_type=price_type,
            product="MIS",
            purpose=purpose,
            created_ns=self.clock.now() if created_ns is None else created_ns,
            price=price,
            trigger_price=trigger_price,
        )

    def place(self, purpose: Purpose = Purpose.ENTRY, **fields: Any) -> PlaceRequest:
        intent = self.intent(purpose, **fields)
        kernel_order = int(intent.intent_id.rsplit("-o", 1)[1])
        return PlaceRequest(next(self._numbers), kernel_order, intent, place_priority(intent))

    def stop(self, quantity: int = 10, trigger: float = 800.0) -> PlaceRequest:
        """A sell stop of the leg, SL-M at `trigger`."""
        return self.place(
            Purpose.STOP, action="SELL", quantity=quantity, price_type="SL-M", trigger_price=trigger
        )

    def modify(
        self, place: PlaceRequest, order_id: str, terms: OrderTerms, priority: Any
    ) -> ModifyRequest:
        intent = place.intent
        old = OrderTerms(intent.quantity, intent.price_type, intent.price, intent.trigger_price)
        number = next(self._numbers)
        return ModifyRequest(number, place.kernel_order, intent, order_id, terms, old, priority)

    def cancel(self, place: PlaceRequest, order_id: str, priority: Any) -> CancelRequest:
        number = next(self._numbers)
        return CancelRequest(number, place.kernel_order, place.intent, order_id, priority)
