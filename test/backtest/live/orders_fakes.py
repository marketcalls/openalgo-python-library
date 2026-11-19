"""In-process fakes for the order path's tests: a manual clock, a journal, a mode guard
and an OpenAlgo client whose answers are scripted. Nothing here contacts a server.

The error answers are shaped as the SDK returns them (openalgo/orders.py
`_make_request`), and the order rows as the server's live and sandbox books write them.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openalgo.backtest._live._types import (
    AlertKind,
    BoundOrder,
    Mode,
    OrderIntent,
    OrderStatus,
    OrderUpdate,
    Purpose,
    QueuedIntent,
    mode_of,
)
from openalgo.backtest._live.orders import OrderPath, RateBudget

NS = 1_000_000_000
#: Monday 2026-11-16, 10:15:20 IST.
START_NS = int(dt.datetime(2026, 11, 16, 4, 45, 20, tzinfo=dt.UTC).timestamp()) * NS
START_IST = "2026-11-16 10:15:20"
STRATEGY = "ema_cross"


class Clock:
    """A manual clock: `sleep` moves it on by the time slept and keeps what was asked."""

    def __init__(self, start_ns: int = START_NS) -> None:
        self.now_ns = start_ns
        self.slept: list[float] = []

    def now(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self.now_ns += round(seconds * NS)


class FakeJournal:
    """Keeps what the order path writes. An intent is unbound from when it is recorded
    until it is bound, queued or marked not placed, as in the SQLite journal."""

    def __init__(self) -> None:
        self.intents: dict[str, OrderIntent] = {}
        self.order_of: dict[str, str] = {}
        self.unbound_reasons: dict[str, str] = {}
        self.not_placed: dict[str, str] = {}
        self.queued_as: dict[str, str] = {}
        self.updates: list[OrderUpdate] = []

    def record_intent(self, intent: OrderIntent) -> None:
        self.intents[intent.intent_id] = intent

    def bind(self, intent_id: str, order_id: str) -> None:
        self.order_of[intent_id] = order_id

    def mark_unbound(self, intent_id: str, reason: str) -> None:
        self.unbound_reasons[intent_id] = reason

    def mark_not_placed(self, intent_id: str, reason: str) -> None:
        self.not_placed[intent_id] = reason

    def mark_queued(self, intent_id: str, pending_order_id: str) -> None:
        self.queued_as[intent_id] = pending_order_id

    def queued(self) -> list[QueuedIntent]:
        return [
            QueuedIntent(self.intents[key], pending)
            for key, pending in self.queued_as.items()
            if key not in self.order_of
        ]

    def unbound(self) -> list[OrderIntent]:
        settled = self.order_of.keys() | self.not_placed.keys() | self.queued_as.keys()
        return [intent for key, intent in self.intents.items() if key not in settled]

    def record_update(self, update: OrderUpdate) -> None:
        self.updates.append(update)

    def bound_orders(self) -> list[BoundOrder]:
        bound = []
        for intent_id, order_id in self.order_of.items():
            own = [update for update in self.updates if update.order_id == order_id]
            bound.append(BoundOrder(self.intents[intent_id], order_id, own[-1] if own else None))
        return bound


class FakeGuard:
    """Records what reaches it. `before_entry` raises `entry_error` when one is set; a
    semi-auto answer, or an answer of the other mode than `mode`, locks it, as the mode
    guard does (an error without a mode says nothing)."""

    def __init__(self, *, mode: Mode = Mode.LIVE, entry_error: Exception | None = None) -> None:
        self.mode = mode
        self.started = True
        self.locked = False
        self.entry_error = entry_error
        self.entry_checks = 0
        self.responses: list[dict[str, Any]] = []
        self.updates: list[OrderUpdate] = []
        self.sent: list[str] = []
        self.in_doubt: list[str] = []

    def before_entry(self) -> None:
        self.entry_checks += 1
        if self.entry_error is not None:
            raise self.entry_error

    def check_response(self, response: dict[str, Any]) -> None:
        self.responses.append(response)
        seen = mode_of(response)
        says_nothing = seen is Mode.LIVE and response.get("status") != "success"
        if "pending_order_id" in response or not (says_nothing or seen is self.mode):
            self.locked = True

    def check_update(self, update: OrderUpdate) -> None:
        self.updates.append(update)

    def note_order_sent(self, order_id: str) -> None:
        self.sent.append(order_id)

    def note_order_in_doubt(self, intent_id: str) -> None:
        self.in_doubt.append(intent_id)


class FakeAlerts:
    """Keeps every alert sent."""

    def __init__(self) -> None:
        self.sent: list[tuple[AlertKind, str]] = []

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))


class FakeClient:
    """Answers each call with the next scripted answer of its method, and records every
    call. `before_call(name, kwargs)` runs as a call arrives, before it is answered."""

    def __init__(self, **answers: list[dict[str, Any]]) -> None:
        self.answers = {name: list(queue) for name, queue in answers.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.before_call: Callable[[str, dict[str, Any]], None] | None = None

    def calls_to(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def _answer(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, kwargs))
        if self.before_call is not None:
            self.before_call(name, kwargs)
        queue = self.answers.get(name)
        if not queue:
            raise AssertionError(f"no answer scripted for {name}({kwargs})")
        return queue.pop(0)

    def placeorder(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("placeorder", kwargs)

    def modifyorder(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("modifyorder", kwargs)

    def cancelorder(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("cancelorder", kwargs)

    def orderstatus(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("orderstatus", kwargs)

    def orderbook(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("orderbook", kwargs)

    def tradebook(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("tradebook", kwargs)


@dataclass
class Rig:
    """An order path with its fakes."""

    path: OrderPath
    client: FakeClient
    journal: FakeJournal
    guard: FakeGuard
    clock: Clock
    alerts: FakeAlerts


def make_rig(
    *,
    guard: FakeGuard | None = None,
    journal: FakeJournal | None = None,
    clock: Clock | None = None,
    **answers: list[dict[str, Any]],
) -> Rig:
    """An order path over fakes, with `answers` scripted per client method."""
    client = FakeClient(**answers)
    journal = journal or FakeJournal()
    guard = guard or FakeGuard()
    clock = clock or Clock()
    alerts = FakeAlerts()
    path = OrderPath(
        client,
        journal=journal,
        guard=guard,
        alerts=alerts,
        now=clock.now,
        sleep=clock.sleep,
        rate=RateBudget(clock.now),
    )
    return Rig(path, client, journal, guard, clock, alerts)


def make_intent(
    intent_id: str = "i-1", *, purpose: Purpose = Purpose.ENTRY, **changes: Any
) -> OrderIntent:
    """A MIS market buy of 10 SBIN, created at START_NS unless `changes` say otherwise."""
    fields: dict[str, Any] = {
        "intent_id": intent_id,
        "strategy": STRATEGY,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "price_type": "MARKET",
        "product": "MIS",
        "purpose": purpose,
        "created_ns": START_NS,
    }
    fields.update(changes)
    return OrderIntent(**fields)


def placed(order_id: str = "1") -> dict[str, Any]:
    """The server's answer to an accepted placement in live mode."""
    return {"status": "success", "orderid": order_id}


def http_error(code: int, body: dict[str, Any] | str = "") -> dict[str, Any]:
    """The SDK's answer to a non-200 reply (openalgo/orders.py `_handle_response`): the
    reply's text kept only inside the message. A dict `body` is the server's JSON as its
    `make_response(jsonify(...))` writes it (compact, keys sorted, a closing newline); a
    str is a reply that is not JSON, such as a proxy's error page."""
    text = (
        body
        if isinstance(body, str)
        else json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n"
    )
    return {
        "status": "error",
        "message": f"HTTP {code}: {text}",
        "code": code,
        "error_type": "http_error",
    }


def server_error(message: str, **fields: Any) -> dict[str, Any]:
    """The server's JSON body of an error reply."""
    return {"status": "error", "message": message, **fields}


#: The SDK's answers when no answer came from the server (openalgo/orders.py).
TIMEOUT = {
    "status": "error",
    "message": "Request timed out. The server took too long to respond.",
    "error_type": "timeout_error",
}
#: The SDK's answer when the connection broke off after the request was sent (an
#: httpx.RemoteProtocolError or ReadError, caught as httpx.HTTPError).
DROPPED = {
    "status": "error",
    "message": "HTTP error occurred: Server disconnected without sending a response.",
    "error_type": "http_error",
}
#: OpenAlgo's own JSON answer when the broker's answer to the order was lost (HTTP 502,
#: utils/httpx_client.py UNANSWERED_ORDER_MESSAGE), as the SDK wraps it.
SERVER_502 = http_error(
    502,
    {
        "status": "error",
        "message": "The connection to the broker dropped after the order was sent, so it "
        "may have been placed. Check the order book before placing it again.",
    },
)
#: A proxy's own error page in front of the server (a gateway timeout), as the SDK wraps it.
PROXY_504 = http_error(504, "<html><body>504 Gateway Time-out</body></html>")
CONNECTION_LOST = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}


def stream_message(
    order_id: str, status: str, filled: int, *, quantity: int = 10, mode: str = "live", **extra: Any
) -> dict[str, Any]:
    """A message of the order-updates stream (subscribers/wsproxy_subscriber.py)."""
    message = {
        "type": "order_update",
        "user_id": "trader",
        "mode": mode,
        "broker": "zerodha" if mode == "live" else "",
        "orderid": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": quantity,
        "price": 0.0,
        "trigger_price": 0.0,
        "pricetype": "MARKET",
        "product": "MIS",
        "order_status": status,
        "filled_quantity": filled,
        "pending_quantity": quantity - filled,
        "average_price": 812.35 if filled else 0.0,
        "rejection_reason": "",
    }
    message.update(extra)
    return message


def status_answer(order_id: str, status: str, **row: Any) -> dict[str, Any]:
    """A live `orderstatus` answer: a broker's order-book row, which carries no filled
    quantity, and no mode."""
    data = {
        "action": "BUY",
        "exchange": "NSE",
        "order_status": status,
        "orderid": order_id,
        "price": 0.0,
        "pricetype": "MARKET",
        "product": "MIS",
        "quantity": 10,
        "symbol": "SBIN",
        "timestamp": START_IST,
        "trigger_price": 0.0,
        "average_price": 0.0,
    }
    data.update(row)
    return {"status": "success", "data": data}


def update(
    order_id: str,
    status: OrderStatus,
    filled: int,
    *,
    source: str = "stream",
    quantity: int = 10,
    price: float | None = None,
    ts_recv_ns: int = START_NS,
) -> OrderUpdate:
    """An order update of a live buy of SBIN, its fills at 812.35 unless `price` says."""
    return OrderUpdate(
        order_id=order_id,
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=quantity,
        filled_quantity=filled,
        pending_quantity=0 if status.terminal else quantity - filled,
        average_price=(812.35 if filled else 0.0) if price is None else price,
        status=status,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=ts_recv_ns,
        source=source,
    )


def trades(*fills: tuple[str, int, float]) -> dict[str, Any]:
    """A `tradebook` answer holding one trade per (order ID, quantity, price)."""
    rows = [
        {"orderid": order_id, "quantity": quantity, "average_price": price}
        for order_id, quantity, price in fills
    ]
    return {"status": "success", "data": rows}


def book_row(order_id: str, *, timestamp: str = START_IST, **changes: Any) -> dict[str, Any]:
    """A sandbox order-book row of the default intent's shape (it names its strategy)."""
    row = {
        "orderid": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "pricetype": "MARKET",
        "product": "MIS",
        "order_status": "complete",
        "timestamp": timestamp,
        "strategy": STRATEGY,
    }
    row.update(changes)
    return row


def book(*rows: dict[str, Any]) -> dict[str, Any]:
    """An `orderbook` answer holding `rows`."""
    return {"status": "success", "data": {"orders": list(rows), "statistics": {}}}
