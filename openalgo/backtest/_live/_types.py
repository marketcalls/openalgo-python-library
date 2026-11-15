"""The values that pass between the modules of the live layer, and the interfaces the
order path uses (one source of truth: no module redefines them).

Times are UTC nanoseconds, prices rupees, quantities units (never lots). Every value is
frozen: a module that needs a changed value builds a new one.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

#: The present, UTC nanoseconds. Injected everywhere, so tests run on a manual clock.
Now = Callable[[], int]
#: Wait this many seconds. Injected everywhere, so tests wait for nothing.
Sleep = Callable[[float], None]

#: The logger of the live layer. Library code never prints.
LOGGER_NAME = "openalgo.backtest.live"

#: The `mode` value OpenAlgo puts on a response or an order update in analyzer
#: (sandbox) mode; a live response carries no `mode`, or "live" on an order update.
SANDBOX_MODE_FIELD = "analyze"


class Mode(enum.Enum):
    """The mode OpenAlgo is in, set by its analyzer switch (D21). Never chosen in code."""

    SANDBOX = "sandbox"
    LIVE = "live"


def mode_of(message: Mapping[str, Any]) -> Mode:
    """The mode a REST response or an order update was handled in: SANDBOX when its
    `mode` field is "analyze", LIVE otherwise."""
    return Mode.SANDBOX if message.get("mode") == SANDBOX_MODE_FIELD else Mode.LIVE


class OrderStatus(enum.Enum):
    """An order's status as OpenAlgo reports it, in the lower-case words of its
    order-updates stream."""

    OPEN = "open"
    TRIGGER_PENDING = "trigger_pending"
    COMPLETE = "complete"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """True once the order can change no more: complete, rejected or cancelled."""
        return self in (OrderStatus.COMPLETE, OrderStatus.REJECTED, OrderStatus.CANCELLED)


class Purpose(enum.Enum):
    """Why the engine sends an order."""

    ENTRY = "entry"
    ADD = "add"
    STOP = "stop"
    EXIT = "exit"
    RISK_EXIT = "risk_exit"
    SQUARE_OFF = "square_off"


class AlertKind(enum.Enum):
    """What an alert is about (08-live-trading section 8)."""

    SESSION = "session"
    ENTRY = "entry"
    EXIT = "exit"
    STOP_MOVED = "stop_moved"
    REJECT = "reject"
    MODE_BREACH = "mode_breach"
    RECONCILE_MISMATCH = "reconcile_mismatch"
    FEED_DEAD = "feed_dead"
    DAILY_SUMMARY = "daily_summary"
    ERROR = "error"


@dataclass(frozen=True)
class Tick:
    """One price update from the OpenAlgo WebSocket.

    `ts_event_ns` is the exchange's time of the last trade when the feed gives it, else
    None; `ts_recv_ns` is when the engine received it. `volume` is the cumulative
    volume of the day when given; `bid` and `ask` only in depth mode.
    """

    symbol: str
    exchange: str
    ltp: float
    ts_recv_ns: int
    ts_event_ns: int | None = None
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class LiveBar:
    """A bar of the live session, stamped at its close (D27): `source` is "ticks" for a
    bar built from WebSocket ticks, "candle" for a broker candle. `ticks` counts the
    ticks it was built from (0 for a candle)."""

    symbol: str
    exchange: str
    ts_open_ns: int
    ts_close_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticks: int
    source: str


@dataclass(frozen=True)
class OrderIntent:
    """An order the engine means to send, written to the journal before the HTTP call.

    `action` is "BUY" or "SELL", `price_type` "MARKET", "LIMIT", "SL" or "SL-M",
    `product` "MIS" in Tier 1. `price` and `trigger_price` are 0 where they do not
    apply. `intent_id` is unique in the account's journal.
    """

    intent_id: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    price_type: str
    product: str
    purpose: Purpose
    created_ns: int
    price: float = 0.0
    trigger_price: float = 0.0


@dataclass(frozen=True)
class OrderUpdate:
    """An order's state, from the order-updates stream (`source` "stream") or from
    polling `orderstatus` (`source` "poll"), with the mode it was handled in."""

    order_id: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    filled_quantity: int
    pending_quantity: int
    average_price: float
    status: OrderStatus
    rejection_reason: str
    mode: Mode
    ts_recv_ns: int
    source: str


class JournalLike(Protocol):
    """What the order path needs of the journal (`journal.Journal`)."""

    def record_intent(self, intent: OrderIntent) -> None:
        """Write the intent before the HTTP call."""

    def bind(self, intent_id: str, order_id: str) -> None:
        """Bind an intent to the broker's order ID."""

    def mark_unbound(self, intent_id: str, reason: str) -> None:
        """Note an intent whose outcome is unknown (a 502, a timeout, a disconnect)."""

    def mark_not_placed(self, intent_id: str, reason: str) -> None:
        """Close an intent that never became a broker order: a definite refusal, or an
        unbound intent the order book does not hold."""

    def unbound(self) -> list[OrderIntent]:
        """The intents still unbound, oldest first."""

    def record_update(self, update: OrderUpdate) -> None:
        """Write an order update."""


class GuardLike(Protocol):
    """What the order path needs of the mode guard (`guard.ModeGuard`)."""

    @property
    def locked(self) -> bool:
        """True once the mode changed or semi-auto appeared: nothing more is sent."""

    def before_entry(self) -> None:
        """Check the mode just before an entry; locks the guard on a change."""

    def check_response(self, response: Mapping[str, Any]) -> None:
        """Check the mode of a REST response; locks the guard on a change."""

    def check_update(self, update: OrderUpdate) -> None:
        """Check the mode of an order update; locks the guard on a change."""

    def note_order_sent(self, order_id: str) -> None:
        """Record an order the session sent, so a breach can list the orders sent after
        the change, and so the guard judges updates of the session's own orders only."""

    def note_order_in_doubt(self, intent_id: str) -> None:
        """Record an intent whose order may have reached the server with no order ID to
        show for it (a timeout, an HTTP 502, a lost connection), so a breach lists it."""


class AlertsLike(Protocol):
    """What the modules need of the alerts (`alerts.Alerts`)."""

    def send(self, kind: AlertKind, text: str) -> None:
        """Send an alert; never raises into the session loop."""


__all__ = [
    "LOGGER_NAME",
    "SANDBOX_MODE_FIELD",
    "AlertKind",
    "AlertsLike",
    "GuardLike",
    "JournalLike",
    "LiveBar",
    "Mode",
    "Now",
    "OrderIntent",
    "OrderStatus",
    "OrderUpdate",
    "Purpose",
    "Sleep",
    "Tick",
    "mode_of",
]
