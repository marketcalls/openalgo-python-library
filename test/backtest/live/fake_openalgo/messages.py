"""The messages of OpenAlgo's WebSocket proxy, written by hand from the server's code.

- Market data: `{"type": "market_data", "symbol", "exchange", "mode", "data", "broker"}`,
  `mode` the subscriber's mode (1 LTP, 2 Quote, 3 Depth) and `data` what the broker's
  adapter publishes for it (websocket_proxy/server.py, its market-data listener;
  broker/zerodha/streaming/zerodha_adapter.py, `_transform_regular_tick`).
- Order updates: the payload of subscribers/wsproxy_subscriber.py (`on_order_update`),
  relayed as it is (websocket_proxy/server.py, `_handle_order_update`). A live broker's
  status words are those of the order adapters, "trigger_pending" with an underscore
  (websocket_proxy/order_adapter.py, `normalize`); the sandbox writes "trigger pending".
  The sandbox announces a placement, a trigger and a cancellation without fill figures
  (filled, pending and average all 0, the event's defaults: sandbox/order_manager.py
  and sandbox/execution_engine.py, `_publish_order_update_event`) and a fill as complete
  for the whole order (`_publish_fill_event`).
- Replies: authentication and the order-stream subscription (websocket_proxy/server.py).

Prices are rupees, quantities units, times epoch milliseconds on the wire.
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from .book import FILLED, Change
from .clock import epoch_ms
from .market import MarketState
from .orders import TRIGGER_PENDING

#: The `broker` the sandbox's order updates carry.
SANDBOX_BROKER = "sandbox"
#: The `mode` of an order update in each mode (subscribers/wsproxy_subscriber.py).
UPDATE_MODES = {Mode.LIVE: "live", Mode.SANDBOX: "analyze"}
#: The live stream's word for a stop resting in the stop-loss book.
_STREAM_TRIGGER_PENDING = "trigger_pending"
#: The feed modes and the name a broker's adapter gives each.
LTP, QUOTE, DEPTH = 1, 2, 3
_ADAPTER_MODES = {LTP: "ltp", QUOTE: "quote", DEPTH: "full"}


def order_update(
    change: Change, mode: Mode, *, broker: str, user_id: str, zero_price: bool = False
) -> dict[str, Any]:
    """The order-updates stream message for one change of an order. `zero_price` reports
    a fill at an average price of 0, as the server's order-book poller does for a book
    row without one (websocket_proxy/order_adapter.py, `PollingOrderUpdateAdapter`)."""
    order = change.order
    message = {
        "type": "order_update",
        "user_id": user_id,
        "mode": UPDATE_MODES[mode],
        "broker": SANDBOX_BROKER if mode is Mode.SANDBOX else broker,
        "orderid": order.order_id,
        "symbol": order.symbol,
        "exchange": order.exchange,
        "action": order.action,
        "quantity": order.quantity,
        "price": float(order.price),
        "trigger_price": float(order.trigger_price),
        "pricetype": order.price_type,
        "product": order.product,
        "order_status": order.status,
        "filled_quantity": order.filled,
        "pending_quantity": order.pending,
        "average_price": 0.0 if zero_price else order.average_price,
        "rejection_reason": order.rejection_reason,
    }
    if mode is Mode.LIVE:
        if order.status == TRIGGER_PENDING:
            message["order_status"] = _STREAM_TRIGGER_PENDING
        return message
    if change.event == FILLED and change.fill is not None:
        message.update(filled_quantity=order.quantity, pending_quantity=0)
        message["average_price"] = change.fill.price
    else:
        message.update(filled_quantity=0, pending_quantity=0, average_price=0.0)
    return message


def market_data(market: MarketState, mode: int, *, broker: str, ts_ns: int) -> dict[str, Any]:
    """The market-data message of a subscriber in `mode` (1, 2 or 3) after a trade at
    `ts_ns` (UTC nanoseconds). Raises ValueError for another mode."""
    if mode not in _ADAPTER_MODES:
        raise ValueError(f"the feed's modes are 1 (LTP), 2 (Quote) and 3 (Depth), not {mode!r}")
    instrument = market.instrument
    stamp = epoch_ms(ts_ns)
    data: dict[str, Any] = {
        "symbol": instrument.symbol,
        "exchange": instrument.exchange,
        "mode": _ADAPTER_MODES[mode],
        "ltp": market.last_price,
        "ltt": epoch_ms(market.last_trade_ns or ts_ns),
        "timestamp": stamp,
    }
    if mode >= QUOTE:
        data.update(
            volume=market.day_volume,
            last_quantity=0,
            average_price=0,
            total_buy_quantity=0,
            total_sell_quantity=0,
            open=market.day_open,
            high=market.day_high,
            low=market.day_low,
            close=market.prev_close,
        )
    if mode == DEPTH:
        data["depth"] = {
            "buy": [{"price": market.best_bid, "quantity": 0, "orders": 0}],
            "sell": [{"price": market.best_ask, "quantity": 0, "orders": 0}],
        }
    return {
        "type": "market_data",
        "symbol": instrument.symbol,
        "exchange": instrument.exchange,
        "mode": mode,
        "data": data,
        "broker": broker,
    }


def auth_reply(broker: str, user_id: str) -> dict[str, Any]:
    """The proxy's answer to a valid `authenticate` frame."""
    return {
        "type": "auth",
        "status": "success",
        "message": "Authentication successful",
        "broker": broker,
        "user_id": user_id,
        "supported_features": {"ltp": True, "quote": True, "depth": True},
    }


def orders_reply(action: str) -> dict[str, Any]:
    """The proxy's answer to `subscribe_orders` or `unsubscribe_orders`."""
    words = "Subscribed to" if action == "subscribe_orders" else "Unsubscribed from"
    return {"type": action, "status": "success", "message": f"{words} order updates"}


__all__ = [
    "DEPTH",
    "LTP",
    "QUOTE",
    "SANDBOX_BROKER",
    "UPDATE_MODES",
    "auth_reply",
    "market_data",
    "order_update",
    "orders_reply",
]
