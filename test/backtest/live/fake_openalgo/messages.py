"""The messages of OpenAlgo's WebSocket proxy, written by hand from the server's code.

- Market data: `{"type": "market_data", "symbol", "exchange", "mode", "data", "broker"}`,
  `mode` the subscriber's mode (1 LTP, 2 Quote, 3 Depth) and `data` what the broker's
  adapter publishes for it (websocket_proxy/server.py, its market-data listener;
  broker/zerodha/streaming/zerodha_adapter.py, `_transform_regular_tick`).
- Order updates: the payload of subscribers/wsproxy_subscriber.py (`on_order_update`),
  relayed as it is (websocket_proxy/server.py, `_handle_order_update`). A live broker's
  status words are its order adapter's: a resting stop is "trigger pending" with a space
  (broker/zerodha/streaming/zerodha_order_adapter.py:72, through the status map of
  broker/zerodha/mapping/order_data.py:34-48), as the sandbox writes it too.
  The sandbox announces a placement, a trigger and a cancellation without fill figures
  (filled, pending and average all 0, the event's defaults: sandbox/order_manager.py
  and sandbox/execution_engine.py, `_publish_order_update_event`) and a fill as complete
  for the whole order (`_publish_fill_event`).
- Replies: authentication and the order-stream subscription (websocket_proxy/server.py).

Prices are rupees, quantities units, times epoch milliseconds on the wire.
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._layout import NANOS_PER_SECOND
from openalgo.backtest._live._types import Mode

from .book import FILLED, MODIFIED, Change
from .clock import epoch_ms
from .market import MarketState
from .orders import OPEN

#: The `broker` the sandbox's order updates carry.
SANDBOX_BROKER = "sandbox"
#: The `mode` of an order update in each mode (subscribers/wsproxy_subscriber.py).
UPDATE_MODES = {Mode.LIVE: "live", Mode.SANDBOX: "analyze"}
#: The feed modes and the name a broker's adapter gives each.
LTP, QUOTE, DEPTH = 1, 2, 3
#: The brokers whose adapter gives the last trade time of an LTP or quote tick as its
#: own clock's stamp (the reference broker's); the fake plays any other broker's adapter
#: as giving the trade's exchange time in milliseconds.
STAMPING_BROKERS = frozenset({"zerodha"})
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
        if change.event == MODIFIED:
            # A broker reports a modification as UPDATE, which the adapter reads as
            # open, even for a stop still resting (broker/zerodha/mapping/
            # order_data.py:39).
            message["order_status"] = OPEN
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
    # The adapter stamps a tick with its own time in milliseconds
    # (broker/zerodha/streaming/zerodha_websocket.py:740), and gives the last trade time
    # as that stamp unless the packet carries the exchange's, which only a depth packet
    # does, in epoch seconds (zerodha_websocket.py:826; zerodha_adapter.py:677-680).
    # Another broker is played as most adapters are: the trade's exchange time in
    # milliseconds in every mode (16-session-design 7.3, for F61).
    stamp = epoch_ms(ts_ns)
    trade_ns = market.last_trade_ns or ts_ns
    if broker not in STAMPING_BROKERS:
        ltt = epoch_ms(trade_ns)
    elif mode == DEPTH:
        ltt = trade_ns // NANOS_PER_SECOND
    else:
        ltt = stamp
    data: dict[str, Any] = {
        "symbol": instrument.symbol,
        "exchange": instrument.exchange,
        "mode": _ADAPTER_MODES[mode],
        "ltp": market.last_price,
        "ltt": ltt,
        "timestamp": stamp,
    }
    if mode >= QUOTE:
        data.update(
            # The packet's quantities are integers (zerodha_websocket.py:779-780).
            volume=int(market.day_volume),
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


#: The proxy's names of the feed modes (websocket_proxy/mode_utils.py, MODE_CANONICAL),
#: and the depth it reports for a subscription.
_MODE_LABELS = {LTP: "LTP", QUOTE: "Quote", DEPTH: "Depth"}
_DEPTH_LEVELS = 5


def subscribe_reply(entries: list[tuple[str, str, int, str]], broker: str) -> dict[str, Any]:
    """The proxy's answer to a `subscribe` frame: each (symbol, exchange, mode, refusal)
    accepted when its refusal is "", refused with it otherwise, and the answer "partial"
    when any was refused (websocket_proxy/server.py, subscribe_client)."""
    rows = []
    for symbol, exchange, mode, refusal in entries:
        base = {"symbol": symbol, "exchange": exchange}
        if refusal:
            rows.append(base | {"status": "error", "message": refusal, "broker": broker})
        else:
            label = _MODE_LABELS.get(mode, mode)
            accepted = {"status": "success", "mode": label, "depth": _DEPTH_LEVELS}
            rows.append(base | accepted | {"broker": broker})
    return {
        "type": "subscribe",
        "status": "partial" if any(entry[3] for entry in entries) else "success",
        "subscriptions": rows,
        "message": "Subscription processing complete",
        "broker": broker,
    }


def token_not_found(symbol: str, exchange: str) -> str:
    """The broker adapter's words for a subscription to an instrument it does not know
    (broker/zerodha/streaming/zerodha_adapter.py:282-284)."""
    return f"Token not found for {symbol} on {exchange}"


def orders_reply(action: str) -> dict[str, Any]:
    """The proxy's answer to `subscribe_orders` or `unsubscribe_orders`."""
    words = "Subscribed to" if action == "subscribe_orders" else "Unsubscribed from"
    return {"type": action, "status": "success", "message": f"{words} order updates"}


__all__ = [
    "DEPTH",
    "LTP",
    "QUOTE",
    "SANDBOX_BROKER",
    "STAMPING_BROKERS",
    "UPDATE_MODES",
    "auth_reply",
    "market_data",
    "order_update",
    "orders_reply",
    "subscribe_reply",
    "token_not_found",
]
