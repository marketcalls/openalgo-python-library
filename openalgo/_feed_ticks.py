"""The shape of the market data the feed stores and hands to callbacks
(private to openalgo.feed). Pure functions: nothing here touches the feed."""

import time
from typing import Any

# The quote fields a callback and get_quotes() see, in this order, each with the names
# the server may send it under (the first one present wins). A field the server leaves
# out is 0. The server's broker adapters send the average traded price as
# average_price and the last traded quantity as last_quantity; the SDK's own names
# are still read for a server that sends them.
QUOTE_SOURCES = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "ltp": ("ltp",),
    "volume": ("volume",),
    "last_trade_quantity": ("last_quantity", "last_trade_quantity"),
    "avg_trade_price": ("average_price", "avg_trade_price"),
    "change": ("change",),
    "change_percent": ("change_percent",),
    "ltt": ("ltt",),
    "total_buy_quantity": ("total_buy_quantity",),
    "total_sell_quantity": ("total_sell_quantity",),
    "oi": ("oi",),
}
QUOTE_FIELDS = tuple(QUOTE_SOURCES)


def _timestamp(market_data: dict[str, Any]) -> Any:
    """The server's timestamp, or the local time in milliseconds when it sent none."""
    return market_data.get("timestamp", int(time.time() * 1000))


def ltp_update(market_data: dict[str, Any]) -> dict[str, Any]:
    """The "data" of an LTP update: price, timestamp, and the last trade time if any."""
    update = {"ltp": market_data.get("ltp"), "timestamp": _timestamp(market_data)}
    # The last trade time arrives at the top level, or nested one level down.
    if "ltt" in market_data:
        update["ltt"] = market_data["ltt"]
    elif "ltt" in market_data.get("data", {}):
        update["ltt"] = market_data["data"]["ltt"]
    return update


def _first_present(market_data: dict[str, Any], names: tuple[str, ...]) -> Any:
    """The value under the first of `names` the server sent, else 0."""
    for name in names:
        if name in market_data:
            return market_data[name]
    return 0


def quote_update(market_data: dict[str, Any]) -> dict[str, Any]:
    """The "data" of a quote update: the quote fields and the timestamp."""
    update = {field: _first_present(market_data, names) for field, names in QUOTE_SOURCES.items()}
    update["timestamp"] = _timestamp(market_data)
    return update


def depth_update(market_data: dict[str, Any]) -> dict[str, Any]:
    """The "data" of a depth update: price, timestamp and the two sides of the book."""
    return {
        "ltp": market_data.get("ltp", 0),
        "timestamp": _timestamp(market_data),
        "depth": market_data.get("depth", {"buy": [], "sell": []}),
    }


def callback_message(message: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """The clean market data update a data callback receives."""
    return {
        "type": "market_data",
        "symbol": message["symbol"],
        "exchange": message["exchange"],
        "mode": message.get("mode"),
        "data": data,
    }


def depth_ladder(depth_data: dict[str, Any]) -> list[str]:
    """The buy and sell levels side by side: the lines verbose level 2 prints."""
    buy_depth = depth_data.get("depth", {}).get("buy", [])
    sell_depth = depth_data.get("depth", {}).get("sell", [])
    heading = f"{'Price':<10} {'Qty':<10} {'Orders':<8}"
    lines = [f"{'BUY':<30} | {'SELL':<30}", f"{heading} | {heading}", "-" * 62]
    for i in range(max(len(buy_depth), len(sell_depth), 1)):
        sides = []
        for levels in (buy_depth, sell_depth):
            level = levels[i] if i < len(levels) else {}
            cells = (level.get(name, "-") for name in ("price", "quantity", "orders"))
            price, quantity, orders = (str(cell) for cell in cells)
            sides.append(f"{price:<10} {quantity:<10} {orders:<8}")
        lines.append(" | ".join(sides))
    return [f"         {line}" for line in lines]
