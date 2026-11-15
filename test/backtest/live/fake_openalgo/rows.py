"""The rows of the server's books, in live mode (a broker's books as OpenAlgo's services
reshape them) and in sandbox mode (the sandbox's own books), written by hand from the
server's code.

Live rows are those every broker's `transform_*_data` writes (broker/zerodha/mapping/
order_data.py), formatted by services/orderbook_service.py, tradebook_service.py and
positionbook_service.py (amounts to 2 decimals, quantities whole). A live order-book row
shows a resting stop as "open" and carries no filled quantity and no average price: the
fills are in the trade book, and
`orderstatus` takes the average price from the first trade-book row of the order
(services/orderstatus_service.py). Sandbox rows are sandbox/order_manager.py's
(get_orderbook, get_order_status) and sandbox/position_manager.py's (get_tradebook,
get_open_positions).

Prices are rupees, quantities units.
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from .book import Book
from .clock import book_time
from .market import Instrument, MarketState
from .orders import (
    CANCELLED,
    COMPLETE,
    MARKET,
    OPEN,
    REJECTED,
    TRIGGER_PENDING,
    FakeOrder,
    Fill,
)
from .positions import Position

#: Decimals of the amounts the server's books write.
_AMOUNT_DECIMALS = 2


def _amount(value: float) -> float:
    return round(float(value), _AMOUNT_DECIMALS)


def symbol_row(instrument: Instrument, row_id: int) -> dict[str, Any]:
    """The `data` of a `symbol` answer (services/symbol_service.py); `row_id` is the
    instrument's row in the symbol table, also used as its token."""
    return {
        "id": row_id,
        "symbol": instrument.symbol,
        "brsymbol": instrument.symbol,
        "name": instrument.symbol,
        "exchange": instrument.exchange,
        "brexchange": instrument.exchange,
        "token": str(row_id),
        "expiry": instrument.expiry,
        "strike": instrument.strike,
        "lotsize": instrument.lot_size,
        "instrumenttype": instrument.instrument_type,
        "tick_size": instrument.tick_size,
        "freeze_qty": instrument.freeze_qty,
    }


def order_row(order: FakeOrder, mode: Mode) -> dict[str, Any]:
    """An order-book row."""
    if mode is Mode.SANDBOX:
        return {
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
            "average_price": _amount(order.average_price),
            "filled_quantity": order.filled,
            # A cancel changes the status only, so the cancelled order keeps its pending
            # units (sandbox/order_manager.py:1139-1142); a fill or a rejection at
            # placement leaves none.
            "pending_quantity": order.remaining if order.status == CANCELLED else order.pending,
            "rejection_reason": order.rejection_reason,
            "timestamp": book_time(order.placed_ns),
            "strategy": order.strategy,
        }
    return {
        "symbol": order.symbol,
        "exchange": order.exchange,
        "action": order.action,
        "quantity": order.quantity,
        # services/orderbook_service.py writes 0 as the price of every MARKET order.
        "price": 0.0 if order.price_type == MARKET else _amount(order.price),
        "trigger_price": _amount(order.trigger_price),
        "pricetype": order.price_type,
        "product": order.product,
        "orderid": order.order_id,
        # The REST book shows a resting stop as open, its working state
        # (broker/zerodha/mapping/order_data.py:194-195); the stream keeps the word.
        "order_status": OPEN if order.status == TRIGGER_PENDING else order.status,
        "timestamp": book_time(order.placed_ns),
    }


def order_statistics(orders: list[FakeOrder], mode: Mode) -> dict[str, int]:
    """The `statistics` of an order-book answer. A broker counts trigger-pending orders
    as open (broker/zerodha/mapping/order_data.py); the sandbox counts them apart
    (sandbox/order_manager.py, `_calculate_order_statistics`)."""
    stats = {
        "total_buy_orders": sum(1 for o in orders if o.action == "BUY"),
        "total_sell_orders": sum(1 for o in orders if o.action == "SELL"),
        "total_completed_orders": sum(1 for o in orders if o.status == COMPLETE),
        "total_open_orders": sum(1 for o in orders if o.status == OPEN),
        "total_rejected_orders": sum(1 for o in orders if o.status == REJECTED),
    }
    pending = sum(1 for o in orders if o.status == TRIGGER_PENDING)
    if mode is Mode.SANDBOX:
        stats["total_trigger_pending_orders"] = pending
    else:
        stats["total_open_orders"] += pending
    return stats


def order_status_data(order: FakeOrder, book: Book) -> dict[str, Any]:
    """The `data` of an `orderstatus` answer."""
    if book.mode is Mode.SANDBOX:
        row = order_row(order, Mode.SANDBOX)
        row["price_type"] = row.pop("pricetype")
        del row["rejection_reason"]
        return row
    row = order_row(order, Mode.LIVE)
    first = next((fill for fill in book.fills if fill.order_id == order.order_id), None)
    complete = order.status == COMPLETE and first is not None
    row["average_price"] = _amount(first.price) if complete else 0.0
    return row


def trade_row(fill: Fill, mode: Mode) -> dict[str, Any]:
    """A trade-book row."""
    value = _amount(fill.price * fill.quantity)
    if mode is Mode.SANDBOX:
        return {
            "tradeid": fill.trade_id,
            "orderid": fill.order_id,
            "symbol": fill.symbol,
            "exchange": fill.exchange,
            "action": fill.action,
            "quantity": fill.quantity,
            "average_price": _amount(fill.price),
            "price": _amount(fill.price),
            "trade_value": value,
            "product": fill.product,
            "strategy": fill.strategy,
            "timestamp": book_time(fill.ts_ns),
        }
    return {
        "symbol": fill.symbol,
        "exchange": fill.exchange,
        "product": fill.product,
        "action": fill.action,
        "quantity": fill.quantity,
        "average_price": _amount(fill.price),
        "trade_value": value,
        "orderid": fill.order_id,
        "tradeid": fill.trade_id,
        "timestamp": book_time(fill.ts_ns),
    }


def position_row(position: Position, market: MarketState | None, mode: Mode) -> dict[str, Any]:
    """A position-book row; `market` gives the last price (None when never traded)."""
    last = market.last_price if market is not None else None
    unrealized = position.unrealized(last)
    pnl = _amount(position.realized + unrealized)
    if mode is Mode.SANDBOX:
        return {
            "symbol": position.symbol,
            "exchange": position.exchange,
            "product": position.product,
            "quantity": position.quantity,
            "average_price": float(position.average_price),
            "ltp": float(last or 0.0),
            "pnl": pnl,
            "pnlpercent": _percent(pnl, position),
            "unrealized_pnl": _amount(unrealized),
            "today_realized_pnl": _amount(position.realized),
            "total_pnl_today": pnl,
            "lot_size": 1.0,
        }
    return {
        "symbol": position.symbol,
        "exchange": position.exchange,
        "product": position.product,
        "quantity": position.quantity,
        "pnl": pnl,
        # A broker's average price arrives as text with two decimals.
        "average_price": f"{position.average_price:.2f}",
        "ltp": _amount(last or 0.0),
    }


def _percent(pnl: float, position: Position) -> float:
    investment = abs(position.average_price * position.quantity)
    return _amount(pnl / investment * 100) if investment else 0.0


def funds_data(book: Book, markets: dict[tuple[str, str], MarketState]) -> dict[str, Any]:
    """The `data` of a `funds` answer: text amounts from a broker
    (broker/zerodha/api/funds.py), numbers from the sandbox (sandbox/fund_manager.py)."""
    realized = book.positions.realized()
    unrealized = sum(
        row.unrealized(_last(markets, row.symbol, row.exchange)) for row in book.positions.rows()
    )
    if book.mode is Mode.SANDBOX:
        return {
            "availablecash": float(book.cash),
            "collateral": 0.0,
            "m2munrealized": _amount(unrealized),
            "m2mrealized": _amount(realized),
            "total_realized_pnl": _amount(realized),
            "today_realized_pnl": _amount(realized),
            "utiliseddebits": 0.0,
            "grossexposure": 0.0,
            "totalpnl": _amount(realized + unrealized),
        }
    return {
        "availablecash": f"{book.cash:.2f}",
        "collateral": "0.00",
        "m2mrealized": f"{realized:.2f}",
        "m2munrealized": f"{unrealized:.2f}",
        "utiliseddebits": "0.00",
    }


def _last(markets: dict[tuple[str, str], MarketState], symbol: str, exchange: str) -> float | None:
    market = markets.get((symbol, exchange))
    return market.last_price if market is not None else None


__all__ = [
    "funds_data",
    "order_row",
    "order_statistics",
    "order_status_data",
    "position_row",
    "symbol_row",
    "trade_row",
]
