"""The account routes: `orderbook`, `tradebook`, `positionbook`, `funds`, `analyzer` (the
SDK's `analyzerstatus`) and `ping`, each a function of the venue and the request's JSON
that returns (body, HTTP status), run under the venue's lock.

Live mode reads the broker: an expired login fails the books with the HTTP client's error
(HTTP 500, `_broker_failed`) and answers funds with success and no data (the broker's
margin call returns nothing, broker/zerodha/api/funds.py, `get_margin_data`). Sandbox
mode reads the sandbox's own
books, whatever the broker's login. `ping` and `analyzer` ask OpenAlgo only, so they
answer while the broker's login has expired (services/ping_service.py,
services/analyzer_service.py).
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from .answers import BOOK_PATHS, broker_http_error
from .rows import funds_data, order_row, order_statistics, position_row, trade_row
from .venue import Venue

Answer = tuple[dict[str, Any], int]
_ANALYZE = {"mode": "analyze"}


def orderbook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`orderbook`: a broker lists its orders oldest first, the sandbox newest first
    (sandbox/order_manager.py, get_orderbook)."""
    del payload  # the API key only, checked before any route
    book = venue.book
    orders = list(book.orders.values())
    if book.mode is Mode.SANDBOX:
        rows = [order_row(order, Mode.SANDBOX) for order in reversed(orders)]
        data = {"orders": rows, "statistics": order_statistics(orders, Mode.SANDBOX)}
        return {"status": "success", "data": data, **_ANALYZE}, 200
    if venue.session_expired:
        return _broker_failed("orders")
    rows = [order_row(order, Mode.LIVE) for order in orders]
    data = {"orders": rows, "statistics": order_statistics(orders, Mode.LIVE)}
    return {"status": "success", "data": data}, 200


def tradebook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`tradebook`: one row per fill, oldest first from a broker, newest first from the
    sandbox (sandbox/position_manager.py, get_tradebook)."""
    del payload  # the API key only, checked before any route
    book = venue.book
    if book.mode is Mode.SANDBOX:
        rows = [trade_row(fill, Mode.SANDBOX) for fill in reversed(book.fills)]
        return {"status": "success", "data": rows, **_ANALYZE}, 200
    if venue.session_expired:
        return _broker_failed("trades")
    return {"status": "success", "data": [trade_row(fill, Mode.LIVE) for fill in book.fills]}, 200


def positionbook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`positionbook`: every position of the day, flat ones included."""
    del payload  # the API key only, checked before any route
    book = venue.book
    rows = [
        position_row(row, venue.market(row.symbol, row.exchange), book.mode)
        for row in book.positions.rows()
    ]
    if book.mode is Mode.LIVE:
        if venue.session_expired:
            return _broker_failed("positions")
        return {"status": "success", "data": rows}, 200
    unrealized = sum(row["unrealized_pnl"] for row in rows)
    realized = sum(row["today_realized_pnl"] for row in rows)
    total = round(unrealized + realized, 2)
    answer = {
        "status": "success",
        "data": rows,
        "total_pnl": total,
        "total_unrealized_pnl": round(unrealized, 2),
        "total_today_realized_pnl": round(realized, 2),
        "total_pnl_today": total,
    }
    return {**answer, **_ANALYZE}, 200


def funds(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`funds` (services/funds_service.py)."""
    del payload  # the API key only, checked before any route
    book = venue.book
    data = funds_data(book, venue.markets)
    if book.mode is Mode.SANDBOX:
        return {"status": "success", "data": data, **_ANALYZE}, 200
    if venue.session_expired:
        return {"status": "success", "data": {}}, 200
    return {"status": "success", "data": data}, 200


def analyzer(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`analyzer`, the analyzer switch (services/analyzer_service.py,
    get_analyzer_status_with_auth)."""
    del payload  # the API key only, checked before any route
    data = {
        "mode": "analyze" if venue.analyzer else "live",
        "analyze_mode": venue.analyzer,
        "total_logs": venue.analyzer_logs,
    }
    return {"status": "success", "data": data}, 200


def ping(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`ping` (services/ping_service.py): OpenAlgo holds a login for the broker."""
    del payload  # the API key only, checked before any route
    return {"status": "success", "data": {"message": "pong", "broker": venue.broker}}, 200


def _broker_failed(book: str) -> Answer:
    """A book the broker refused for an expired login: the HTTP client's error, re-raised
    by the broker module (broker/zerodha/api/order_api.py:76-91) and answered as its text
    with HTTP 500 (services/orderbook_service.py:182-184, tradebook_service.py:135-137,
    positionbook_service.py:148-150)."""
    return {"status": "error", "message": broker_http_error(403, BOOK_PATHS[book])}, 500


__all__ = ["analyzer", "funds", "orderbook", "ping", "positionbook", "tradebook"]
