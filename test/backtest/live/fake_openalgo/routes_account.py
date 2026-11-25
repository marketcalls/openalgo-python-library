"""The account routes: `orderbook`, `tradebook`, `positionbook`, `holdings`, `funds`,
`gttorderbook`, `analyzer` (the SDK's `analyzerstatus`) and `ping`, each a function of the
venue and the request's JSON that returns (body, HTTP status), run under the venue's
lock.

Live mode reads the broker: an expired login fails the books with the HTTP client's error
(HTTP 500, `_broker_failed`) and answers funds with success and no data (the broker's
margin call returns nothing, broker/zerodha/api/funds.py, `get_margin_data`). Sandbox
mode reads the sandbox's own
books, whatever the broker's login. `ping` and `analyzer` ask OpenAlgo only, so they
answer while the broker's login has expired (services/ping_service.py,
services/analyzer_service.py). The GTT book of a broker that ships no GTT module is
OpenAlgo's HTTP 501, whatever the login; the sandbox answers it for every broker
(services/gtt_orderbook_service.py). A broker's GTT book on an expired login is the
broker's own refusal, passed on with its status (broker/zerodha/api/gtt_api.py,
get_gtt_book). A row a test hides is left out of the next reads of its book
(`hidden_rows`).
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from .answers import (
    BOOK_PATHS,
    EXPIRED_TOKEN,
    GTT_BOOK_STATUSES,
    GTT_NOT_SUPPORTED,
    broker_http_error,
)
from .gtts import gtt_rows
from .hidden_rows import GTT_BOOK, ORDER_BOOK, TRADE_BOOK
from .rows import (
    funds_data,
    holdings_data,
    order_row,
    order_statistics,
    position_row,
    trade_row,
)
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
        rows = venue.hidden.shown(ORDER_BOOK, rows)
        data = {"orders": rows, "statistics": order_statistics(orders, Mode.SANDBOX)}
        return {"status": "success", "data": data, **_ANALYZE}, 200
    if venue.session_expired:
        return _broker_failed("orders")
    rows = venue.hidden.shown(ORDER_BOOK, [order_row(order, Mode.LIVE) for order in orders])
    data = {"orders": rows, "statistics": order_statistics(orders, Mode.LIVE)}
    return {"status": "success", "data": data}, 200


def tradebook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`tradebook`: one row per fill, oldest first from a broker, newest first from the
    sandbox (sandbox/position_manager.py, get_tradebook)."""
    del payload  # the API key only, checked before any route
    book = venue.book
    if book.mode is Mode.SANDBOX:
        rows = [trade_row(fill, Mode.SANDBOX) for fill in reversed(book.fills)]
        rows = venue.hidden.shown(TRADE_BOOK, rows)
        return {"status": "success", "data": rows, **_ANALYZE}, 200
    if venue.session_expired:
        return _broker_failed("trades")
    rows = venue.hidden.shown(TRADE_BOOK, [trade_row(fill, Mode.LIVE) for fill in book.fills])
    return {"status": "success", "data": rows}, 200


def positionbook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`positionbook`: every position of the day, flat ones included (as they stood when
    the book began to lag its fills, while it does: `Venue.lag_positions`)."""
    del payload  # the API key only, checked before any route
    book = venue.book
    rows = [
        position_row(row, venue.market(row.symbol, row.exchange), book.mode)
        for row in venue.position_rows(book)
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


def holdings(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`holdings` (services/holdings_service.py): the book's delivery holdings."""
    del payload  # the API key only, checked before any route
    book = venue.book
    data = holdings_data(book, venue.markets)
    if book.mode is Mode.SANDBOX:
        return {"status": "success", "data": data, **_ANALYZE}, 200
    if venue.session_expired:
        return _broker_failed("holdings")
    return {"status": "success", "data": data}, 200


def gttorderbook(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`gttorderbook` (restx_api/gtt_orderbook.py, services/gtt_orderbook_service.py): the
    GTTs the account holds (`gtts.gtt_rows`: active ones, and with status "all" the ended
    ones where the book keeps them), from the sandbox in analyzer mode and from a broker
    that ships GTT orders; OpenAlgo's 501 from one that does not, and the broker's 403
    refusal of an expired login (module docstring)."""
    status = payload.get("status", "active")
    if status not in GTT_BOOK_STATUSES:
        words = {"status": [f"Must be one of: {', '.join(GTT_BOOK_STATUSES)}."]}
        return {"status": "error", "message": words}, 400
    book = venue.book
    rows = gtt_rows(list(book.gtts.values()), book.mode, status=status, history=venue.gtt_history)
    if venue.mode is Mode.SANDBOX:
        return {"status": "success", "data": venue.hidden.shown(GTT_BOOK, rows), **_ANALYZE}, 200
    if not venue.gtt_supported:
        return {"status": "error", "message": GTT_NOT_SUPPORTED.format(broker=venue.broker)}, 501
    if venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 403
    return {"status": "success", "data": venue.hidden.shown(GTT_BOOK, rows)}, 200


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


__all__ = [
    "analyzer",
    "funds",
    "gttorderbook",
    "holdings",
    "orderbook",
    "ping",
    "positionbook",
    "tradebook",
]
