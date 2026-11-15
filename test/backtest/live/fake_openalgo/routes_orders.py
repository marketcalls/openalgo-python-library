"""The order routes: `placeorder`, `modifyorder`, `cancelorder` and `orderstatus`, each a
function of the venue and the request's JSON that returns (body, HTTP status). They run
under the venue's lock.

The order of the checks and the words are the server's: the REST layer's schema
(`schema`), semi-auto (services/order_router_service.py queues a placement in either
mode; services/modify_order_service.py and cancel_order_service.py refuse a modify or
cancel in live mode only), then the switch's book: the sandbox's checks and answers
(sandbox/order_manager.py) or the broker's. A broker's refusals are in the broker's own
words, which differ between brokers: the fake's are its own (`BROKER_WORDS`).
"""

from __future__ import annotations

from typing import Any

from openalgo.backtest._live._types import Mode

from . import schema
from .answers import (
    EXPIRED_TOKEN,
    QUEUED_FOR_APPROVAL,
    SEMI_AUTO_MODE,
    SEMI_AUTO_NO_CANCEL,
    SEMI_AUTO_NO_MODIFY,
)
from .book import OrderTerms
from .orders import LIMIT_TYPES, MARKET, STOP_MARKET, STOP_TYPES, FakeOrder
from .rows import order_status_data
from .venue import Venue

Answer = tuple[dict[str, Any], int]

#: The fake broker's refusals at placement and on a modify or cancel (HTTP 400).
BROKER_WORDS = {
    "unknown": "Instrument {exchange}:{symbol} is not tradable at the broker",
    "no_price": "A {price_type} order needs a price above 0",
    "no_trigger": "A {price_type} order needs a trigger price above 0",
    "lot": "Quantity {quantity} is not a multiple of the lot size {lot}",
    "freeze": "Quantity {quantity} is above the freeze quantity {freeze}",
    "option_stop_market": (
        "SL-M orders are not allowed for {exchange} options: place an SL (stop-limit) order"
    ),
    "no_order": "Order {orderid} not found at the broker",
    "finished": "Order {orderid} is {status} and cannot be changed",
}


def place(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`placeorder` (services/place_order_service.py)."""
    if venue.semi_auto:
        queued = {
            "status": "success",
            "message": QUEUED_FOR_APPROVAL,
            "mode": SEMI_AUTO_MODE,
            "pending_order_id": venue.next_pending_order_id(),
        }
        return queued, 200
    loaded, problem = schema.load(schema.PLACE, payload)
    if problem:
        return {"status": "error", "message": problem}, 400
    price_type = loaded["pricetype"]
    terms = OrderTerms(
        strategy=loaded["strategy"],
        symbol=loaded["symbol"],
        exchange=loaded["exchange"],
        action=loaded["action"].upper(),
        quantity=loaded["quantity"],
        price_type=price_type,
        product=loaded["product"],
        # The price and trigger an order type does not use are kept as none
        # (sandbox/order_manager.py, place_order), and written as 0.
        price=loaded["price"] if price_type in LIMIT_TYPES else 0.0,
        trigger_price=loaded["trigger_price"] if price_type in STOP_TYPES else 0.0,
    )
    sandbox = venue.mode is Mode.SANDBOX
    if not sandbox and venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 403
    refusal = _sandbox_refusal(venue, terms) if sandbox else _broker_refusal(venue, terms)
    if not refusal:
        rejection = venue.take_rejection(price_type, at_placement=True)
        refusal = rejection.reason if rejection is not None else ""
    if refusal:
        return _marked(venue, {"status": "error", "message": refusal}), 400
    order = venue.accept(venue.book, terms)
    return _marked(venue, {"status": "success", "orderid": order.order_id}), 200


def modify(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`modifyorder` (services/modify_order_service.py; sandbox/order_manager.py,
    modify_order: the sandbox changes the quantity, price and trigger only)."""
    loaded, problem = schema.load(schema.MODIFY, payload)
    if problem:
        return _marked(venue, {"status": "error", "message": problem}), 400
    if venue.semi_auto and venue.mode is Mode.LIVE:
        return {"status": "error", "message": SEMI_AUTO_NO_MODIFY}, 403
    order_id = loaded["orderid"]
    if venue.mode is Mode.SANDBOX:
        return _sandbox_modify(venue, loaded)
    if venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 403
    order = venue.book.orders.get(order_id)
    refusal = _unchangeable(order, order_id)
    if not refusal and loaded["quantity"] <= order.filled:
        refusal = f"Quantity {loaded['quantity']} is not above the {order.filled} units filled"
    if refusal:
        return {"status": "error", "message": refusal}, 400
    venue.modify(
        venue.book,
        order,
        quantity=loaded["quantity"],
        price_type=loaded["pricetype"],
        price=loaded["price"] if loaded["pricetype"] in LIMIT_TYPES else 0.0,
        trigger=loaded["trigger_price"] if loaded["pricetype"] in STOP_TYPES else 0.0,
    )
    return {"status": "success", "orderid": order_id}, 200


def cancel(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`cancelorder` (services/cancel_order_service.py; sandbox/order_manager.py,
    cancel_order)."""
    loaded, problem = schema.load(schema.CANCEL, payload)
    if problem:
        return _marked(venue, {"status": "error", "message": problem}), 400
    if venue.semi_auto and venue.mode is Mode.LIVE:
        return {"status": "error", "message": SEMI_AUTO_NO_CANCEL}, 403
    order_id = loaded["orderid"]
    order = venue.book.orders.get(order_id)
    if venue.mode is Mode.SANDBOX:
        if order is None:
            return _not_found(venue, order_id), 404
        if not order.working:
            words = f"Cannot cancel order in {order.status} status"
            return _marked(venue, {"status": "error", "message": words}), 400
        venue.cancel(venue.book, order)
        done = {"status": "success", "orderid": order_id, "message": "Order cancelled successfully"}
        return _marked(venue, done), 200
    if venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 403
    refusal = _unchangeable(order, order_id)
    if refusal:
        return {"status": "error", "message": refusal}, 400
    venue.cancel(venue.book, order)
    return {"status": "success", "orderid": order_id}, 200


def status(venue: Venue, payload: dict[str, Any]) -> Answer:
    """`orderstatus` (services/orderstatus_service.py: live mode reads the broker's order
    book, so an expired login fails as the order book does)."""
    loaded, problem = schema.load(schema.STATUS, payload)
    if problem:
        return _marked(venue, {"status": "error", "message": problem}), 400
    if venue.mode is Mode.LIVE and venue.session_expired:
        return {"status": "error", "message": EXPIRED_TOKEN}, 500
    order = venue.book.orders.get(loaded["orderid"])
    if order is None:
        return _not_found(venue, loaded["orderid"]), 404
    return _marked(venue, {"status": "success", "data": order_status_data(order, venue.book)}), 200


def _sandbox_modify(venue: Venue, loaded: dict[str, Any]) -> Answer:
    order_id = loaded["orderid"]
    order = venue.book.orders.get(order_id)
    if order is None:
        return _not_found(venue, order_id), 404
    if not order.working:
        words = f"Cannot modify order in {order.status} status"
    elif loaded["price"] and order.price_type not in LIMIT_TYPES:
        words = f"{order.price_type} orders do not accept a price"
    elif loaded["trigger_price"] and order.price_type not in STOP_TYPES:
        words = f"{order.price_type} orders do not accept a trigger_price"
    else:
        words = _lot_problem(venue, order.symbol, order.exchange, loaded["quantity"])
    if words:
        return _marked(venue, {"status": "error", "message": words}), 400
    venue.modify(
        venue.book,
        order,
        quantity=loaded["quantity"],
        price_type=order.price_type,
        price=loaded["price"] or order.price,
        trigger=loaded["trigger_price"] or order.trigger_price,
    )
    done = {"status": "success", "orderid": order_id, "message": "Order modified successfully"}
    return _marked(venue, done), 200


def _sandbox_refusal(venue: Venue, terms: OrderTerms) -> str:
    """The sandbox's refusals, in its order (sandbox/order_manager.py, `_validate_order`
    then place_order)."""
    if terms.price_type in LIMIT_TYPES and not terms.price:
        return f"{terms.price_type} orders require price"
    if terms.price_type in STOP_TYPES and not terms.trigger_price:
        return f"{terms.price_type} orders require trigger_price"
    market = venue.market(terms.symbol, terms.exchange)
    if market is None:
        return f"Symbol {terms.symbol} not found on {terms.exchange}"
    lot = _lot_problem(venue, terms.symbol, terms.exchange, terms.quantity)
    if lot:
        return lot
    if terms.price_type == MARKET and market.last_price is None:
        return (
            f"Cannot place MARKET order for {terms.symbol} - unable to fetch current price. "
            "Please try again later or use LIMIT order with a specific price."
        )
    return ""


def _broker_refusal(venue: Venue, terms: OrderTerms) -> str:
    """The fake broker's refusals at placement."""
    fields = {"symbol": terms.symbol, "exchange": terms.exchange, "quantity": terms.quantity}
    market = venue.market(terms.symbol, terms.exchange)
    if market is None:
        return BROKER_WORDS["unknown"].format(**fields)
    instrument = market.instrument
    if terms.price_type in LIMIT_TYPES and not terms.price:
        return BROKER_WORDS["no_price"].format(price_type=terms.price_type)
    if terms.price_type in STOP_TYPES and not terms.trigger_price:
        return BROKER_WORDS["no_trigger"].format(price_type=terms.price_type)
    if terms.price_type == STOP_MARKET and instrument.is_option:
        return BROKER_WORDS["option_stop_market"].format(**fields)
    if instrument.trades_in_lots and terms.quantity % instrument.lot_size:
        return BROKER_WORDS["lot"].format(lot=instrument.lot_size, **fields)
    if instrument.freeze_qty and terms.quantity > instrument.freeze_qty:
        return BROKER_WORDS["freeze"].format(freeze=instrument.freeze_qty, **fields)
    return ""


def _lot_problem(venue: Venue, symbol: str, exchange: str, quantity: int) -> str:
    market = venue.market(symbol, exchange)
    if market is None or not market.instrument.trades_in_lots:
        return ""
    lot = market.instrument.lot_size
    return f"Quantity must be in multiples of lot size {lot}" if quantity % lot else ""


def _unchangeable(order: FakeOrder | None, order_id: str) -> str:
    if order is None:
        return BROKER_WORDS["no_order"].format(orderid=order_id)
    if not order.working:
        return BROKER_WORDS["finished"].format(orderid=order_id, status=order.status)
    return ""


def _not_found(venue: Venue, order_id: str) -> dict[str, Any]:
    return _marked(venue, {"status": "error", "message": f"Order {order_id} not found"})


def _marked(venue: Venue, body: dict[str, Any]) -> dict[str, Any]:
    """The body with the sandbox's `"mode": "analyze"` when the switch is on."""
    if venue.mode is Mode.SANDBOX:
        venue.analyzer_logs += 1
        return {**body, "mode": "analyze"}
    return body


__all__ = ["BROKER_WORDS", "cancel", "modify", "place", "status"]
