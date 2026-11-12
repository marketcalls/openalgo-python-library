"""The order-update stream with sandbox orders (needs analyzer mode and the MCX session).

Safety. `place_in_sandbox` asserts that the server is in analyzer mode immediately
before every order and that every reply says `"mode": "analyze"`. The module places at
most four orders, each of one lot of a near-month MCX future: a LIMIT buy far below
the market that is cancelled, a MARKET buy and the MARKET sell that offsets it, and
one offsetting order only if a position of this module were left open. Teardown
cancels only the order IDs placed here and offsets only this module's position. No
account-wide action (close all positions, cancel all orders, the analyzer switch) is
ever called.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from live_support import (
    live_only,
    near_month_future,
    new_client,
    require_sandbox,
    session_is_open,
    wait_until,
)

pytestmark = live_only

UNDERLYING = "CRUDEOILM"  # the MCX future with the smallest contract value
EXCHANGE = "MCX"
PRODUCT = "MIS"
STRATEGY = "sdk-live-test"
# A buy limit this far below the last price cannot fill during the test.
LIMIT_DISCOUNT = 0.15
UPDATE_WAIT_SECONDS = 15
SUBSCRIBE_SETTLE_SECONDS = 2
OPEN_STATES = ("open", "pending", "trigger pending")


@dataclass
class OrderStream:
    """A feed client subscribed to order updates, and what this module placed."""

    rest: Any
    feed: Any
    symbol: str
    lot: int
    tick: float
    updates: list[dict] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def place(self, **order: Any) -> str:
        """Place one order in the sandbox and remember its ID."""
        require_sandbox(self.rest)
        reply = self.rest.placeorder(
            strategy=STRATEGY, symbol=self.symbol, exchange=EXCHANGE, product=PRODUCT, **order
        )
        assert reply.get("mode") == "analyze", f"an order reply without analyze mode: {reply}"
        assert reply.get("status") == "success", reply
        self.placed.append(reply["orderid"])
        return reply["orderid"]

    def updates_for(self, order_id: str, status: str | None = None) -> list[dict]:
        with self.lock:
            return [
                update
                for update in self.updates
                if update.get("orderid") == order_id
                and (status is None or str(update.get("order_status")).lower() == status)
            ]

    def wait_for(self, order_id: str, status: str) -> list[dict]:
        wait_until(lambda: self.updates_for(order_id, status), UPDATE_WAIT_SECONDS)
        return self.updates_for(order_id, status)

    def position(self) -> int:
        reply = self.rest.openposition(
            strategy=STRATEGY, symbol=self.symbol, exchange=EXCHANGE, product=PRODUCT
        )
        assert reply.get("mode") == "analyze", reply
        return int(float(reply.get("quantity") or 0))

    def clean_up(self) -> None:
        """Cancel this module's working orders and offset this module's position."""
        for order_id in self.placed:
            status = self.rest.orderstatus(order_id=order_id, strategy=STRATEGY)
            state = str((status.get("data") or {}).get("order_status", "")).lower()
            if state in OPEN_STATES:
                self.rest.cancelorder(order_id=order_id, strategy=STRATEGY)
        held = self.position()
        if held:
            self.place(
                action="SELL" if held > 0 else "BUY", price_type="MARKET", quantity=abs(held)
            )


@pytest.fixture(scope="module")
def stream(client):
    if not session_is_open(EXCHANGE, until_ns=lambda times: times.entry_until):
        pytest.skip("the MCX session is not open for new intraday orders")
    row = near_month_future(client, UNDERLYING, EXCHANGE)
    feed = new_client()
    orders = OrderStream(client, feed, row["symbol"], int(row["lotsize"]), float(row["tick_size"]))

    def on_update(update: dict) -> None:
        with orders.lock:
            orders.updates.append(update)

    try:
        assert feed.connect(), "connect() did not authenticate with the feed"
        assert feed.subscribe_orders(on_update)
        # The server's acknowledgement of the subscription is not exposed; give it time.
        time.sleep(SUBSCRIBE_SETTLE_SECONDS)
        yield orders
    finally:
        try:
            orders.clean_up()
        finally:
            feed.disconnect()
            feed.close()


@pytest.fixture(scope="module")
def cancelled_limit(stream) -> dict:
    """A LIMIT buy far below the market, then its cancellation."""
    quote = stream.rest.quotes(symbol=stream.symbol, exchange=EXCHANGE)
    last = float(quote["data"]["ltp"])
    price = round(last * (1 - LIMIT_DISCOUNT) / stream.tick) * stream.tick
    order_id = stream.place(action="BUY", price_type="LIMIT", quantity=stream.lot, price=price)
    opened = stream.wait_for(order_id, "open")
    status_open = stream.rest.orderstatus(order_id=order_id, strategy=STRATEGY)
    cancel = stream.rest.cancelorder(order_id=order_id, strategy=STRATEGY)
    cancelled = stream.wait_for(order_id, "cancelled")
    status_cancelled = stream.rest.orderstatus(order_id=order_id, strategy=STRATEGY)
    return {
        "order_id": order_id,
        "opened": opened,
        "status_open": status_open,
        "cancel": cancel,
        "cancelled": cancelled,
        "status_cancelled": status_cancelled,
    }


@pytest.fixture(scope="module")
def round_trip(stream) -> dict:
    """A MARKET buy of one lot and the MARKET sell that offsets it."""
    buy_id = stream.place(action="BUY", price_type="MARKET", quantity=stream.lot)
    bought = stream.wait_for(buy_id, "complete")
    held = stream.position()
    sell_id = stream.place(action="SELL", price_type="MARKET", quantity=stream.lot)
    sold = stream.wait_for(sell_id, "complete")
    return {"bought": bought, "held": held, "sold": sold, "after": stream.position()}


def test_a_resting_limit_order_is_streamed_as_open(cancelled_limit, stream):
    assert cancelled_limit["opened"], "no open update arrived for the LIMIT order"
    update = cancelled_limit["opened"][0]
    assert update["symbol"] == stream.symbol
    assert update["action"] == "BUY"
    assert update["pricetype"] == "LIMIT"
    assert update["mode"] == "analyze"


def test_orderstatus_reports_the_resting_order_as_open(cancelled_limit):
    status = cancelled_limit["status_open"]
    assert status["mode"] == "analyze"
    assert status["data"]["order_status"] == "open"


def test_a_cancel_by_id_is_streamed_as_cancelled(cancelled_limit):
    assert cancelled_limit["cancel"]["mode"] == "analyze"
    assert cancelled_limit["cancel"]["status"] == "success"
    assert cancelled_limit["cancelled"], "no cancelled update arrived"
    assert cancelled_limit["status_cancelled"]["data"]["order_status"] == "cancelled"


def test_a_market_buy_fills_one_lot(round_trip, stream):
    assert round_trip["bought"], "no complete update arrived for the MARKET buy"
    fill = round_trip["bought"][-1]
    assert fill["filled_quantity"] == stream.lot
    assert fill["average_price"] > 0
    assert round_trip["held"] == stream.lot


def test_the_offsetting_sell_leaves_no_position(round_trip, stream):
    assert round_trip["sold"], "no complete update arrived for the MARKET sell"
    assert round_trip["sold"][-1]["filled_quantity"] == stream.lot
    assert round_trip["after"] == 0


def test_get_orders_keeps_every_streamed_update(round_trip, cancelled_limit, stream):
    kept = {update["orderid"] for update in stream.feed.get_orders()["orders"]}
    assert set(stream.placed) <= kept
