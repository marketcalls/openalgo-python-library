"""The fake broker behind the SDK's order calls: how MARKET, LIMIT, SL-M and SL orders
work against the price path the test pushes (the sandbox's rules,
sandbox/execution_engine.py), part fills and rejections on request, the D31 rule that NFO
and BFO options refuse SL-M at a live broker, semi-auto mode, and the live and sandbox
books, each in the server's shape (services/orderstatus_service.py, orderbook_service.py,
tradebook_service.py, positionbook_service.py; sandbox/order_manager.py,
sandbox/position_manager.py).
"""

from __future__ import annotations

import json

import pytest
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._types import Mode

STRATEGY = "ema_cross"
OPTION = "NIFTY26NOV2624000CE"


def place(client, **changes):
    """Place an order through the SDK: 10 SBIN on NSE, MIS, at market unless changed."""
    fields = {
        "strategy": STRATEGY,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "price_type": "MARKET",
        "product": "MIS",
        "quantity": 10,
    }
    return client.placeorder(**(fields | changes))


def modify(client, order_id, **changes):
    """Modify a 10 SBIN BUY LIMIT order through the SDK, changing what is given."""
    fields = {
        "order_id": order_id,
        "strategy": STRATEGY,
        "symbol": "SBIN",
        "action": "BUY",
        "exchange": "NSE",
        "price_type": "LIMIT",
        "product": "MIS",
        "quantity": 10,
        "price": 800.0,
    }
    return client.modifyorder(**(fields | changes))


def status_of(client, order_id):
    """The `data` of the order's `orderstatus` answer."""
    answer = client.orderstatus(order_id=order_id, strategy=STRATEGY)
    assert answer["status"] == "success", answer
    return answer["data"]


def http_body(answer):
    """The server's JSON body inside the SDK's "HTTP <code>: <body>" message."""
    return json.loads(answer["message"].partition(": ")[2])


@pytest.fixture
def broker(fake_openalgo):
    """The fake in live mode, SBIN known and last traded at 812.0."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


@pytest.fixture
def sandbox(make_fake_openalgo):
    """The fake in sandbox mode, SBIN known and last traded at 812.0."""
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    return fake


# -- price rules ----------------------------------------------------------------


def test_a_market_order_fills_at_once_at_the_last_price(broker):
    client = broker.client()

    answer = place(client)

    assert answer == {"status": "success", "orderid": "261116000000001"}
    data = status_of(client, answer["orderid"])
    assert (data["order_status"], data["average_price"]) == ("complete", 812.0)


def test_a_fill_is_stamped_when_it_happens_not_when_the_price_last_traded(broker):
    client = broker.client()
    broker.clock.advance(5)

    place(client)

    [trade] = client.tradebook()["data"]
    assert trade["timestamp"] == "2026-11-16 10:15:05"


def test_a_market_order_with_no_price_yet_fills_at_the_first_trade(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    client = fake_openalgo.client()
    order_id = place(client)["orderid"]
    assert status_of(client, order_id)["order_status"] == "open"

    fake_openalgo.trade("SBIN", "NSE", 811.0)

    assert status_of(client, order_id)["average_price"] == 811.0


def test_the_sandbox_refuses_a_market_order_with_no_price(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")

    answer = place(fake.client())

    assert answer["code"] == 400
    assert http_body(answer) == {
        "status": "error",
        "message": "Cannot place MARKET order for SBIN - unable to fetch current price. "
        "Please try again later or use LIMIT order with a specific price.",
        "mode": "analyze",
    }


def test_a_limit_order_rests_until_the_price_reaches_it_then_fills_at_its_limit(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]

    broker.trade("SBIN", "NSE", 805.0)
    assert status_of(client, order_id)["order_status"] == "open"
    broker.trade("SBIN", "NSE", 799.5)

    data = status_of(client, order_id)
    assert (data["order_status"], data["average_price"]) == ("complete", 800.0)


def test_a_resting_live_stop_shows_open_in_the_book_and_orderstatus(broker):
    # broker/zerodha/mapping/order_data.py:194-195 rewrites "trigger pending" as open.
    client = broker.client()
    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    broker.trade("SBIN", "NSE", 801.0)

    assert status_of(client, order_id)["order_status"] == "open"
    assert client.orderbook()["data"]["orders"][0]["order_status"] == "open"
    assert broker.orders()[0].status == "trigger pending"


def test_a_resting_sandbox_stop_shows_trigger_pending(sandbox):
    # sandbox/order_manager.py:694-696 keeps the stop-loss book's word.
    client = sandbox.client()

    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    assert status_of(client, order_id)["order_status"] == "trigger pending"


def test_an_sl_m_stop_fills_at_the_last_price_once_triggered(broker):
    client = broker.client()
    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    broker.trade("SBIN", "NSE", 797.0)

    data = status_of(client, order_id)
    assert (data["order_status"], data["average_price"]) == ("complete", 797.0)


def test_a_live_sl_m_stop_fills_at_the_best_bid_given_with_its_trigger(broker):
    client = broker.client()
    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    broker.trade("SBIN", "NSE", 797.0, bid=796.5, ask=797.5)

    assert status_of(client, order_id)["average_price"] == 796.5


def test_a_stop_limit_the_price_runs_through_stays_open_unfilled(broker):
    client = broker.client()
    order = place(client, action="SELL", price_type="SL", trigger_price=800.0, price=799.0)

    broker.trade("SBIN", "NSE", 797.0)

    assert status_of(client, order["orderid"])["order_status"] == "open"
    assert broker.fills() == []


def test_a_stop_limit_run_through_fills_when_the_price_comes_back_within_its_limit(broker):
    client = broker.client()
    order = place(client, action="SELL", price_type="SL", trigger_price=800.0, price=799.0)
    broker.trade("SBIN", "NSE", 797.0)

    broker.trade("SBIN", "NSE", 799.5)

    data = status_of(client, order["orderid"])
    assert (data["order_status"], data["average_price"]) == ("complete", 799.5)


def test_a_stop_whose_trigger_the_price_already_passed_fills_at_once(broker):
    client = broker.client()

    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=820.0)["orderid"]

    data = status_of(client, order_id)
    assert (data["order_status"], data["average_price"]) == ("complete", 812.0)


def test_a_stop_limit_modified_to_a_marketable_limit_fills_at_the_last_price(broker):
    client = broker.client()
    order = place(client, action="SELL", price_type="SL", trigger_price=800.0, price=799.0)
    broker.trade("SBIN", "NSE", 797.0)

    answer = client.modifyorder(
        order_id=order["orderid"], strategy=STRATEGY, symbol="SBIN", action="SELL",
        exchange="NSE", price_type="SL", product="MIS", quantity=10, price=790.0,
        trigger_price=800.0,
    )  # fmt: skip

    assert answer == {"status": "success", "orderid": order["orderid"]}
    assert status_of(client, order["orderid"])["average_price"] == 797.0


def cancel_refusal(order_id, status=400, phrase="Bad Request"):
    """The body of a live cancel the broker refused (broker/zerodha/api/order_api.py:520,
    536): the HTTP client's error, without the line naming its documentation."""
    url = f"https://broker.invalid/orders/regular/{order_id}"
    words = f"Failed to cancel order: Client error '{status} {phrase}' for url '{url}'"
    return {"status": "error", "message": words}


def test_a_live_cancel_of_a_cancelled_order_is_the_brokers_500(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]

    first = client.cancelorder(order_id=order_id, strategy=STRATEGY)
    again = client.cancelorder(order_id=order_id, strategy=STRATEGY)

    assert first == {"status": "success", "orderid": order_id}
    assert status_of(client, order_id)["order_status"] == "cancelled"
    assert again["code"] == 500
    assert http_body(again) == cancel_refusal(order_id)


def test_a_live_cancel_that_a_fill_beat_is_the_brokers_500(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]
    broker.trade("SBIN", "NSE", 799.0)

    answer = client.cancelorder(order_id=order_id, strategy=STRATEGY)

    assert (answer["code"], http_body(answer)) == (500, cancel_refusal(order_id))
    assert status_of(client, order_id)["order_status"] == "complete"


def test_a_live_cancel_of_an_unknown_order_is_the_brokers_500(broker):
    answer = broker.client().cancelorder(order_id="999", strategy=STRATEGY)

    assert (answer["code"], http_body(answer)) == (500, cancel_refusal("999"))


def test_a_live_cancel_with_an_expired_login_is_the_brokers_500(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]
    broker.faults.expire_session()

    answer = client.cancelorder(order_id=order_id, strategy=STRATEGY)

    assert http_body(answer) == cancel_refusal(order_id, 403, "Forbidden")
    assert broker.orders()[0].status == "open"


def test_a_cancelled_sandbox_order_keeps_its_pending_quantity(sandbox):
    # sandbox/order_manager.py:1139-1142 changes the status only.
    client = sandbox.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]

    client.cancelorder(order_id=order_id, strategy=STRATEGY)

    data = status_of(client, order_id)
    assert (data["order_status"], data["filled_quantity"], data["pending_quantity"]) == (
        "cancelled",
        0,
        10,
    )


# -- fill prices -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "price", "expected"),
    [("BUY", 1500.0, 812.1), ("SELL", 100.0, 811.9)],
    ids=["buy at the ask", "sell at the bid"],
)
def test_a_marketable_live_limit_fills_at_the_market_not_its_limit(broker, action, price, expected):
    broker.trade("SBIN", "NSE", 812.0, bid=811.9, ask=812.1)

    place(broker.client(), action=action, price_type="LIMIT", price=price)

    assert [fill.price for fill in broker.fills()] == [expected]


def test_a_marketable_live_limit_fills_at_the_last_price_with_no_quote_given(broker):
    place(broker.client(), price_type="LIMIT", price=1500.0)

    assert [fill.price for fill in broker.fills()] == [812.0]


def test_a_marketable_sandbox_limit_fills_at_the_last_price(sandbox):
    # sandbox/order_manager.py:769-773 fills a marketable limit at the quote's LTP.
    sandbox.trade("SBIN", "NSE", 812.0, bid=811.9, ask=812.1)

    place(sandbox.client(), price_type="LIMIT", price=1500.0)

    assert [fill.price for fill in sandbox.fills()] == [812.0]


@pytest.mark.parametrize("analyzer", [False, True], ids=["live", "sandbox"])
@pytest.mark.parametrize(
    ("action", "expected"), [("BUY", 813.0), ("SELL", 811.0)], ids=["buy", "sell"]
)
def test_a_market_order_takes_the_ask_or_the_bid_given(
    make_fake_openalgo, analyzer, action, expected
):
    # sandbox/execution_engine.py:520-529: a BUY at the ask, a SELL at the bid.
    fake = make_fake_openalgo(analyzer=analyzer)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0, bid=811.0, ask=813.0)

    place(fake.client(), action=action)

    assert [fill.price for fill in fake.fills()] == [expected]


# -- part fills and rejections on request --------------------------------------------


def test_a_split_fill_leaves_the_rest_working_until_the_next_price(broker):
    client = broker.client()
    broker.split_next_fill(4)
    order_id = place(client)["orderid"]
    assert status_of(client, order_id)["order_status"] == "open"

    broker.trade("SBIN", "NSE", 813.0)

    trades = client.tradebook()["data"]
    assert [(t["quantity"], t["average_price"]) for t in trades] == [(4, 812.0), (6, 813.0)]
    assert status_of(client, order_id)["order_status"] == "complete"


def test_orderstatus_takes_a_live_orders_average_from_its_first_trade(broker):
    client = broker.client()
    broker.split_next_fill(4)
    order_id = place(client)["orderid"]

    broker.trade("SBIN", "NSE", 813.0)

    assert status_of(client, order_id)["average_price"] == 812.0


def test_fill_on_request_fills_part_of_a_resting_order_at_the_price_given(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]

    broker.fill(order_id, 3, price=799.0)

    assert [(t["orderid"], t["quantity"]) for t in client.tradebook()["data"]] == [(order_id, 3)]
    assert broker.position("SBIN", "NSE") == 3


def test_the_sandbox_fills_whole_orders_only(sandbox):
    with pytest.raises(ValueError, match="whole orders"):
        sandbox.split_next_fill(4)


def test_the_broker_rejects_the_next_order_right_after_accepting_it(broker):
    client = broker.client()
    broker.reject_next("RMS: margin exceeds")

    answer = place(client)

    assert answer["status"] == "success"
    assert status_of(client, answer["orderid"])["order_status"] == "rejected"
    assert broker.position("SBIN", "NSE") == 0


def test_the_broker_refuses_the_next_stop_at_placement(broker):
    client = broker.client()
    broker.reject_next("Trigger price too close", price_type="SL-M", at_placement=True)

    entry = place(client)
    stop = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)

    assert entry["status"] == "success"
    assert stop["code"] == 400
    assert http_body(stop) == {"status": "error", "message": "Trigger price too close"}


def test_the_sandbox_never_rejects_an_order_it_accepted(sandbox):
    with pytest.raises(ValueError, match="only at placement"):
        sandbox.reject_next("RMS: margin exceeds")


def test_a_sandbox_refusal_at_placement_keeps_the_rejected_order(sandbox):
    # sandbox/order_manager.py:632-680: the order is kept rejected, its ID in the answer.
    client = sandbox.client()
    sandbox.reject_next("Insufficient holdings", at_placement=True)

    answer = place(client, action="SELL")

    [order] = sandbox.orders()
    assert answer["code"] == 400
    assert http_body(answer) == {
        "status": "error",
        "orderid": order.order_id,
        "message": "Insufficient holdings",
        "mode": "analyze",
    }
    row = status_of(client, order.order_id)
    assert (row["order_status"], row["pending_quantity"]) == ("rejected", 0)


def test_an_option_refuses_an_sl_m_order_at_a_live_broker(broker):
    broker.add_instrument(OPTION, "NFO", lot_size=75, instrument_type="CE")
    client = broker.client()

    answer = place(
        client, symbol=OPTION, exchange="NFO", action="SELL", quantity=75,
        price_type="SL-M", trigger_price=90.0,
    )  # fmt: skip

    assert answer["code"] == 400
    assert "SL-M orders are not allowed for NFO options" in http_body(answer)["message"]
    assert broker.orders() == []


def test_an_option_takes_a_stop_limit_order_at_a_live_broker(broker):
    broker.add_instrument(OPTION, "NFO", lot_size=75, instrument_type="CE")
    client = broker.client()

    answer = place(
        client, symbol=OPTION, exchange="NFO", action="SELL", quantity=75,
        price_type="SL", trigger_price=90.0, price=85.5,
    )  # fmt: skip

    assert answer["status"] == "success"
    [order] = broker.orders()
    assert (order.price_type, order.status) == ("SL", "trigger pending")


def test_the_sandbox_accepts_an_sl_m_order_on_an_option(sandbox):
    sandbox.add_instrument(OPTION, "NFO", lot_size=75, instrument_type="CE")

    answer = place(
        sandbox.client(), symbol=OPTION, exchange="NFO", action="SELL", quantity=75,
        price_type="SL-M", trigger_price=90.0,
    )  # fmt: skip

    assert (answer["status"], answer["mode"]) == ("success", "analyze")


def test_the_sandbox_refuses_a_quantity_that_is_not_whole_lots(sandbox):
    sandbox.add_instrument(OPTION, "NFO", lot_size=75, instrument_type="CE")

    answer = place(sandbox.client(), symbol=OPTION, exchange="NFO", quantity=70)

    assert http_body(answer)["message"] == "Quantity must be in multiples of lot size 75"


def test_a_live_broker_refuses_a_quantity_above_the_freeze_quantity(broker):
    broker.add_instrument(OPTION, "NFO", lot_size=75, freeze_qty=1800, instrument_type="CE")

    answer = place(broker.client(), symbol=OPTION, exchange="NFO", quantity=1875)

    assert http_body(answer)["message"] == "Quantity 1875 is above the freeze quantity 1800"


def test_the_rest_layer_refuses_a_quantity_of_zero_in_the_schemas_words(broker):
    answer = place(broker.client(), quantity=0)

    assert answer["code"] == 400
    assert http_body(answer) == {
        "status": "error",
        "message": "{'quantity': ['Quantity must be a positive number.']}",
    }


# -- semi-auto mode ------------------------------------------------------------------


def test_semi_auto_queues_a_placement_for_the_action_center(broker):
    broker.set_semi_auto(True)

    answer = place(broker.client())

    assert answer == {
        "status": "success",
        "message": "Order queued for approval in Action Center",
        "mode": "semi_auto",
        "pending_order_id": 1,
    }
    assert broker.orders() == []


def test_semi_auto_checks_the_schema_before_queueing(broker):
    # restx_api/place_order.py:31-35 loads the schema before the service queues.
    broker.set_semi_auto(True)

    answer = place(broker.client(), quantity=0)

    assert answer["code"] == 400
    assert http_body(answer) == {
        "status": "error",
        "message": "{'quantity': ['Quantity must be a positive number.']}",
    }


def test_semi_auto_refuses_a_cancel_in_live_mode(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]
    broker.set_semi_auto(True)

    answer = client.cancelorder(order_id=order_id, strategy=STRATEGY)

    assert answer["code"] == 403
    assert http_body(answer)["message"].startswith(
        "Cancel order operation is not allowed in Semi-Auto mode."
    )


def test_semi_auto_refuses_a_modify_in_live_mode(broker):
    client = broker.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]
    broker.set_semi_auto(True)

    answer = modify(client, order_id, price=801.0)

    assert answer["code"] == 403
    assert http_body(answer)["message"].startswith(
        "Modify order operation is not allowed in Semi-Auto mode."
    )


def test_semi_auto_lets_a_cancel_through_in_sandbox_mode(sandbox):
    client = sandbox.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]
    sandbox.set_semi_auto(True)

    answer = client.cancelorder(order_id=order_id, strategy=STRATEGY)

    assert answer == {
        "status": "success",
        "orderid": order_id,
        "message": "Order cancelled successfully",
        "mode": "analyze",
    }


# -- the live and sandbox books -------------------------------------------------------


def test_sandbox_orders_take_the_sandboxs_ids_and_mode(sandbox):
    answer = place(sandbox.client())

    # 10:15:00.000 IST is millisecond 36,900,000 of the IST day.
    assert answer == {"status": "success", "orderid": "26111636900000", "mode": "analyze"}


def test_a_sandbox_modify_answers_in_the_sandboxs_words(sandbox):
    client = sandbox.client()
    order_id = place(client, price_type="LIMIT", price=800.0)["orderid"]

    answer = modify(client, order_id, price=801.0)

    assert answer == {
        "status": "success",
        "orderid": order_id,
        "message": "Order modified successfully",
        "mode": "analyze",
    }
    assert sandbox.orders()[0].price == 801.0


def test_the_sandbox_refuses_a_price_on_an_sl_m_stop(sandbox):
    client = sandbox.client()
    order_id = place(client, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    answer = modify(client, order_id, action="SELL", price_type="SL-M", trigger_price=799.0)

    assert answer["code"] == 400
    assert http_body(answer) == {
        "status": "error",
        "message": "SL-M orders do not accept a price",
        "mode": "analyze",
    }


def test_the_sandbox_position_book_carries_its_totals_and_mode(sandbox):
    client = sandbox.client()
    place(client)
    sandbox.trade("SBIN", "NSE", 815.0)

    answer = client.positionbook()

    assert (answer["mode"], answer["total_unrealized_pnl"], answer["total_pnl"]) == (
        "analyze",
        30.0,
        30.0,
    )
    [row] = answer["data"]
    assert (row["quantity"], row["average_price"], row["ltp"], row["pnl"]) == (
        10,
        812.0,
        815.0,
        30.0,
    )


def test_a_live_order_book_row_has_the_brokers_fields_only(broker):
    client = broker.client()
    place(client, price_type="LIMIT", price=800.0)

    answer = client.orderbook()

    [row] = answer["data"]["orders"]
    assert sorted(row) == sorted(
        ["symbol", "exchange", "action", "quantity", "price", "trigger_price", "pricetype",
         "product", "orderid", "order_status", "timestamp"]
    )  # fmt: skip
    assert row["timestamp"] == "2026-11-16 10:15:00"
    assert answer["data"]["statistics"]["total_open_orders"] == 1


def test_a_sandbox_order_status_has_the_sandboxs_fields(sandbox):
    client = sandbox.client()
    order_id = place(client)["orderid"]

    answer = client.orderstatus(order_id=order_id, strategy=STRATEGY)

    assert answer["mode"] == "analyze"
    assert answer["data"] == {
        "orderid": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "price": 0.0,
        "trigger_price": 0.0,
        "price_type": "MARKET",
        "product": "MIS",
        "order_status": "complete",
        "average_price": 812.0,
        "filled_quantity": 10,
        "pending_quantity": 0,
        "timestamp": "2026-11-16 10:15:00",
        "strategy": STRATEGY,
    }


def test_the_sandbox_lists_its_books_newest_first(sandbox):
    client = sandbox.client()
    first = place(client)["orderid"]
    sandbox.clock.advance(1)
    second = place(client, action="SELL")["orderid"]

    orders = client.orderbook()["data"]["orders"]
    trades = client.tradebook()["data"]

    assert [row["orderid"] for row in orders] == [second, first]
    assert [row["orderid"] for row in trades] == [second, first]


def test_a_position_nets_the_fills_and_counts_realized_and_open_pnl(broker):
    client = broker.client()
    place(client)
    broker.trade("SBIN", "NSE", 820.0)
    place(client, action="SELL", quantity=4)

    [row] = client.positionbook()["data"]

    # 4 sold 8 rupees above the 812.00 average, and 6 held 8 rupees up.
    assert row == {
        "symbol": "SBIN",
        "exchange": "NSE",
        "product": "MIS",
        "quantity": 6,
        "pnl": 80.0,
        "average_price": "812.00",
        "ltp": 820.0,
    }


def test_the_two_books_never_share_an_order(broker):
    client = broker.client()
    order_id = place(client)["orderid"]

    broker.set_analyzer(True)
    answer = client.orderstatus(order_id=order_id, strategy=STRATEGY)

    assert answer["code"] == 404
    assert http_body(answer) == {
        "status": "error",
        "message": f"Order {order_id} not found",
        "mode": "analyze",
    }
    assert client.positionbook()["data"] == []


def test_a_brokers_stop_keeps_working_while_the_switch_is_on(broker):
    order_id = place(broker.client(), action="SELL", price_type="SL-M", trigger_price=800.0)[
        "orderid"
    ]
    broker.set_analyzer(True)

    broker.trade("SBIN", "NSE", 799.0)

    [order] = broker.orders(Mode.LIVE)
    assert (order.order_id, order.status, order.average_price) == (order_id, "complete", 799.0)


def test_the_sandbox_does_not_work_its_orders_while_the_switch_is_off(sandbox):
    order_id = place(sandbox.client(), price_type="LIMIT", price=800.0)["orderid"]
    sandbox.set_analyzer(False)

    sandbox.trade("SBIN", "NSE", 799.0)

    [order] = sandbox.orders(Mode.SANDBOX)
    assert (order.order_id, order.status) == (order_id, "open")
