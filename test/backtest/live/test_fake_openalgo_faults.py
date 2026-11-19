"""The REST faults a test schedules on the fake OpenAlgo, each as the SDK returns it to the
live layer (openalgo/orders.py and the other mixins, `_make_request`), whether the call
reached the broker, the expired broker login, and the log every fault is written to.
"""

from __future__ import annotations

import pytest
from fake_openalgo import USERNAME
from fake_openalgo.fixtures import *  # noqa: F403

STRATEGY = "ema_cross"
#: The SDK's own answers when no answer came back (openalgo/orders.py, _make_request).
SDK_TIMEOUT = {
    "status": "error",
    "message": "Request timed out. The server took too long to respond.",
    "error_type": "timeout_error",
}
SDK_DROPPED = {
    "status": "error",
    "message": "HTTP error occurred: Server disconnected without sending a response.",
    "error_type": "http_error",
}
SDK_REFUSED = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}
NS = 1_000_000_000
INVALID_KEY = '{"message":"Invalid openalgo apikey","status":"error"}\n'
UNANSWERED = (
    "The connection to the broker dropped after the order was sent, so it may have been "
    "placed. Check the order book before placing it again."
)
EXPIRED = "Incorrect `api_key` or `access_token`."


def buy(client, quantity=10):
    return client.placeorder(
        strategy=STRATEGY, symbol="SBIN", action="BUY", exchange="NSE",
        price_type="MARKET", product="MIS", quantity=quantity,
    )  # fmt: skip


def sdk_http_error(code, text):
    """A non-200 answer as the SDK's order and account calls return it."""
    return {"status": "error", "message": f"HTTP {code}: {text}", "code": code,
            "error_type": "http_error"}  # fmt: skip


@pytest.fixture
def broker(fake_openalgo):
    """The fake in live mode, SBIN known and last traded at 812.0."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


def test_a_timeout_is_the_sdks_timeout_answer_and_places_nothing(broker):
    broker.faults.timeout("placeorder")

    answer = buy(broker.client())

    assert answer == SDK_TIMEOUT
    assert broker.orders() == []


def test_a_timeout_costs_the_clients_read_timeout_on_the_clock(broker):
    before = broker.clock.now()
    broker.faults.timeout("orderbook")

    broker.client().orderbook()

    # The SDK waits its `timeout`, 120 seconds by default (openalgo/base.py).
    assert broker.clock.now() - before == 120 * NS


def test_a_slow_answer_past_the_read_timeout_is_a_timeout_after_the_work(broker):
    before = broker.clock.now()
    broker.faults.slow("placeorder", 600)

    answer = buy(broker.client())

    assert answer == SDK_TIMEOUT
    assert broker.clock.now() - before == 120 * NS
    [order] = broker.orders()
    assert order.status == "complete"


def test_a_refused_connection_is_the_sdks_connection_error_and_reaches_nothing(broker):
    broker.faults.refuse_connection("placeorder")

    answer = buy(broker.client())

    assert answer == SDK_REFUSED
    assert broker.orders() == []
    assert broker.log.faults() == ["refused_connection"]


def test_a_timeout_after_the_broker_took_the_order_leaves_it_placed(broker):
    broker.faults.timeout("placeorder", reached_broker=True)

    answer = buy(broker.client())

    assert answer == SDK_TIMEOUT
    [order] = broker.orders()
    assert order.status == "complete"


def test_a_502_page_comes_back_as_the_sdks_http_error_with_the_page(broker):
    broker.faults.bad_gateway_page("placeorder")

    answer = buy(broker.client())

    assert (answer["code"], answer["error_type"]) == (502, "http_error")
    assert answer["message"].startswith("HTTP 502: <html>")
    assert "502 Bad Gateway" in answer["message"]


def test_a_gateway_json_error_has_no_status_of_its_own(broker):
    broker.faults.gateway_error("orderbook")

    answer = broker.client().orderbook()

    assert answer == sdk_http_error(504, '{"message": "Endpoint request timed out"}')


def test_a_gateway_error_after_the_broker_took_the_order_leaves_it_placed(broker):
    broker.faults.gateway_error("placeorder", status=503, reached_broker=True)

    answer = buy(broker.client())

    assert answer["code"] == 503
    assert len(broker.orders()) == 1


def test_a_gateway_error_is_a_server_status(broker):
    with pytest.raises(ValueError, match="500 or more"):
        broker.faults.gateway_error("placeorder", status=404)


def test_a_dropped_connection_is_the_sdks_http_error_without_a_code(broker):
    broker.faults.drop_connection("placeorder")

    answer = buy(broker.client())

    assert answer == SDK_DROPPED
    assert broker.orders() == []


def test_a_lost_broker_answer_is_the_servers_502_and_the_order_is_placed(broker):
    broker.faults.lose_answer("placeorder")

    answer = buy(broker.client())

    assert answer == sdk_http_error(502, f'{{"message":"{UNANSWERED}","status":"error"}}\n')
    [order] = broker.orders()
    assert order.status == "complete"


def test_a_lost_answer_to_a_modify_is_the_servers_500_and_the_change_is_made(broker):
    client = broker.client()
    order_id = client.placeorder(
        strategy=STRATEGY, symbol="SBIN", action="BUY", exchange="NSE",
        price_type="LIMIT", product="MIS", quantity=10, price=800.0,
    )["orderid"]  # fmt: skip
    broker.faults.lose_answer("modifyorder")

    answer = client.modifyorder(
        order_id=order_id, strategy=STRATEGY, symbol="SBIN", action="BUY", exchange="NSE",
        price_type="LIMIT", product="MIS", quantity=10, price=805.0,
    )  # fmt: skip

    assert answer["code"] == 500
    assert "Failed to modify order due to internal error" in answer["message"]
    assert broker.orders()[0].price == 805.0


def test_only_an_order_write_can_lose_the_brokers_answer(broker):
    with pytest.raises(ValueError, match="lose a broker's answer"):
        broker.faults.lose_answer("orderbook")


def test_the_sandbox_has_no_brokers_answer_to_lose(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)

    with pytest.raises(ValueError, match="sends nothing to a broker"):
        fake.faults.lose_answer("placeorder")


def test_a_slow_answer_moves_the_clock_on_before_answering(broker):
    before = broker.clock.now()
    broker.faults.slow("orderstatus", 4.5)
    client = broker.client()
    order_id = buy(client)["orderid"]

    answer = client.orderstatus(order_id=order_id, strategy=STRATEGY)

    assert answer["status"] == "success"
    assert broker.clock.now() - before == 4_500_000_000


def test_the_rate_limiters_429_is_the_rest_layers_own_body(broker):
    broker.faults.rate_limit("placeorder")

    answer = buy(broker.client())

    assert answer == sdk_http_error(429, '{"message": "10 per 1 second"}\n')
    assert broker.orders() == []


def test_telegram_notifys_429_names_its_limit_per_minute(broker):
    broker.faults.rate_limit("telegram/notify")

    answer = broker.client().telegram(username=USERNAME, message="hello")

    assert answer == sdk_http_error(429, '{"message": "30 per 1 minute"}\n')
    assert broker.telegram_sent == []


def test_each_fault_acts_once_in_the_order_scheduled(broker):
    client = broker.client()
    broker.faults.timeout("placeorder")
    broker.faults.drop_connection("placeorder")

    answers = [buy(client), buy(client), buy(client)]

    assert [a.get("error_type") for a in answers] == ["timeout_error", "http_error", None]
    assert len(broker.orders()) == 1


def test_a_fault_on_one_endpoint_leaves_the_others_alone(broker):
    client = broker.client()
    broker.faults.timeout("placeorder")

    assert client.orderbook()["status"] == "success"
    assert buy(client) == SDK_TIMEOUT


def test_a_fault_on_an_endpoint_the_fake_does_not_answer_is_refused(broker):
    with pytest.raises(ValueError, match="no endpoint 'closeposition'"):
        broker.faults.timeout("closeposition")


def test_every_fault_is_in_the_log(broker):
    client = broker.client()
    broker.faults.timeout("placeorder")
    broker.faults.rate_limit("funds")
    buy(client)
    client.funds()

    assert broker.log.faults() == ["timeout", "rate_limited"]
    assert [entry.detail["endpoint"] for entry in broker.log.of("fault")] == [
        "placeorder",
        "funds",
    ]


# -- the broker's login expires --------------------------------------------------------


def test_an_expired_login_still_answers_ping(broker):
    broker.faults.expire_session()

    assert broker.client().ping()["status"] == "success"


def test_an_expired_login_answers_funds_with_success_and_no_data(broker):
    broker.faults.expire_session()

    assert broker.client().funds() == {"status": "success", "data": {}}


def test_an_expired_login_fails_quotes_in_the_brokers_words(broker):
    broker.faults.expire_session()

    answer = broker.client().quotes(symbol="SBIN", exchange="NSE")

    message = f"Error fetching quotes: API Error: {EXPIRED}"
    assert answer == sdk_http_error(500, f'{{"message":"{message}","status":"error"}}\n')


def test_an_expired_login_refuses_a_placement_in_the_brokers_words(broker):
    broker.faults.expire_session()

    answer = buy(broker.client())

    assert answer == sdk_http_error(403, f'{{"message":"{EXPIRED}","status":"error"}}\n')
    assert broker.orders() == []


@pytest.mark.parametrize(
    ("book", "path"),
    [("orderbook", "/orders"), ("tradebook", "/trades"), ("positionbook", "/portfolio/positions")],
)
def test_an_expired_login_fails_a_book_with_the_http_clients_error(broker, book, path):
    # broker/zerodha/api/order_api.py:76-91 re-raises; services/orderbook_service.py:182-184.
    broker.faults.expire_session()

    answer = getattr(broker.client(), book)()

    words = f"Client error '403 Forbidden' for url 'https://broker.invalid{path}'"
    assert answer == sdk_http_error(500, f'{{"message":"{words}","status":"error"}}\n')


@pytest.fixture
def sandbox(make_fake_openalgo):
    """The fake in sandbox mode, SBIN known and last traded at 812.0."""
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    return fake


def test_an_expired_login_refuses_a_sandbox_market_order_with_no_position(sandbox):
    # sandbox/order_manager.py:306, 338-347: no quote and no position to price it from.
    before = sandbox.clock.now()
    sandbox.faults.expire_session()

    answer = buy(sandbox.client())

    assert answer["code"] == 400
    assert "Cannot place MARKET order for SBIN - unable to fetch current price" in answer["message"]
    assert sandbox.orders() == []
    # Three tries for a quote, 0.3 and 0.6 seconds apart (order_manager.py:319-320).
    assert sandbox.clock.now() - before == 900_000_000


def test_an_expired_login_takes_a_sandbox_market_order_with_a_position_unfilled(sandbox):
    # sandbox/order_manager.py:323-333: priced from the position's last price, no fill.
    client = sandbox.client()
    buy(client)
    sandbox.faults.expire_session()

    answer = buy(client, quantity=5)

    assert (answer["status"], answer["mode"]) == ("success", "analyze")
    assert [order.status for order in sandbox.orders()] == ["complete", "open"]
    assert len(sandbox.fills()) == 1


def test_an_expired_login_stops_resting_sandbox_orders_filling_until_restored(sandbox):
    client = sandbox.client()
    order_id = client.placeorder(
        strategy=STRATEGY, symbol="SBIN", action="BUY", exchange="NSE",
        price_type="LIMIT", product="MIS", quantity=10, price=800.0,
    )["orderid"]  # fmt: skip
    sandbox.faults.expire_session()

    sandbox.trade("SBIN", "NSE", 799.0)
    assert sandbox.orders()[0].status == "open"
    sandbox.faults.restore_session()
    sandbox.trade("SBIN", "NSE", 799.0)

    [order] = sandbox.orders()
    assert (order.order_id, order.status, order.average_price) == (order_id, "complete", 800.0)


def test_an_expired_login_leaves_the_sandboxs_books_readable(sandbox):
    client = sandbox.client()
    buy(client)
    sandbox.faults.expire_session()

    answer = client.orderbook()

    assert (answer["status"], answer["mode"], len(answer["data"]["orders"])) == (
        "success",
        "analyze",
        1,
    )


def test_a_restored_login_answers_funds_again(broker):
    broker.faults.expire_session()
    broker.faults.restore_session()

    assert broker.client().funds()["data"]["availablecash"] == "250000.00"
    assert broker.log.faults() == ["session_expired", "session_restored"]


# -- OpenAlgo revokes the broker's login ---------------------------------------------------


@pytest.mark.parametrize("call", ["ping", "analyzerstatus", "funds", "orderbook", "intervals"])
def test_a_revoked_login_refuses_every_call_with_403(broker, call):
    # app.py:572-583 revokes the token; get_auth_token_broker then finds none.
    broker.faults.revoke_login()

    answer = getattr(broker.client(), call)()

    assert answer == sdk_http_error(403, INVALID_KEY)


def test_a_revoked_login_refuses_a_placement_and_places_nothing(broker):
    broker.faults.revoke_login()

    assert buy(broker.client())["code"] == 403
    assert broker.orders() == []


def test_a_revoked_login_leaves_telegram_notify_answering(broker):
    # restx_api/telegram_bot.py, SendNotification, checks the API key alone.
    broker.faults.revoke_login()

    answer = broker.client().telegram(username=USERNAME, message="login lost")

    assert answer["status"] == "success"


def test_a_restored_login_answers_ping_again(broker):
    broker.faults.revoke_login()
    broker.faults.restore_login()

    assert broker.client().ping()["status"] == "success"
    assert broker.log.faults() == ["login_revoked", "login_restored"]


# -- the server's rate limits, on the manual clock ------------------------------------------


def test_the_eleventh_call_of_a_route_within_a_second_is_refused_with_429(broker):
    # limiter.py's moving window; restx_api/orderbook.py: "10 per second".
    client = broker.client()

    answers = [client.orderbook() for _ in range(11)]

    assert [a["status"] for a in answers[:10]] == ["success"] * 10
    assert answers[10] == sdk_http_error(429, '{"message": "10 per 1 second"}\n')


def test_a_route_takes_calls_again_once_the_window_has_passed(broker):
    client = broker.client()
    for _ in range(10):
        client.orderbook()

    broker.clock.advance(1.001)

    assert client.orderbook()["status"] == "success"


def test_each_route_has_its_own_rate_limit(broker):
    client = broker.client()
    for _ in range(10):
        client.orderbook()

    assert client.tradebook()["status"] == "success"


def test_a_refused_call_does_not_count_against_the_window(broker):
    client = broker.client()
    for _ in range(10):
        client.orderbook()
    client.orderbook()
    broker.clock.advance(1.001)

    answers = [client.orderbook() for _ in range(10)]

    assert [a["status"] for a in answers] == ["success"] * 10


def test_telegram_notify_takes_thirty_calls_a_minute(broker):
    client = broker.client()

    answers = [client.telegram(username=USERNAME, message=f"m{i}") for i in range(31)]

    assert [a["status"] for a in answers[:30]] == ["success"] * 30
    assert answers[30] == sdk_http_error(429, '{"message": "30 per 1 minute"}\n')
