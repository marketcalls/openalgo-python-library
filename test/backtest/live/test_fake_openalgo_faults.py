"""The REST faults a test schedules on the fake OpenAlgo, each as the SDK returns it to the
live layer (openalgo/orders.py and the other mixins, `_make_request`), whether the call
reached the broker, the expired broker login, and the log every fault is written to.
"""

from __future__ import annotations

import pytest
from fake_openalgo import USERNAME
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._types import Mode

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


def test_an_expired_login_refuses_orders_and_fails_the_books(broker):
    broker.faults.expire_session()
    client = broker.client()

    assert buy(client)["code"] == 403
    assert client.orderbook()["code"] == 500
    assert broker.orders() == []


def test_an_expired_login_leaves_the_sandboxs_own_books_alone(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    fake.faults.expire_session()

    answer = buy(fake.client())

    assert (answer["status"], answer["mode"]) == ("success", "analyze")
    assert len(fake.orders(Mode.SANDBOX)) == 1


def test_a_restored_login_answers_funds_again(broker):
    broker.faults.expire_session()
    broker.faults.restore_session()

    assert broker.client().funds()["data"]["availablecash"] == "250000.00"
    assert broker.log.faults() == ["session_expired", "session_restored"]
