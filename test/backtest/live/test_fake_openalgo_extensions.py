"""The fake OpenAlgo's extensions for the fault suite (16-session-design 7.3, unit 14): the
broker's order stream down while its login is expired or revoked, the position book that
lags its fills (`lag_positionbook`, `catch_up_positionbook`), and the name of the client
each call and each order came from (`FakeOpenAlgo.client`), which the session invariants
use to tell the session's orders and reads from another client's.

The other extensions of 7.3 have their own tests: the direct feed and `refuse_feed` in
`test_fake_openalgo_direct_feed.py`, the flavour whose last trade time is the trade's
exchange time in milliseconds in `test_fake_openalgo_feed.py`
(`test_another_brokers_tick_carries_the_trades_time_in_milliseconds`).
"""

from __future__ import annotations

import pytest
from fake_openalgo import client_name
from fake_openalgo.fixtures import *  # noqa: F403

STRATEGY = "ema_cross"
SBIN = {"exchange": "NSE", "symbol": "SBIN"}


def place(client, action="BUY", quantity=10, **changes):
    fields = {
        "strategy": STRATEGY, "symbol": "SBIN", "action": action, "exchange": "NSE",
        "price_type": "MARKET", "product": "MIS", "quantity": quantity,
    }  # fmt: skip
    return client.placeorder(**(fields | changes))


def resting_sell(client, price=820.0):
    """A SELL LIMIT of 10 SBIN at `price`, above the market: it rests until the price
    comes."""
    return place(client, "SELL", price_type="LIMIT", price=price)["orderid"]


@pytest.fixture
def broker(fake_openalgo):
    """The fake in live mode, SBIN known and last traded at 812.0."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


@pytest.fixture
def streamed(broker):
    """A client of the fake with its order stream subscribed into `updates`."""
    client = broker.client()
    assert client.connect() is True
    client.updates = []
    client.subscribe_orders(client.updates.append)
    return client


# -- the order stream while the broker's login is gone -----------------------------------

LOGIN_LOST = pytest.mark.parametrize(
    ("lose", "restore"),
    [("expire_session", "restore_session"), ("revoke_login", "restore_login")],
    ids=["expired", "revoked"],
)


@LOGIN_LOST
def test_a_fill_while_the_brokers_login_is_gone_sends_no_order_update(
    broker, streamed, lose, restore
):
    order_id = resting_sell(streamed)
    streamed.updates.clear()
    getattr(broker.faults, lose)()

    broker.trade("SBIN", "NSE", 820.0)

    assert streamed.updates == []
    lost = broker.log.of("order_update", "lost")
    assert [(e.detail["orderid"], e.detail["reason"]) for e in lost] == [
        (order_id, "broker login gone")
    ]
    assert broker.orders()[0].status == "complete"


@LOGIN_LOST
def test_an_update_lost_while_the_login_was_gone_stays_lost_once_it_is_back(
    broker, streamed, lose, restore
):
    resting_sell(streamed)
    getattr(broker.faults, lose)()
    broker.trade("SBIN", "NSE", 820.0)
    streamed.updates.clear()

    getattr(broker.faults, restore)()
    second = resting_sell(streamed, 830.0)

    assert [(u["orderid"], u["order_status"]) for u in streamed.updates] == [(second, "open")]


def test_a_sandbox_order_update_still_flows_while_the_login_is_expired(make_fake_openalgo):
    # The sandbox's updates are OpenAlgo's own events, not the broker's order stream.
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    client = fake.client()
    assert client.connect() is True
    updates = []
    client.subscribe_orders(updates.append)
    fake.faults.expire_session()

    order_id = resting_sell(client)

    assert [(u["orderid"], u["mode"]) for u in updates] == [(order_id, "analyze")]


# -- the position book lagging its fills -----------------------------------------------------


def positions(client) -> list[tuple[str, int]]:
    return [(row["symbol"], int(row["quantity"])) for row in client.positionbook()["data"]]


def test_a_lagging_position_book_answers_the_positions_from_before_the_latest_fills(broker):
    client = broker.client()
    place(client)
    broker.lag_positionbook()

    place(client, "SELL")

    assert positions(client) == [("SBIN", 10)]
    assert broker.position("SBIN", "NSE") == 0


def test_the_order_and_trade_books_show_the_fills_the_lagging_position_book_does_not(broker):
    client = broker.client()
    place(client)
    broker.lag_positionbook()

    place(client, "SELL")

    assert [row["order_status"] for row in client.orderbook()["data"]["orders"]] == [
        "complete",
        "complete",
    ]
    assert [row["action"] for row in client.tradebook()["data"]] == ["BUY", "SELL"]


def test_a_position_book_caught_up_shows_every_fill_again(broker):
    client = broker.client()
    place(client)
    broker.lag_positionbook()
    place(client, "SELL")

    broker.catch_up_positionbook()

    assert positions(client) == [("SBIN", 0)]


# -- the client each call and each order came from --------------------------------------------


def test_each_client_of_the_fake_has_its_own_name(broker):
    first, second = broker.client(), broker.client()
    assert client_name(first) != client_name(second)
    assert client_name(first).startswith("client-")


def test_an_order_keeps_the_name_of_the_client_that_placed_it(broker):
    session, other = broker.client(), broker.client()
    place(session)
    place(other, "SELL", 5)

    assert [order.client for order in broker.orders()] == [
        client_name(session),
        client_name(other),
    ]


def test_the_log_names_the_client_of_every_call(broker):
    session, other = broker.client(), broker.client()
    session.orderbook()
    other.funds()

    calls = [(call.what, call.detail["client"]) for call in broker.log.calls()]
    assert calls == [("orderbook", client_name(session)), ("funds", client_name(other))]
