"""The fake OpenAlgo's answers read by the live layer's own parsers, as the fault suite
will feed them: the mode guard's `switch_mode`, the order path's `outcome_of` (what a
placement's answer says became of it) and `normalize_update` (a stream message or an
`orderstatus` answer as an `OrderUpdate`).
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._answers import server_answer
from openalgo.backtest._live._guard_answers import switch_mode
from openalgo.backtest._live._order_answers import Outcome, outcome_of
from openalgo.backtest._live._order_updates import normalize_update
from openalgo.backtest._live._types import Mode, OrderStatus

STRATEGY = "ema_cross"


def buy(client, **changes):
    fields = {
        "strategy": STRATEGY, "symbol": "SBIN", "action": "BUY", "exchange": "NSE",
        "price_type": "MARKET", "product": "MIS", "quantity": 10,
    }  # fmt: skip
    return client.placeorder(**(fields | changes))


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


def test_the_guard_reads_the_switch_from_analyzerstatus(broker):
    client = broker.client()
    before = switch_mode(client.analyzerstatus())

    broker.set_analyzer(True)

    assert (before, switch_mode(client.analyzerstatus())) == (Mode.LIVE, Mode.SANDBOX)


@pytest.mark.parametrize(
    ("schedule", "outcome"),
    [
        (lambda fake: None, Outcome.ACCEPTED),
        (lambda fake: fake.faults.timeout("placeorder"), Outcome.UNANSWERED),
        (lambda fake: fake.faults.bad_gateway_page("placeorder"), Outcome.UNANSWERED),
        (lambda fake: fake.faults.gateway_error("placeorder"), Outcome.UNANSWERED),
        (lambda fake: fake.faults.drop_connection("placeorder"), Outcome.UNANSWERED),
        (lambda fake: fake.faults.lose_answer("placeorder"), Outcome.UNANSWERED),
        (lambda fake: fake.faults.rate_limit("placeorder"), Outcome.REJECTED),
        (lambda fake: fake.reject_next("RMS", at_placement=True), Outcome.REJECTED),
        (lambda fake: fake.set_semi_auto(True), Outcome.QUEUED),
    ],
    ids=[
        "accepted",
        "timeout",
        "502 page",
        "gateway json",
        "dropped",
        "lost answer",
        "rate limited",
        "refused",
        "semi-auto",
    ],
)
def test_the_order_path_reads_what_became_of_a_placement(broker, schedule, outcome):
    schedule(broker)

    answer = buy(broker.client())

    assert outcome_of(server_answer(answer)).kind is outcome


def test_the_order_path_reads_the_live_streams_updates(broker):
    client = broker.client()
    client.connect()
    updates = []
    client.subscribe_orders(updates.append)

    buy(client, action="SELL", price_type="SL-M", trigger_price=800.0)
    broker.trade("SBIN", "NSE", 799.0)

    read = [normalize_update(u, ts_recv_ns=broker.clock.now(), source="stream") for u in updates]
    assert [(u.status, u.filled_quantity, u.average_price, u.mode) for u in read] == [
        (OrderStatus.TRIGGER_PENDING, 0, 0.0, Mode.LIVE),
        (OrderStatus.COMPLETE, 10, 799.0, Mode.LIVE),
    ]


def test_the_order_path_reads_the_sandboxs_updates(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    client = fake.client()
    client.connect()
    updates = []
    client.subscribe_orders(updates.append)

    buy(client)

    read = [normalize_update(u, ts_recv_ns=fake.clock.now(), source="stream") for u in updates]
    assert [(u.status, u.filled_quantity, u.mode) for u in read] == [
        (OrderStatus.OPEN, 0, Mode.SANDBOX),
        (OrderStatus.COMPLETE, 10, Mode.SANDBOX),
    ]


def test_the_order_path_reads_a_live_orderstatus_answer(broker):
    client = broker.client()
    order_id = buy(client)["orderid"]

    answer = client.orderstatus(order_id=order_id, strategy=STRATEGY)

    update = normalize_update(answer, ts_recv_ns=broker.clock.now(), source="poll")
    assert (update.status, update.filled_quantity, update.average_price) == (
        OrderStatus.COMPLETE,
        10,
        812.0,
    )
