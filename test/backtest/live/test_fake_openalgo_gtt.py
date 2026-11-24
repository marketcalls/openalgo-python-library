"""The fake OpenAlgo's GTT orders (20-positional-live 13.2 PL9; the lead's ruling N3a-3):
SINGLE GTTs placed, modified and cancelled through the SDK's own calls, the refusals and
faults a test schedules, the GTT book in its two shapes (a broker that keeps the GTTs that
ended, one that lists the active ones only), the child order a GTT places when a trade
reaches its trigger, the sandbox's link to that child, and a day boundary that moves a CNC
position into the holdings.

The expected answers and rows are written out from the server's code (restx_api/schemas.py,
services/*_gtt_order_service.py, services/gtt_orderbook_service.py, sandbox/gtt_manager.py,
broker/zerodha/mapping/gtt_data.py), with prices and units worked by hand. The fake's day
is Monday 2026-11-16 from 10:15:00 IST. Nothing leaves the process.
"""

from __future__ import annotations

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._types import Mode

#: A long CNC leg's stop: sell 10 SBIN when it trades at 790.00 or below, the child's limit
#: 780.00; SBIN trades at 800.00 when it is placed.
STOP = {
    "strategy": "ema",
    "symbol": "SBIN",
    "action": "SELL",
    "exchange": "NSE",
    "product": "CNC",
    "quantity": 10,
    "price": 780.0,
    "triggerprice_sl": 790.0,
}
PRICE = 800.0


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    return fake_openalgo


@pytest.fixture
def sandbox(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", PRICE)
    return fake


def bought(fake, units: int = 10, product: str = "CNC", action: str = "BUY") -> None:
    """The account buys (or with `action` "SELL" sells) `units` SBIN at the market in
    `product`."""
    fake.client().placeorder(
        strategy="ema",
        symbol="SBIN",
        action=action,
        exchange="NSE",
        price_type="MARKET",
        product=product,
        quantity=units,
    )


def book(fake, status: str = "all") -> list[dict]:
    return fake.client().gttorderbook(status=status)["data"]


# ----- placing ----------------------------------------------------------------------------


def test_a_brokers_single_gtt_is_placed_and_listed_as_its_book_writes_it(broker):
    answer = broker.client().placegttorder(**STOP)

    assert answer == {"status": "success", "trigger_id": "261116000001"}
    (row,) = book(broker, "active")
    leg = {"action": "SELL", "quantity": 10, "price": 780.0, "pricetype": "LIMIT", "product": "CNC"}
    assert row == {
        "trigger_id": "261116000001",
        "trigger_type": "single",
        "status": "active",
        "symbol": "SBIN",
        "exchange": "NSE",
        "trigger_prices": [790.0],
        "last_price": 800.0,
        "legs": [leg],
        "created_at": "2026-11-16 10:15:00",
        "updated_at": "",
        "expires_at": "",
    }


def test_the_sandboxs_gtt_has_its_id_its_strategy_and_no_child_yet(sandbox):
    answer = sandbox.client().placegttorder(**STOP)

    assert answer == {"status": "success", "mode": "analyze", "trigger_id": "GTT-261116-00000001"}
    (row,) = book(sandbox)
    assert (row["strategy"], row["created_at"]) == ("ema", "2026-11-16T10:15:00")
    assert row["legs"][0]["triggered_order_id"] is None


def test_a_gtt_in_mis_is_refused_by_the_schema(broker):
    answer = broker.client().placegttorder(**(STOP | {"product": "MIS"}))

    words = "GTT supports only CNC (delivery) or NRML (overnight F&O); MIS is intraday-only."
    assert (answer["code"], answer["message"]) == (400, str({"product": [words]}))
    assert broker.gtts() == []


@pytest.mark.parametrize(
    "reason",
    ["Trigger price is too close to the last price", "GTT is not allowed for this segment"],
    ids=["a price refusal", "another refusal"],
)
def test_a_refusal_the_test_schedules_is_the_brokers_400_and_places_nothing(broker, reason):
    broker.refuse_next_gtt(reason)

    answer = broker.client().placegttorder(**STOP)

    assert answer == {"status": "error", "message": reason, "code": 400}
    assert broker.gtts() == []


def test_a_broker_without_gtt_orders_answers_a_place_with_the_servers_501(make_fake_openalgo):
    fake = make_fake_openalgo(gtt=False)
    fake.add_instrument("SBIN", "NSE")

    answer = fake.client().placegttorder(**STOP)

    words = f"GTT orders are not supported for broker '{BROKER}' yet"
    assert answer == {"status": "error", "message": words, "code": 501}


@pytest.mark.parametrize(("reached", "placed"), [(True, 1), (False, 0)], ids=["reached", "not"])
def test_a_place_that_times_out_is_at_the_broker_only_when_it_reached_it(broker, reached, placed):
    broker.faults.timeout("placegttorder", reached_broker=reached)

    answer = broker.client().placegttorder(**STOP)

    assert answer["error_type"] == "timeout_error"
    assert len(broker.gtts()) == placed


def test_a_place_whose_answer_is_lost_is_the_servers_500_and_the_gtt_is_made(broker):
    broker.faults.lose_answer("placegttorder")

    answer = broker.client().placegttorder(**STOP)

    assert (answer["code"], answer["message"]) == (500, "Failed to place GTT due to internal error")
    assert [gtt.status for gtt in broker.gtts()] == ["active"]


# ----- modifying and cancelling ---------------------------------------------------------------


def test_a_modify_replaces_the_trigger_the_limit_and_the_units(broker):
    client = broker.client()
    trigger_id = client.placegttorder(**STOP)["trigger_id"]

    new = STOP | {"triggerprice_sl": 795.0, "price": 785.0, "quantity": 6}
    answer = client.modifygttorder(trigger_id=trigger_id, **new)

    assert answer == {"status": "success", "trigger_id": trigger_id}
    (row,) = book(broker)
    assert (row["trigger_prices"], row["legs"][0]["quantity"], row["legs"][0]["price"]) == (
        [795.0],
        6,
        785.0,
    )


def test_a_modify_applied_in_part_changes_the_trigger_only_and_fails(broker):
    client = broker.client()
    trigger_id = client.placegttorder(**STOP)["trigger_id"]
    broker.half_apply_next_gtt_modify()

    answer = client.modifygttorder(
        trigger_id=trigger_id, **(STOP | {"triggerprice_sl": 795.0, "quantity": 6})
    )

    assert (answer["code"], answer["message"]) == (
        500,
        "Failed to modify GTT due to internal error",
    )
    (gtt,) = broker.gtts()
    assert (gtt.trigger, gtt.quantity) == (795.0, 10)


def test_the_sandbox_has_no_modify_applied_in_part(sandbox):
    with pytest.raises(ValueError, match="all at once"):
        sandbox.half_apply_next_gtt_modify()


def test_a_cancel_ends_the_gtt_cancelled(broker):
    client = broker.client()
    trigger_id = client.placegttorder(**STOP)["trigger_id"]

    answer = client.cancelgttorder(trigger_id=trigger_id, strategy="ema")

    assert answer == {"status": "success", "trigger_id": trigger_id}
    assert [gtt.status for gtt in broker.gtts()] == ["cancelled"]


@pytest.mark.parametrize(
    ("analyzer", "code", "words"),
    [
        (False, 400, "Trigger {trigger_id} is not active and cannot be changed"),
        (True, 404, "No active GTT with trigger_id '{trigger_id}'"),
    ],
    ids=["broker", "sandbox"],
)
def test_a_gtt_that_ended_cannot_be_modified_or_cancelled(
    make_fake_openalgo, analyzer, code, words
):
    fake = make_fake_openalgo(analyzer=analyzer)
    fake.add_instrument("SBIN", "NSE")
    client = fake.client()
    trigger_id = client.placegttorder(**STOP)["trigger_id"]
    client.cancelgttorder(trigger_id=trigger_id, strategy="ema")

    modified = client.modifygttorder(trigger_id=trigger_id, **STOP)
    cancelled = client.cancelgttorder(trigger_id=trigger_id, strategy="ema")

    expected = (code, words.format(trigger_id=trigger_id))
    assert (modified["code"], modified["message"]) == expected
    assert (cancelled["code"], cancelled["message"]) == expected


def test_a_gtt_the_user_ends_outside_the_session_shows_the_brokers_word(broker):
    trigger_id = broker.client().placegttorder(**STOP)["trigger_id"]

    broker.set_gtt_status(trigger_id, "deleted")

    assert [row["status"] for row in book(broker)] == ["deleted"]
    assert broker.orders() == []


# ----- the book's two shapes -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("history", "listed"),
    [(True, ["active", "cancelled"]), (False, ["active"])],
    ids=["keeps history", "active only"],
)
def test_a_brokers_book_lists_the_gtts_that_ended_only_when_it_keeps_them(
    make_fake_openalgo, history, listed
):
    fake = make_fake_openalgo(gtt_history=history)
    fake.add_instrument("SBIN", "NSE")
    client = fake.client()
    first = client.placegttorder(**STOP)["trigger_id"]
    client.placegttorder(**(STOP | {"triggerprice_sl": 785.0}))
    client.cancelgttorder(trigger_id=first, strategy="ema")

    assert [row["status"] for row in book(fake, "all")] == listed


def test_the_active_book_lists_the_active_gtts_newest_first(broker):
    client = broker.client()
    first = client.placegttorder(**STOP)["trigger_id"]
    broker.clock.advance(1)
    second = client.placegttorder(**(STOP | {"triggerprice_sl": 785.0}))["trigger_id"]

    assert [row["trigger_id"] for row in book(broker, "active")] == [second, first]


# ----- firing -----------------------------------------------------------------------------------


def test_a_trade_above_a_stops_trigger_fires_nothing(broker):
    broker.client().placegttorder(**STOP)

    broker.trade("SBIN", "NSE", 790.05)

    assert [gtt.status for gtt in broker.gtts()] == ["active"]


def test_a_trade_at_a_stops_trigger_fires_it_and_its_child_sells_the_position(broker):
    bought(broker)
    broker.client().placegttorder(**STOP)

    broker.trade("SBIN", "NSE", 790.0)

    assert [gtt.status for gtt in broker.gtts()] == ["triggered"]
    child = broker.orders()[-1]
    assert (child.action, child.price_type, child.price, child.quantity, child.product) == (
        "SELL",
        "LIMIT",
        780.0,
        10,
        "CNC",
    )
    # A marketable LIMIT arriving at a broker fills at the market: 790.00.
    assert (child.status, child.average_price) == ("complete", 790.0)
    assert broker.position("SBIN", "NSE") == 0


def test_a_shorts_stop_above_the_price_fires_on_a_rise_and_buys_the_short_back(broker):
    bought(broker, product="NRML", action="SELL")
    short_stop = STOP | {"action": "BUY", "triggerprice_sl": 0.0, "triggerprice_tg": 810.0}
    broker.client().placegttorder(**(short_stop | {"price": 820.0, "product": "NRML"}))

    broker.trade("SBIN", "NSE", 809.95)
    broker.trade("SBIN", "NSE", 810.0)

    assert [gtt.status for gtt in broker.gtts()] == ["triggered"]
    child = broker.orders()[-1]
    assert (child.action, child.price, child.average_price) == ("BUY", 820.0, 810.0)
    assert broker.position("SBIN", "NSE") == 0


def test_a_child_the_price_gapped_through_rests_unfilled(broker):
    bought(broker)
    broker.client().placegttorder(**STOP)

    broker.trade("SBIN", "NSE", 770.0)

    child = broker.orders()[-1]
    assert (child.status, child.filled) == ("open", 0)
    assert broker.position("SBIN", "NSE") == 10


@pytest.mark.parametrize(
    ("analyzer", "named"), [(False, False), (True, True)], ids=["broker", "sandbox"]
)
def test_only_the_sandboxs_book_names_the_child_of_a_gtt_that_fired(
    make_fake_openalgo, analyzer, named
):
    fake = make_fake_openalgo(analyzer=analyzer)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", PRICE)
    fake.client().placegttorder(**STOP)

    fake.trade("SBIN", "NSE", 790.0)

    (row,) = book(fake)
    child = fake.orders()[-1].order_id
    assert row["status"] == "triggered"
    assert ("triggered_order_id" in row["legs"][0], row["legs"][0].get("triggered_order_id")) == (
        named,
        child if named else None,
    )


def test_a_sandbox_gtt_waits_while_the_switch_is_off(sandbox):
    sandbox.client().placegttorder(**STOP)
    sandbox.set_analyzer(False)

    sandbox.trade("SBIN", "NSE", 790.0)

    assert [gtt.status for gtt in sandbox.gtts(Mode.SANDBOX)] == ["active"]


# ----- the day boundary ---------------------------------------------------------------------------


def test_a_day_boundary_moves_a_cnc_position_into_the_holdings(broker):
    broker.set_holding("SBIN", "NSE", 40, average_price=760.0)
    bought(broker)

    broker.end_day()

    # 40 held at 760.00 and 10 bought at 800.00: 50 at (40 * 760 + 10 * 800) / 50 = 768.00.
    (held,) = broker.client().holdings()["data"]["holdings"]
    assert (held["quantity"], held["average_price"]) == (50, 768.0)
    assert broker.client().positionbook()["data"] == []


def test_a_day_boundary_carries_an_nrml_position_and_clears_the_days_books(broker):
    bought(broker, product="NRML")

    broker.end_day()

    (row,) = broker.client().positionbook()["data"]
    assert (row["product"], row["quantity"], row["average_price"]) == ("NRML", 10, "800.00")
    assert (broker.orders(), broker.fills()) == ([], [])


def test_a_gtt_stays_across_a_day_boundary(broker):
    bought(broker)
    broker.client().placegttorder(**STOP)

    broker.end_day()

    assert [row["status"] for row in book(broker, "active")] == ["active"]


def test_a_day_boundary_with_an_mis_position_open_is_refused(broker):
    bought(broker, product="MIS")

    with pytest.raises(ValueError, match="squares MIS off"):
        broker.end_day()
