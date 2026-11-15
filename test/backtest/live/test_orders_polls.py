"""The polling fallback on a manual clock: every second while an order has no word or is
working but not resting, every 30 seconds for a resting order the stream has spoken for,
in rounds of at most one a second and three orders."""

from orders_fakes import (
    NS,
    STRATEGY,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
    status_answer,
    trades,
    update,
)

from openalgo.backtest._live._types import OrderStatus, Purpose

NOT_FOUND = http_error(404, server_error("Order 1 not found"))
NO_TRADES = trades()


def sent_rig(intent=None, **answers):
    """A rig whose intent (a market buy unless given) was placed as order 1."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(intent or make_intent("i-1"))
    return rig


def poll_times(rig, until_seconds, step_seconds=0.5):
    """Call `due_polls` every `step_seconds` until `until_seconds` after now, and return
    the seconds at which `orderstatus` was called."""
    start = rig.clock.now_ns
    called_at = []

    def note(name, kwargs):
        if name == "orderstatus":
            called_at.append((rig.clock.now_ns - start) / NS)

    rig.client.before_call = note
    while rig.clock.now_ns - start < until_seconds * NS:
        rig.clock.advance(step_seconds)
        rig.path.due_polls()
    return called_at


def stream_word(rig, status):
    """The stream reports order 1 in `status`, with no fill, now."""
    rig.path.on_update(update("1", status, 0, ts_recv_ns=rig.clock.now_ns))


def test_an_order_with_no_word_is_polled_from_two_seconds_then_every_second():
    rig = sent_rig(orderstatus=[NOT_FOUND] * 4)

    assert poll_times(rig, until_seconds=5) == [2.0, 3.0, 4.0, 5.0]


def test_an_order_is_not_polled_before_two_seconds():
    rig = sent_rig()

    assert poll_times(rig, until_seconds=1.5) == []


def test_a_market_order_the_stream_reports_open_is_polled_every_second_until_terminal():
    rig = sent_rig(orderstatus=[status_answer("1", "open")] * 3, tradebook=[NO_TRADES] * 3)
    stream_word(rig, OrderStatus.OPEN)

    assert poll_times(rig, until_seconds=3) == [1.0, 2.0, 3.0]


def test_a_limit_order_the_stream_reports_open_is_polled_every_thirty_seconds():
    limit = make_intent("i-1", price_type="LIMIT", price=811.0)
    rig = sent_rig(limit, orderstatus=[status_answer("1", "open")] * 2, tradebook=[NO_TRADES] * 2)
    stream_word(rig, OrderStatus.OPEN)

    assert poll_times(rig, until_seconds=61) == [30.0, 60.0]


def test_a_stop_the_stream_reports_waiting_is_polled_every_thirty_seconds():
    stop = make_intent("i-1", purpose=Purpose.STOP, action="SELL", price_type="SL-M")
    rig = sent_rig(stop, orderstatus=[status_answer("1", "trigger pending")])
    stream_word(rig, OrderStatus.TRIGGER_PENDING)

    assert poll_times(rig, until_seconds=59) == [30.0]


def test_a_resting_order_only_a_poll_has_seen_is_polled_every_second():
    stop = make_intent("i-1", purpose=Purpose.STOP, action="SELL", price_type="SL-M")
    rig = sent_rig(stop, orderstatus=[status_answer("1", "trigger pending")] * 3)

    assert poll_times(rig, until_seconds=4) == [2.0, 3.0, 4.0]


def test_a_terminal_stream_update_stops_the_polls():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10))

    assert poll_times(rig, until_seconds=40) == []


def test_a_terminal_poll_stops_the_polls():
    rig = sent_rig(
        orderstatus=[status_answer("1", "complete", average_price=812.35)],
        tradebook=[trades(("1", 10, 812.35))],
    )

    assert poll_times(rig, until_seconds=40) == [2.0]


def test_a_safety_poll_that_repeats_what_the_stream_said_returns_nothing():
    stop = make_intent("i-1", purpose=Purpose.STOP, action="SELL", price_type="SL-M")
    rig = sent_rig(stop, orderstatus=[status_answer("1", "trigger pending")])
    stream_word(rig, OrderStatus.TRIGGER_PENDING)
    rig.clock.advance(30)

    assert rig.path.due_polls() == []
    assert len(rig.client.calls_to("orderstatus")) == 1


def test_a_poll_asks_for_the_order_with_its_strategy():
    rig = sent_rig(orderstatus=[NOT_FOUND])
    rig.clock.advance(2)

    rig.path.due_polls()

    assert rig.client.calls_to("orderstatus") == [{"order_id": "1", "strategy": STRATEGY}]


def test_a_failed_poll_is_made_again_a_second_later():
    rig = sent_rig(orderstatus=[NOT_FOUND, NOT_FOUND])

    assert poll_times(rig, until_seconds=3) == [2.0, 3.0]


def test_rounds_are_at_most_one_a_second_however_often_they_are_asked_for():
    rig = sent_rig(orderstatus=[NOT_FOUND] * 100)

    times = poll_times(rig, until_seconds=10, step_seconds=0.1)

    assert len(times) == 9


def test_a_round_reads_the_tradebook_once_for_all_its_orders():
    rig = make_rig(
        placeorder=[placed("1"), placed("2")],
        orderstatus=[status_answer("1", "cancelled"), status_answer("2", "cancelled")],
        tradebook=[trades(("1", 4, 812.0), ("2", 5, 813.0))],
    )
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))
    rig.clock.advance(2)

    polled = rig.path.due_polls()

    assert [(u.order_id, u.filled_quantity) for u in polled] == [("1", 4), ("2", 5)]
    assert len(rig.client.calls_to("tradebook")) == 1


def test_a_round_polls_at_most_three_orders_the_most_overdue_first():
    ids = [str(n) for n in range(1, 6)]
    rig = make_rig(placeorder=[placed(order_id) for order_id in ids], orderstatus=[NOT_FOUND] * 10)
    for order_id in ids:
        rig.path.send(make_intent(f"i-{order_id}"))
        rig.clock.advance(0.1)
    rig.clock.advance(2)

    rig.path.due_polls()
    rig.clock.advance(1)
    rig.path.due_polls()

    polled = [kwargs["order_id"] for kwargs in rig.client.calls_to("orderstatus")]
    assert polled == ["1", "2", "3", "4", "5", "1"]


def test_a_poll_with_an_unknown_status_is_logged_and_the_round_goes_on(caplog):
    rig = make_rig(
        placeorder=[placed("1"), placed("2")],
        orderstatus=[
            status_answer("1", "after market order req received"),
            status_answer("2", "rejected"),
        ],
    )
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))
    rig.clock.advance(2)

    polled = rig.path.due_polls()

    assert [u.order_id for u in polled] == ["2"]
    assert "after market order req received" in caplog.text


def test_polls_stop_while_the_guard_is_locked():
    rig = sent_rig(orderstatus=[NOT_FOUND])
    rig.guard.locked = True

    assert poll_times(rig, until_seconds=5) == []


def test_a_poll_answered_in_the_other_mode_is_not_taken():
    sandbox = {**status_answer("1", "rejected"), "mode": "analyze"}
    rig = sent_rig(orderstatus=[sandbox])
    rig.clock.advance(2)

    assert rig.path.due_polls() == []
    assert rig.guard.locked
    assert rig.journal.updates == []


def test_a_tradebook_answered_in_the_other_mode_prices_nothing():
    sandbox = {**trades(("1", 10, 812.35)), "mode": "analyze"}
    rig = sent_rig(orderstatus=[status_answer("1", "complete")], tradebook=[sandbox])
    rig.clock.advance(2)

    assert rig.path.due_polls() == []
    assert rig.guard.responses[-1] is sandbox
