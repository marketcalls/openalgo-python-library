"""One targeted read of an order (`check_order`, 16-session-design 4.3 and 4.5): the row's
terms on each kind of answer, its news taken as a poll's, the guard and the server watch;
and the latest word known of an order (`last_known`) after each kind of update."""

import pytest
from orders_fakes import (
    STRATEGY,
    TIMEOUT,
    FakeGuard,
    FakeJournal,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
    status_answer,
    trades,
    update,
)

from openalgo.backtest._live._types import Mode, OrderStatus, Purpose
from openalgo.backtest._live.orders import OrderCheck
from openalgo.backtest.errors import LiveError, ModeChangedError

STOP = make_intent("i-1", purpose=Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=805)
STOP_ROW = {"action": "SELL", "pricetype": "SL-M", "trigger_price": 805.0}
NOT_FOUND = http_error(404, server_error("Order 1 not found"))


def stop_rig(**answers):
    """A rig whose SL-M sell stop of 10 SBIN at 805 was placed as order 1."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(STOP)
    return rig


def stop_answer(status, **row):
    """A live `orderstatus` answer for the stop: a broker's row, without its fills."""
    return status_answer("1", status, **STOP_ROW | row)


# ----- the row's terms ---------------------------------------------------------------


def test_a_check_reads_the_status_quantity_prices_and_price_type_of_the_row():
    rig = stop_rig(orderstatus=[stop_answer("trigger pending")])

    check = rig.path.check_order("1")

    assert (check.order_id, check.status, check.quantity) == ("1", OrderStatus.TRIGGER_PENDING, 10)
    assert (check.price, check.trigger_price, check.price_type) == (0.0, 805.0, "SL-M")
    assert check.readable and check.reason == ""


def test_a_row_without_its_fills_gives_no_filled_quantity():
    rig = stop_rig(orderstatus=[stop_answer("trigger pending")])

    assert rig.path.check_order("1").filled_quantity is None


def test_a_sandbox_row_gives_its_filled_quantity_and_price_type():
    # The sandbox's row (sandbox/order_manager.py): "price_type" and its fills.
    sandbox_row = {"price_type": "SL", "pricetype": None, "price": 804.0, "filled_quantity": 4}
    answer = {**stop_answer("open", **sandbox_row), "mode": "analyze"}
    rig = make_rig(
        guard=FakeGuard(mode=Mode.SANDBOX),
        placeorder=[{**placed("1"), "mode": "analyze"}],
        orderstatus=[answer],
    )
    rig.path.send(STOP)

    check = rig.path.check_order("1")

    assert (check.filled_quantity, check.price, check.price_type) == (4, 804.0, "SL")


def test_a_check_asks_for_the_order_with_its_strategy():
    rig = stop_rig(orderstatus=[stop_answer("trigger pending")])

    rig.path.check_order("1")

    assert rig.client.calls_to("orderstatus") == [{"order_id": "1", "strategy": STRATEGY}]


# ----- the news ----------------------------------------------------------------------


def test_a_first_word_is_news_handed_over_journaled_and_checked_by_the_guard():
    rig = stop_rig(orderstatus=[stop_answer("trigger pending")])

    news = rig.path.check_order("1").news

    assert (news.status, news.filled_quantity, news.source) == (
        OrderStatus.TRIGGER_PENDING,
        0,
        "poll",
    )
    assert rig.journal.updates == [news]
    assert rig.guard.updates == [news]


def test_a_word_the_stream_already_gave_is_not_news():
    rig = stop_rig(orderstatus=[stop_answer("trigger pending")])
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    check = rig.path.check_order("1")

    assert check.news is None
    assert check.status is OrderStatus.TRIGGER_PENDING


def test_a_rejected_row_is_news_at_once():
    rig = stop_rig(orderstatus=[stop_answer("rejected", rejection_reason="RMS: margin")])

    news = rig.path.check_order("1").news

    assert (news.status, news.rejection_reason) == (OrderStatus.REJECTED, "RMS: margin")


def test_a_fill_found_by_a_check_waits_for_the_tradebook_then_the_next_round_hands_it_over():
    rig = stop_rig(
        orderstatus=[stop_answer("complete", average_price=804.5)],
        tradebook=[trades(("1", 6, 804.5), ("1", 4, 804.0))],
    )

    assert rig.path.check_order("1").news is None
    (news,) = rig.path.due_polls()

    assert (news.status, news.filled_quantity) == (OrderStatus.COMPLETE, 10)
    assert news.average_price == pytest.approx((6 * 804.5 + 4 * 804.0) / 10)
    assert rig.client.calls_to("orderstatus") == [{"order_id": "1", "strategy": STRATEGY}]


def test_a_cancel_found_by_a_check_waits_for_the_tradebook_to_confirm_its_fills():
    rig = stop_rig(orderstatus=[stop_answer("cancelled", filled_quantity=0)])

    check = rig.path.check_order("1")

    assert (check.status, check.news) == (OrderStatus.CANCELLED, None)
    assert [held.filled_quantity for held in rig.path.held_fills()] == [0]


def test_a_row_without_its_fills_asks_the_next_round_to_poll_it_with_the_tradebook():
    # The stream spoke for the stop, so without the check its next poll is 30 s away.
    rig = stop_rig(
        orderstatus=[stop_answer("cancelled"), stop_answer("cancelled")],
        tradebook=[trades(("1", 3, 804.0))],
    )
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.path.check_order("1")
    rig.clock.advance(1)

    (news,) = rig.path.due_polls()

    assert (news.status, news.filled_quantity) == (OrderStatus.CANCELLED, 3)


# ----- answers that cannot be read ---------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "says"),
    [
        (NOT_FOUND, "Order 1 not found"),
        (TIMEOUT, "Request timed out"),
        (stop_answer("after market order req received"), "after market order req received"),
        (stop_answer("open", quantity="10.5"), "quantity 10.5 is not a whole number"),
        (stop_answer("open", trigger_price=None), "trigger_price None is not a number"),
        (stop_answer("open", pricetype=""), "gives no price type"),
    ],
    ids=["not found", "timeout", "unknown status", "quantity", "no trigger price", "price type"],
)
def test_an_answer_that_cannot_be_read_gives_an_unreadable_check_saying_why(answer, says):
    rig = stop_rig(orderstatus=[answer])

    check = rig.path.check_order("1")

    assert not check.readable
    assert check == OrderCheck("1", reason=check.reason)
    assert says in check.reason


def test_a_check_that_times_out_leaves_openalgo_silent():
    rig = stop_rig(orderstatus=[TIMEOUT])

    rig.path.check_order("1")

    assert rig.path.silent_since_ns is not None


def test_a_check_answered_in_openalgos_own_json_lifts_the_silent_hold():
    rig = make_rig(placeorder=[placed("1"), TIMEOUT], orderstatus=[NOT_FOUND])
    rig.path.send(STOP)
    rig.path.send(make_intent("i-2", symbol="INFY"))
    assert rig.path.silent_since_ns is not None

    rig.path.check_order("1")

    assert rig.path.silent_since_ns is None


# ----- refused -----------------------------------------------------------------------


def test_a_check_while_the_guard_is_locked_raises_and_reads_nothing():
    rig = stop_rig()
    rig.guard.locked = True

    with pytest.raises(ModeChangedError, match="the mode guard is locked"):
        rig.path.check_order("1")
    assert rig.client.calls_to("orderstatus") == []


def test_a_check_answered_in_the_other_mode_raises_and_takes_nothing():
    rig = stop_rig(orderstatus=[{**stop_answer("rejected"), "mode": "analyze"}])

    with pytest.raises(ModeChangedError, match="other mode"):
        rig.path.check_order("1")
    assert rig.journal.updates == []


def test_a_check_of_an_order_this_session_did_not_send_raises():
    rig = stop_rig()

    with pytest.raises(LiveError, match="order 9 was not sent by this session"):
        rig.path.check_order("9")


# ----- the latest word known ---------------------------------------------------------


def sent_rig(**answers):
    """A rig whose LIMIT buy of 10 SBIN was placed as order 1."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1", price_type="LIMIT", price=812.0))
    return rig


def test_last_known_is_none_before_any_word():
    assert sent_rig().path.last_known("1") is None


def test_last_known_is_none_for_an_order_this_session_did_not_send():
    assert sent_rig().path.last_known("9") is None


def test_last_known_is_the_streams_word_handed_over():
    rig = sent_rig()
    word = update("1", OrderStatus.OPEN, 4)
    rig.path.on_update(word)

    assert rig.path.last_known("1") == word


def test_last_known_is_a_polls_word_handed_over():
    rig = sent_rig(orderstatus=[status_answer("1", "open", pricetype="LIMIT", filled_quantity=0)])
    rig.clock.advance(2)
    (polled,) = rig.path.due_polls()

    assert rig.path.last_known("1") == polled


def test_last_known_is_a_cancel_held_for_the_tradebook():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 0))
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    assert rig.path.last_known("1").status is OrderStatus.CANCELLED


def test_last_known_is_a_fill_held_for_its_price():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    known = rig.path.last_known("1")

    assert (known.status, known.filled_quantity, known.average_price) == (
        OrderStatus.COMPLETE,
        10,
        0.0,
    )


def test_last_known_is_a_released_fill():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    released = rig.path.release_fill("1", 812.5)

    assert rig.path.last_known("1") == released


def test_last_known_after_a_restart_is_the_journals_last_update():
    journal = FakeJournal()
    journal.record_intent(make_intent("i-1"))
    journal.bind("i-1", "1")
    word = update("1", OrderStatus.OPEN, 0)
    journal.record_update(word)
    rig = make_rig(journal=journal)

    rig.path.restore()

    assert rig.path.last_known("1") == word
