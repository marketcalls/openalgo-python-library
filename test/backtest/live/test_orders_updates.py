"""Order updates read from the stream and from `orderstatus` (`normalize_update`), in the
shapes the server sends them in live and sandbox mode."""

import pytest
from orders_fakes import NS, START_NS, status_answer, stream_message

from openalgo.backtest._live._types import Mode, OrderStatus, OrderUpdate
from openalgo.backtest._live.orders import normalize_update
from openalgo.backtest.errors import LiveError


def test_a_stream_message_becomes_an_update_with_its_fields():
    message = stream_message("2611160001", "open", 4, average_price=812.35)

    update = normalize_update(message, ts_recv_ns=START_NS, source="stream")

    assert update == OrderUpdate(
        order_id="2611160001",
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=10,
        filled_quantity=4,
        pending_quantity=6,
        average_price=812.35,
        status=OrderStatus.OPEN,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=START_NS,
        source="stream",
    )


@pytest.mark.parametrize(("mode", "expected"), [("analyze", Mode.SANDBOX), ("live", Mode.LIVE)])
def test_a_stream_message_carries_the_mode_it_was_handled_in(mode, expected):
    message = stream_message("1", "complete", 10, mode=mode)

    assert normalize_update(message, ts_recv_ns=START_NS, source="stream").mode is expected


def test_a_live_orderstatus_answer_becomes_an_update():
    answer = status_answer("1", "complete", quantity="10", average_price=812.35)

    update = normalize_update(answer, ts_recv_ns=START_NS + NS, source="poll")

    assert (update.order_id, update.status, update.mode, update.source) == (
        "1",
        OrderStatus.COMPLETE,
        Mode.LIVE,
        "poll",
    )
    assert (update.quantity, update.filled_quantity, update.pending_quantity) == (10, 10, 0)
    assert update.average_price == 812.35


def test_a_sandbox_orderstatus_answer_becomes_a_sandbox_update():
    # The sandbox's own row (sandbox/order_manager.py get_order_status): "price_type",
    # its fills, its strategy, and the answer marked "analyze".
    answer = {
        "status": "success",
        "data": {
            "orderid": "26111600000001",
            "symbol": "SBIN",
            "exchange": "NSE",
            "action": "SELL",
            "quantity": 10,
            "price": 0.0,
            "trigger_price": 805.0,
            "price_type": "SL-M",
            "product": "MIS",
            "order_status": "trigger pending",
            "average_price": 0.0,
            "filled_quantity": 0,
            "pending_quantity": 10,
            "timestamp": "2026-11-16 10:15:20",
            "strategy": "ema_cross",
        },
        "mode": "analyze",
    }

    update = normalize_update(answer, ts_recv_ns=START_NS, source="poll")

    assert (update.status, update.mode, update.action) == (
        OrderStatus.TRIGGER_PENDING,
        Mode.SANDBOX,
        "SELL",
    )
    assert (update.filled_quantity, update.pending_quantity) == (0, 10)


@pytest.mark.parametrize(
    ("word", "status"),
    [
        ("open", OrderStatus.OPEN),
        ("trigger pending", OrderStatus.TRIGGER_PENDING),
        ("trigger_pending", OrderStatus.TRIGGER_PENDING),
        ("complete", OrderStatus.COMPLETE),
        ("rejected", OrderStatus.REJECTED),
        ("cancelled", OrderStatus.CANCELLED),
        ("Complete", OrderStatus.COMPLETE),
        (" TRIGGER PENDING ", OrderStatus.TRIGGER_PENDING),
    ],
)
def test_every_status_word_of_the_stream_and_the_books_is_read(word, status):
    message = stream_message("1", word, 0)

    assert normalize_update(message, ts_recv_ns=START_NS, source="stream").status is status


def test_an_unknown_status_raises_naming_it():
    message = stream_message("1", "unknown", 0)

    with pytest.raises(LiveError, match="'unknown'"):
        normalize_update(message, ts_recv_ns=START_NS, source="stream")


def test_a_complete_order_reported_with_no_fills_has_filled_all_of_it():
    # The server's polled stream sends 0 for a book row without a filled quantity.
    message = stream_message("1", "complete", 0, pending_quantity=0, average_price=812.35)

    update = normalize_update(message, ts_recv_ns=START_NS, source="stream")

    assert (update.filled_quantity, update.average_price) == (10, 812.35)


def test_a_fill_reported_without_its_price_is_read_with_price_zero():
    # Read as OpenAlgo wrote it; the order path holds such a fill until it is priced.
    message = stream_message("1", "complete", 10, average_price=0.0)

    assert normalize_update(message, ts_recv_ns=START_NS, source="stream").average_price == 0.0


@pytest.mark.parametrize("status", ["rejected", "trigger pending"])
def test_a_status_that_settles_the_fills_needs_no_filled_quantity(status):
    answer = status_answer("1", status)

    update = normalize_update(answer, ts_recv_ns=START_NS, source="poll")

    assert update.filled_quantity == 0


@pytest.mark.parametrize("status", ["open", "cancelled"])
def test_an_open_or_cancelled_order_without_its_fills_raises(status):
    answer = status_answer("1", status)

    with pytest.raises(LiveError, match="tradebook"):
        normalize_update(answer, ts_recv_ns=START_NS, source="poll")


@pytest.mark.parametrize(
    ("status", "pending"), [("open", 10), ("trigger pending", 10), ("rejected", 0)]
)
def test_the_pending_quantity_not_given_is_what_is_left_of_a_working_order(status, pending):
    answer = status_answer("1", status, filled_quantity=0)

    assert normalize_update(answer, ts_recv_ns=START_NS, source="poll").pending_quantity == pending


@pytest.mark.parametrize(
    "message",
    [
        {"status": "error", "message": "Order 1 not found", "mode": "analyze"},
        {"status": "success", "data": []},
        {"orderid": "1", "order_status": "open"},
    ],
)
def test_a_message_that_is_neither_an_update_nor_an_orderstatus_answer_raises(message):
    with pytest.raises(LiveError, match="not an order update"):
        normalize_update(message, ts_recv_ns=START_NS, source="poll")


def test_a_source_other_than_stream_or_poll_raises():
    with pytest.raises(LiveError, match="'stream' or 'poll'"):
        normalize_update(stream_message("1", "open", 0), ts_recv_ns=START_NS, source="book")


def test_an_update_without_an_order_id_raises():
    with pytest.raises(LiveError, match="without an order ID"):
        normalize_update(stream_message("", "open", 0), ts_recv_ns=START_NS, source="stream")


@pytest.mark.parametrize("quantity", ["10.5", "ten", None])
def test_a_quantity_that_is_not_a_whole_number_of_units_raises(quantity):
    message = stream_message("1", "open", 0) | {"quantity": quantity}

    with pytest.raises(LiveError, match="quantity"):
        normalize_update(message, ts_recv_ns=START_NS, source="stream")
