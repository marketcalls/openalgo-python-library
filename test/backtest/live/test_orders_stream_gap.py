"""A gap in the order-updates stream (16-session-design 2.5 A.12 and 3.1 step 14): every
followed order still working that the stream has not spoken for since the gap began is
polled once, in the polling rounds (at most three orders a round, one round a second),
instead of waiting for its 30-second safety poll."""

import pytest
from orders_fakes import (
    NS,
    PROXY_504,
    START_NS,
    TIMEOUT,
    FakeJournal,
    make_intent,
    make_rig,
    placed,
    status_answer,
    trades,
    update,
)

from openalgo.backtest._live._types import OrderStatus

ORDER_IDS = ["1", "2", "3", "4", "5"]


def resting(order_id):
    """The poll answer of a resting LIMIT buy, with its fills (none) given."""
    return status_answer(order_id, "open", pricetype="LIMIT", price=811.0, filled_quantity=0)


def streamed_rig(order_ids=ORDER_IDS, **answers):
    """A rig whose LIMIT buys were placed and reported open by the stream at START_NS, so
    each waits for its 30-second safety poll."""
    rig = make_rig(placeorder=[placed(order_id) for order_id in order_ids], **answers)
    for order_id in order_ids:
        rig.path.send(make_intent(f"i-{order_id}", price_type="LIMIT", price=811.0))
        rig.path.on_update(update(order_id, OrderStatus.OPEN, 0, ts_recv_ns=START_NS))
    return rig


def polled_per_round(rig, rounds):
    """The orders `orderstatus` was asked for in each of `rounds` rounds, a second apart."""
    per_round = []
    for _ in range(rounds):
        before = len(rig.client.calls_to("orderstatus"))
        rig.path.due_polls()
        per_round.append([call["order_id"] for call in rig.client.calls_to("orderstatus")[before:]])
        rig.clock.advance(1)
    return per_round


def test_a_gap_polls_every_followed_order_once_three_a_round():
    rig = streamed_rig(orderstatus=[resting(order_id) for order_id in ORDER_IDS])
    rig.clock.advance(5)

    rig.path.stream_gap(START_NS + 1 * NS)

    assert polled_per_round(rig, rounds=3) == [["1", "2", "3"], ["4", "5"], []]


def test_a_gap_reads_nothing_by_itself():
    rig = streamed_rig()

    rig.path.stream_gap(START_NS + 1 * NS)

    assert rig.client.calls_to("orderstatus") == []


def test_without_a_gap_the_streamed_orders_wait_for_their_safety_poll():
    rig = streamed_rig()
    rig.clock.advance(5)

    assert polled_per_round(rig, rounds=3) == [[], [], []]


def test_an_order_the_stream_spoke_for_since_the_gap_began_is_not_polled():
    rig = streamed_rig(["1", "2"], orderstatus=[resting("1")])
    rig.clock.advance(5)
    rig.path.on_update(update("2", OrderStatus.OPEN, 0, ts_recv_ns=rig.clock.now_ns))

    rig.path.stream_gap(START_NS + 1 * NS)

    assert polled_per_round(rig, rounds=2) == [["1"], []]


def test_a_terminal_order_is_not_polled_for_a_gap():
    rig = streamed_rig(["1", "2"], orderstatus=[resting("2")])
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, ts_recv_ns=START_NS))
    rig.clock.advance(5)

    rig.path.stream_gap(START_NS + 1 * NS)

    assert polled_per_round(rig, rounds=2) == [["2"], []]


def test_after_its_gap_poll_an_order_waits_for_its_safety_poll_again():
    rig = streamed_rig(["1"], orderstatus=[resting("1"), resting("1")])
    rig.clock.advance(5)
    rig.path.stream_gap(START_NS + 1 * NS)

    polled = polled_per_round(rig, rounds=31)

    assert [n for n, orders in enumerate(polled) if orders] == [0, 30]


@pytest.mark.parametrize(
    "failed",
    [TIMEOUT, PROXY_504, status_answer("1", "after market order req received")],
    ids=["timeout", "gateway page", "unreadable row"],
)
def test_a_gap_poll_that_gives_no_word_is_made_again_in_the_next_round(failed):
    rig = streamed_rig(["1"], orderstatus=[failed, resting("1")])
    rig.clock.advance(5)
    rig.path.stream_gap(START_NS + 1 * NS)

    assert polled_per_round(rig, rounds=3) == [["1"], ["1"], []]


def test_a_check_that_gives_no_word_leaves_the_gap_poll_due_in_the_next_round():
    rig = streamed_rig(["1"], orderstatus=[TIMEOUT, resting("1")])
    rig.clock.advance(5)
    rig.path.stream_gap(START_NS + 1 * NS)
    rig.path.check_order("1")
    rig.clock.advance(1)

    assert polled_per_round(rig, rounds=2) == [["1"], []]


def test_a_gap_poll_that_keeps_failing_takes_its_turn_with_the_others():
    rig = streamed_rig(["1", "2", "3", "4"], orderstatus=[TIMEOUT] * 6)
    rig.clock.advance(5)
    rig.path.stream_gap(START_NS + 1 * NS)

    assert polled_per_round(rig, rounds=2) == [["1", "2", "3"], ["4", "1", "2"]]


def test_a_gap_poll_hands_over_a_fill_the_stream_missed():
    complete = status_answer("1", "complete", pricetype="LIMIT", price=811.0, average_price=811.0)
    rig = streamed_rig(["1"], orderstatus=[complete], tradebook=[trades(("1", 10, 811.0))])
    rig.clock.advance(5)
    rig.path.stream_gap(START_NS + 1 * NS)

    (news,) = rig.path.due_polls()

    assert (news.status, news.filled_quantity, news.average_price) == (
        OrderStatus.COMPLETE,
        10,
        811.0,
    )


def test_a_gap_after_a_restart_polls_each_restored_order():
    journal = FakeJournal()
    for order_id in ("1", "2"):
        intent = make_intent(f"i-{order_id}", price_type="LIMIT", price=811.0)
        journal.record_intent(intent)
        journal.bind(intent.intent_id, order_id)
        journal.record_update(update(order_id, OrderStatus.OPEN, 0, ts_recv_ns=START_NS))
    rig = make_rig(journal=journal, orderstatus=[resting("1"), resting("2")])
    rig.clock.advance(60)
    rig.path.restore()

    rig.path.stream_gap(rig.clock.now_ns)

    assert polled_per_round(rig, rounds=2) == [["1", "2"], []]
