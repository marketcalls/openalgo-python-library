"""A probe while OpenAlgo is silent (16-session-design 4.9): one order-book read, which lifts
the hold on sending only when OpenAlgo answers in its own JSON, and the time the silence
began."""

import pytest
from orders_fakes import (
    DROPPED,
    NS,
    PROXY_504,
    START_NS,
    TIMEOUT,
    book,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
)

from openalgo.backtest._live._types import Purpose

OWN_ERROR = http_error(500, server_error("Internal server error"))
GATEWAY_JSON = http_error(504, {"message": "Endpoint request timed out"})


def silent_rig(**answers):
    """A rig whose first placement timed out at START_NS: OpenAlgo is silent."""
    rig = make_rig(placeorder=[TIMEOUT, *answers.pop("placeorder", [])], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


@pytest.mark.parametrize("answer", [book(), OWN_ERROR], ids=["order book", "its own error"])
def test_a_probe_answered_in_openalgos_own_json_lifts_the_silent_hold(answer):
    rig = silent_rig(orderbook=[answer])

    assert rig.path.probe() is True
    assert rig.path.silent_since_ns is None


@pytest.mark.parametrize(
    "answer",
    [TIMEOUT, DROPPED, PROXY_504, GATEWAY_JSON],
    ids=["timeout", "dropped", "gateway page", "gateway json"],
)
def test_a_probe_without_openalgos_own_answer_keeps_the_silent_hold(answer):
    rig = silent_rig(orderbook=[answer])
    rig.clock.advance(2)

    assert rig.path.probe() is False
    assert rig.path.silent_since_ns == START_NS


def test_once_a_probe_lifts_the_hold_an_exit_of_another_instrument_is_sent():
    rig = silent_rig(orderbook=[book()], placeorder=[placed("2")])
    rig.path.probe()

    outcome = rig.path.send(make_intent("i-2", symbol="INFY", purpose=Purpose.EXIT))

    assert outcome.order_id == "2"


def test_a_probe_reads_the_order_book_once_and_hands_its_answer_to_the_guard():
    answer = book()
    rig = silent_rig(orderbook=[answer])

    rig.path.probe()

    assert rig.client.calls_to("orderbook") == [{}]
    assert rig.guard.responses[-1] is answer


def test_a_probe_reads_nothing_while_the_guard_is_locked():
    rig = silent_rig()
    rig.guard.locked = True

    assert rig.path.probe() is False
    assert rig.client.calls_to("orderbook") == []


def test_a_probe_answered_in_the_other_mode_locks_the_guard():
    rig = silent_rig(orderbook=[{**book(), "mode": "analyze"}])

    rig.path.probe()

    assert rig.guard.locked


def test_openalgo_is_silent_since_the_answer_that_went_silent():
    rig = make_rig(placeorder=[TIMEOUT])
    rig.client.before_call = lambda name, kwargs: rig.clock.advance(3)

    rig.path.send(make_intent("i-1"))

    assert rig.path.silent_since_ns == START_NS + 3 * NS


def test_a_probe_that_times_out_makes_an_answering_openalgo_silent():
    rig = make_rig(orderbook=[TIMEOUT])

    assert rig.path.silent_since_ns is None
    assert rig.path.probe() is False
    assert rig.path.silent_since_ns == START_NS
