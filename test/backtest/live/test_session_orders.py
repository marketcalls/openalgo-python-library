"""A live session's orders through faults of the order calls, on the fake OpenAlgo
(`_live/session.py`; 16-session-design 7.2 rows F01 to F10, unit 8, run without leg
rules): an entry whose answer is lost, refused, slow or rate limited, and an exit that is
rate limited.

The strategy is `BuyAndHold`: it buys 10 SBIN at its first decision (14:42:01 IST) and
sells them three bars later. Each fault is scheduled before the session starts, so it
meets the session's first order call of its endpoint.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, BuyAndHold, alerts, ist, rig

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.orders import BROKER_WINDOW_NS

ENTRY_DECIDED = ist(14, 42, 1)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def placements(world, action: str = "BUY") -> list:
    """The placeorder calls the fake saw for SBIN with `action`."""
    return [
        call
        for call in world.fake.log.calls("placeorder")
        if call.detail["request"].get("action") == action
    ]


def entries_at_the_fake(world) -> list:
    return [order for order in world.orders() if order.action == "BUY"]


class Seen(BuyAndHold):
    """BuyAndHold that keeps the names of the order events it was given."""

    names: list[str] = []

    def on_order(self, order):
        type(self).names.append(order["name"])


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(lambda f: f.bad_gateway_page("placeorder", reached_broker=True), id="F01"),
        pytest.param(lambda f: f.timeout("placeorder", reached_broker=True), id="F03"),
        pytest.param(
            lambda f: f.gateway_error("placeorder", status=504, reached_broker=True), id="F04"
        ),
        pytest.param(lambda f: f.drop_connection("placeorder", reached_broker=True), id="F05"),
        pytest.param(lambda f: f.lose_answer("placeorder"), id="F07"),
    ],
)
def test_an_entry_whose_answer_is_lost_after_it_reached_the_broker_is_bound_once(world, fault):
    fault(world.fake.faults)
    outcome = world.run(BuyAndHold)
    assert len(entries_at_the_fake(world)) == 1
    assert len(placements(world)) == 1
    sells = [order for order in world.orders() if order.action == "SELL"]
    assert [order.quantity for order in sells] == [10]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "session_end"


class InfyHold(BuyAndHold):
    """BuyAndHold on INFY."""

    symbol = "INFY"


def test_F03_nothing_is_sent_for_any_instrument_between_a_timeout_and_an_answered_read(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.fake.faults.timeout("placeorder", reached_broker=True)
    outcome = world.run([BuyAndHold, InfyHold])
    calls = world.fake.log.calls()
    entry = next(i for i, call in enumerate(calls) if call.what == "placeorder")
    answered = next(
        i
        for i, call in enumerate(calls[entry + 1 :], start=entry + 1)
        if call.what in ("orderbook", "orderstatus", "tradebook")
    )
    assert [call.what for call in calls[entry + 1 : answered]].count("placeorder") == 0
    # INFY's entry, decided with SBIN's, waited behind the silence and was dropped.
    assert list(outcome.outputs["InfyHold"]["orders"]["status"]) == ["Rejected"]
    assert [order.symbol for order in world.orders() if order.action == "BUY"] == ["SBIN"]


def test_F04_a_gateway_error_is_not_taken_as_not_placed_within_the_brokers_window(world):
    world.fake.faults.gateway_error("placeorder", status=504, reached_broker=True)
    outcome = world.run(BuyAndHold)
    rejected = [
        row for row in outcome.outputs["BuyAndHold"]["orders"]["status"] if row == "Rejected"
    ]
    assert rejected == []


def test_F02_an_entry_that_never_reached_the_broker_is_rejected_after_the_window(world):
    Seen.names = []
    world.fake.faults.bad_gateway_page("placeorder")
    outcome = world.run(Seen)
    assert entries_at_the_fake(world) == []
    statuses = list(outcome.outputs["Seen"]["orders"]["status"])
    assert statuses == ["Rejected"]
    assert "Rejected" in Seen.names
    rejected_at = _event_time(outcome, "resolved")
    assert rejected_at >= ENTRY_DECIDED + BROKER_WINDOW_NS


def test_F06_an_entry_refused_a_connection_is_rejected_within_seconds(world):
    world.fake.faults.refuse_connection("placeorder")
    outcome = world.run(BuyAndHold)
    assert entries_at_the_fake(world) == []
    assert list(outcome.outputs["BuyAndHold"]["orders"]["status"]) == ["Rejected"]
    assert _event_time(outcome, "resolved") < ENTRY_DECIDED + 15 * NS


def test_F08_a_rate_limited_entry_is_rejected_once_and_not_sent_again(world, caplog):
    world.fake.faults.rate_limit("placeorder")
    outcome = world.run(BuyAndHold)
    assert entries_at_the_fake(world) == []
    assert len(placements(world)) == 1
    assert len(alerts(caplog, "reject")) == 1
    assert list(outcome.outputs["BuyAndHold"]["orders"]["status"]) == ["Rejected"]


def test_F09_a_rate_limited_exit_is_asked_for_again_a_second_later_and_fills(world):
    world.script.at(ist(14, 44, 30), lambda: world.fake.faults.rate_limit("placeorder"))
    outcome = world.run(BuyAndHold)
    sells = placements(world, "SELL")
    assert len(sells) == 2
    assert NS <= sells[1].ts_ns - sells[0].ts_ns <= 3 * NS
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()


def test_F10_a_slow_answer_moves_the_clock_and_the_session_catches_up(world):
    world.fake.faults.slow("placeorder", 5)
    outcome = world.run(BuyAndHold)
    assert len(entries_at_the_fake(world)) == 1
    sells = [order for order in world.orders() if order.action == "SELL"]
    # The entry filled on the 14:42 bar; the strategy sells three bars later, as without
    # the slow answer.
    assert ist(14, 45) <= sells[0].placed_ns < ist(14, 45, 5)
    assert outcome.stopped_by == "session_end"


def _event_time(outcome, kind: str) -> int:
    """When the session's event stream first noted `kind`."""
    import json

    with open(outcome.paths["events"], encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] == kind:
                return event["ts_ns"]
    raise AssertionError(f"no {kind} event")


def test_F60_an_entry_the_book_cannot_tell_apart_locks_the_instrument(world, caplog):
    world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)
    other = world.fake.client()

    def same_order_by_another_client():
        # The same order, under the same strategy name, right after the session's.
        other.placeorder(
            strategy="BuyAndHold",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        )

    world.script.at(ist(14, 42, 2) + NS // 2, same_order_by_another_client)
    outcome = world.run(BuyAndHold)
    ambiguous = [text for text in alerts(caplog, "error") if "cannot be told apart" in text]
    assert len(ambiguous) == 1
    # The session's placement and the other client's, and nothing more afterwards.
    assert len(placements(world)) == 2
    assert placements(world, "SELL") == []
    assert outcome.outputs["BuyAndHold"]["orders"]["status"] == ["Submitted"]
