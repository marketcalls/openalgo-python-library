"""A live session whose feed connection drops, over the SDK's real feed on the fake
OpenAlgo's in-process socket (`_live/watch.py`, `_live/feedlink.py`; 16-session-design
2.5 A.12, 6.5 and 7.2 rows F41 and F85, unit 11): the outage's span resumes the bar
builders, the order stream's gap is polled, the SDK replays the subscriptions, and a
reconnect held long enough is a dead feed too.

The session runs under the test seam on the fake's manual clock, with its feed client a
real `openalgo.api` whose socket thread is real: a script action that drops or lets the
connection back waits (in real time, bounded) until the SDK's hooks have run, so the
loop takes their events at the action's instant. The market is `session_samples`': SBIN
trades 10 units at 500.00 every 10 s, at :05, :15 and so on, from 14:40:05 IST.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import ist, rig
from watch_samples import HoldWithTarget, HookWitness, RestingBuy, events, run

from openalgo.backtest._live._types import LOGGER_NAME

DROP_AT, BACK_AT = ist(14, 43), ist(14, 43, 30)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 45, 30))
    return world


@pytest.fixture
def clients(world):
    """The session's clients, its feed client the SDK's own on the fake's socket."""
    return world.clients()


def drop(world, clients, *, hold_reconnect: bool) -> HookWitness:
    """Drop the feed's connection now and wait until the session's disconnect hook ran
    (and, with an immediate reconnect, its reconnect hook too)."""
    witness = HookWitness(clients.feed)
    world.fake.faults.drop_feed(hold_reconnect=hold_reconnect)
    witness.wait("disconnect")
    if not hold_reconnect:
        witness.wait("reconnect")
    return witness


def subscribed_at(world, instant_ns: int) -> list[bool]:
    """Whether, at `instant_ns`, the open socket has SBIN in quote mode and the order
    stream (read while the session runs: it disconnects its feed when it closes)."""
    seen: list[bool] = []

    def look() -> None:
        server = world.fake.feed_server
        sock = server.live_socket()
        entries, orders = server.subscriptions(sock) if sock is not None else (set(), False)
        seen.append(("SBIN", "NSE", 2) in entries and orders)

    world.script.at(instant_ns, look)
    return seen


# -- F41: a drop whose reconnect is held, with a fill meanwhile -----------------------------


@pytest.fixture
def held(world, clients):
    """The connection drops at 14:43:00, the resting buy fills at 400.00 at 14:43:10
    while it is down, and the reconnect is let through at 14:43:30."""
    witness: list[HookWitness] = []

    def fill_resting_buy():
        (order,) = world.orders()
        world.fake.fill(order.order_id, price=400.0)

    def reconnect():
        world.fake.faults.let_feed_reconnect()
        witness[0].wait("reconnect")

    world.script.at(DROP_AT, lambda: witness.append(drop(world, clients, hold_reconnect=True)))
    world.script.at(ist(14, 43, 10), fill_resting_buy)
    world.script.at(BACK_AT, reconnect)
    return world


def test_F41_a_fill_while_the_connection_was_down_is_found_by_the_stream_gaps_poll(held, clients):
    outcome = run(held, RestingBuy, clients=clients, sdk_feed=True)
    fills = [(e["units"], e["price"], e["source"]) for e in events(outcome, "fill")]
    assert fills[0] == (10, 400.0, "poll")
    gaps = [e for e in events(outcome, "request") if e["type"] == "StreamGap"]
    assert [e["ts_ns"] for e in gaps] == [BACK_AT]


def test_F41_the_bar_builders_resume_over_the_outages_span(held, clients):
    outcome = run(held, RestingBuy, clients=clients, sdk_feed=True)
    spans = [(e["start_ns"], e["end_ns"]) for e in events(outcome, "outage")]
    assert spans == [(DROP_AT, BACK_AT)]


def test_F41_the_sdk_replays_the_subscriptions_on_the_new_connection(held, clients):
    seen = subscribed_at(held, ist(14, 44))
    run(held, RestingBuy, clients=clients, sdk_feed=True)
    assert seen == [True]


def test_F41_a_reconnect_held_past_the_dead_time_is_a_dead_feed_until_ticks_return(
    held, clients, caplog
):
    outcome = run(held, RestingBuy, clients=clients, sdk_feed=True)
    dead = events(outcome, "feed_dead")
    back = events(outcome, "feed_back")
    # The last tick before the drop is 14:42:55; the first after the reconnect 14:43:35.
    assert [(e["ts_ns"], e["since_ns"]) for e in dead] == [(ist(14, 43, 5), ist(14, 42, 55))]
    assert [e["ts_ns"] for e in back] == [ist(14, 43, 35)]


# -- F85: a drop with an immediate reconnect --------------------------------------------------


@pytest.fixture
def dropped(world, clients):
    """The connection drops at 14:43:00 and the SDK reconnects at once."""
    world.script.at(DROP_AT, lambda: drop(world, clients, hold_reconnect=False))
    return world


def test_F85_an_immediate_reconnect_resumes_the_bar_builders_over_its_span(dropped, clients):
    outcome = run(dropped, HoldWithTarget, clients=clients, sdk_feed=True)
    spans = [(e["start_ns"], e["end_ns"]) for e in events(outcome, "outage")]
    assert spans == [(DROP_AT, DROP_AT)]


def test_F85_an_immediate_reconnect_polls_the_order_streams_gap(dropped, clients):
    outcome = run(dropped, HoldWithTarget, clients=clients, sdk_feed=True)
    gaps = [e for e in events(outcome, "request") if e["type"] == "StreamGap"]
    assert [e["ts_ns"] for e in gaps] == [DROP_AT]


def test_F85_the_sdk_replays_the_subscriptions_and_ticks_flow_again(dropped, clients):
    seen = subscribed_at(dropped, ist(14, 44))
    outcome = run(dropped, HoldWithTarget, clients=clients, sdk_feed=True)
    assert seen == [True]
    assert events(outcome, "feed_dead") == []


def test_F85_the_slice_of_the_drop_makes_no_bar_and_the_next_bar_holds_its_own_volume(
    dropped, clients
):
    outcome = run(dropped, HoldWithTarget, clients=clients, sdk_feed=True)
    closes = [close for close, _ in HoldWithTarget.seen]
    assert ist(14, 44) not in closes
    assert [e["slice_open_ns"] for e in events(outcome, "missing")] == [ist(14, 43)]
    # The 14:44 bar holds its own six trades of 10 units.
    assert (ist(14, 45), 60.0) in HoldWithTarget.seen
