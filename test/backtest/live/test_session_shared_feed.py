"""The market and the books of a shared instrument reach every member, on the fake OpenAlgo
(`_live/_session_feed.py`, `_live/_session_candles.py`, `_live/_session_banner.py`,
`_live/_reconcile_groups.py`; 18-multi-strategy 4.2, 4.3, 4.6 and 4.11, row F115, unit
M6): one subscription and one candle check per instrument, each tick and candle given to
every member in call order, the start banner's lines of a shared instrument, the group
compared with the broker's net, and another client's order tagged with a member's name.

The market and the strategies are `shared_samples`': SBIN at 500.00, A and B buy at
14:42:01 IST, A first. Every test runs the session invariants.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from shared_samples import B_UNITS, TARGET_POINTS, SbinBreakout, SbinTrend, sbin_orders
from stops_samples import STOP_POINTS, StopHold, at_price, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._session_news import RECONCILE_LOCK
from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


def test_a_shared_instrument_is_subscribed_once(world):
    world.run([SbinTrend, SbinBreakout])

    subscribed = [
        (entry["symbol"], entry["exchange"])
        for frame in world.feed.frames
        if frame.get("action") == "subscribe"
        for entry in frame["symbols"]
    ]
    assert subscribed == [("SBIN", "NSE")]


def test_each_candle_check_reads_a_shared_instrument_once_and_feeds_both_members(world):
    world.stop_at(ist(14, 50))
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")

    reads = [c.ts_ns for c in world.fake.log.calls("history") if c.ts_ns >= ist(14, 42)]
    assert reads and len(reads) == len(set(reads))
    drifts = {}
    for event in events(outcome, "drift"):
        drifts.setdefault(event["strategy"], []).append(event["slice_open_ns"])
    assert drifts["SbinTrend"] == drifts["SbinBreakout"] != []


class TargetBoth(StopHold):
    """A B of 50 units with A's stop and target (497.00, 502.00)."""

    qty = B_UNITS
    rules = {"stop": bt.points(STOP_POINTS), "target": bt.points(TARGET_POINTS)}


def test_a_tick_reaches_every_member_so_both_targets_exit_on_it(world):
    at_price(world, ist(14, 44), 502.0)
    outcome = world.run([SbinTrend, TargetBoth])

    exits = [
        (o.strategy, o.quantity)
        for o in world.orders()
        if o.action == "SELL" and o.price_type == "MARKET"
    ]
    assert exits == [("SbinTrend", 100), ("TargetBoth", B_UNITS)]
    for name in ("SbinTrend", "TargetBoth"):
        assert outcome.outputs[name]["trades"]["exit_reason"] == ["leg_target"]


def test_the_banner_names_the_shared_instrument_its_members_and_the_rule_by_hand(world):
    world.run([SbinTrend, SbinBreakout])

    assert (
        "SBIN (NSE, MIS): SbinTrend and SbinBreakout share one broker position; each holds "
        "its own stop; they trade one side at a time and with market orders only"
    ) in world.lines
    assert "Never trade a session's symbol by hand while it runs" in world.lines
    assert [line for line in world.lines if line.startswith("SBIN: freeze quantity")] == [
        "SBIN: freeze quantity none"
    ]


def test_a_strategy_alone_gets_no_shared_line(world):
    world.run(SbinTrend)

    assert not [line for line in world.lines if "share one broker position" in line]
    assert "Never trade a session's symbol by hand while it runs" not in world.lines


def test_the_group_is_compared_with_the_brokers_net_and_agrees(world):
    outcome = world.run([SbinTrend, SbinBreakout])

    compared = [(e["strategy"], e["found"]) for e in events(outcome, "reconcile")]
    assert compared and {found for _, found in compared} == {"agree"}
    assert outcome.reconciliations == ()
    assert events(outcome, "mismatch") == []


def test_F115_another_clients_order_tagged_with_a_members_name_locks_that_member(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    # The sandbox's order book carries each order's strategy, as OpenAlgo's does.
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    other = world.fake.client()
    world.script.at(
        ist(14, 44),
        lambda: other.placeorder(
            strategy="SbinBreakout",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="LIMIT",
            product="MIS",
            quantity=10,
            price=490.0,
        ),
    )
    world.stop_at(ist(14, 46))
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")

    foreign = [r for r in outcome.reconciliations if r.kind == "foreign"]
    assert {r.strategy for r in foreign} == {"SbinBreakout"}
    locks = [(e["strategy"], e["reason"]) for e in events(outcome, "lock")]
    assert ("SbinBreakout", RECONCILE_LOCK) in locks
    assert ("SbinTrend", RECONCILE_LOCK) not in locks
    # Nothing is cancelled: the other client's order works on.
    (tagged,) = [o for o in world.orders() if o.price_type == "LIMIT"]
    assert tagged.status == "open"
    assert world.fake.log.calls("cancelorder") == []
    mismatches = alerts(caplog, "reconcile_mismatch")
    assert mismatches and all("SbinBreakout" in text for text in mismatches)
    assert sbin_orders(world, "SbinTrend")[0].status == "complete"
