"""A live session and OpenAlgo's mode, on the fake OpenAlgo (`_live/session.py`;
16-session-design 6.1, 6.2 and 7.2 rows F35 to F37, F70 and F84, unit 8, run without leg
rules): the analyzer switch flipped during a session, semi-auto mode, a sandbox and a
live session on one journal the same day, and a guard read that hangs.

The strategy is `BuyAndHold` (buys 10 SBIN at its first decision, 14:42:01 IST, sells
three bars later). After a guard lock the session sends nothing, reads no book, and ends
at Ctrl+C with what it knew.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_invariants import InvariantBroken, assert_session_invariants
from session_samples import NS, BuyAndHold, alerts, ist, rig

from openalgo.backtest._live._session_loop import Session, StopFlag
from openalgo.backtest._live._types import LOGGER_NAME, Mode

#: The calls that send an order or read the session's books.
SENDS_AND_READS = frozenset(
    {
        "placeorder",
        "modifyorder",
        "cancelorder",
        "orderstatus",
        "orderbook",
        "tradebook",
        "positionbook",
    }
)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def calls_after(world, instant_ns: int) -> list[str]:
    return [
        call.what
        for call in world.fake.log.calls()
        if call.ts_ns > instant_ns and call.what in SENDS_AND_READS
    ]


def test_F35_after_the_switch_flips_to_sandbox_nothing_is_sent_or_read(world, caplog):
    flipped = ist(14, 43, 30)
    world.script.at(flipped, lambda: world.fake.set_analyzer(True))
    world.stop_at(ist(14, 47))
    outcome = world.run(BuyAndHold)
    assert outcome.stopped_by == "mode_breach"
    assert outcome.mode_breach is not None
    locked = flipped + 6 * NS
    assert calls_after(world, locked) == []
    assert world.fake.position("SBIN", "NSE", Mode.LIVE) == 10
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]
    assert "mode breach" in outcome.open_positions[0].note
    assert alerts(caplog, "mode_breach")
    assert any("SBIN NSE 10 units" in text for text in alerts(caplog, "error"))


def test_F35_a_book_read_in_the_instant_of_the_guard_lock_breaks_the_invariants(world, monkeypatch):
    # Finding 3 of unit 14 round 1: a read at the very instant the guard locked (the same
    # time on the manual clock) is found by the fake log's order, not missed by its time.
    locked = Session.guard_locked

    def read_then_lock(session, breach):
        session.parts.clients.guard.positionbook()
        locked(session, breach)

    monkeypatch.setattr(Session, "guard_locked", read_then_lock)
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    world.stop_at(ist(14, 47))
    world.run(BuyAndHold)
    with pytest.raises(InvariantBroken, match="calls after the guard lock"):
        world.trace.check_now(assert_session_invariants)


def test_F35_a_session_locked_by_the_guard_ends_at_the_close_plus_its_grace(world):
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    outcome = world.run(BuyAndHold)
    assert outcome.stopped_by == "mode_breach"
    assert outcome.ended_ns >= ist(15, 35)


def test_F36_after_the_switch_flips_to_live_nothing_reaches_the_live_book(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(False))
    world.stop_at(ist(14, 47))
    outcome = world.run(BuyAndHold)
    assert outcome.mode == "sandbox"
    assert outcome.stopped_by == "mode_breach"
    assert world.fake.orders(Mode.LIVE) == []
    assert [order.action for order in world.fake.orders(Mode.SANDBOX)] == ["BUY"]


def test_F37_semi_auto_mode_queues_the_entry_and_the_session_sends_nothing_more(world, caplog):
    world.script.at(ist(14, 41), lambda: world.fake.set_semi_auto(True))
    world.stop_at(ist(14, 47))
    outcome = world.run(BuyAndHold)
    assert world.orders() == []
    assert outcome.stopped_by == "mode_breach"
    assert [call.what for call in world.fake.log.calls("placeorder")] == ["placeorder"]
    assert any("Action Center" in text for text in alerts(caplog, "error"))


def test_F70_a_sandbox_session_then_a_live_one_on_one_journal_keep_their_own_intents(world):
    world.fake.set_analyzer(True)
    world.stop_at(ist(14, 44))
    sandbox = world.run(BuyAndHold)
    assert sandbox.mode == "sandbox"
    world.fake.set_analyzer(False)
    world.stop, world.feed = StopFlag(), None
    live = world.run(BuyAndHold)
    assert live.mode == "live"
    assert live.paths["journal"] == sandbox.paths["journal"]
    assert [order.action for order in world.fake.orders(Mode.LIVE)] == ["BUY", "SELL"]
    with open(live.paths["events"], encoding="utf-8") as stream:
        sent = [json.loads(line) for line in stream if '"kind": "sent"' in line]
    entries = {event["intent_id"] for event in sent if event["intent_id"].endswith("-o1")}
    assert entries == {
        "20261116-sandbox-1-BuyAndHold-o1",
        "20261116-live-1-BuyAndHold-o1",
    }


def test_F84_a_hung_guard_read_does_not_hold_back_an_exit_that_is_due(world, caplog):
    world.script.at(ist(14, 45, 1), lambda: world.fake.faults.slow("analyzer", 60))
    outcome = world.run(BuyAndHold)
    sell = next(order for order in world.orders() if order.action == "SELL")
    assert ist(14, 45, 1) <= sell.placed_ns <= ist(14, 45, 2)
    failed = [
        r.getMessage() for r in caplog.records if "analyzerstatus read failed" in r.getMessage()
    ]
    assert len(failed) == 1 and "(1 in a row)" in failed[0]
    assert outcome.mode_breach is None
    assert outcome.stopped_by == "session_end"
