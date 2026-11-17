"""A dead feed and a broker login lost in a live session, on the fake OpenAlgo
(`_live/watch.py`; 16-session-design 6.2, 6.5 and 2.5 B, and 7.2 rows F38, F39 and F40,
unit 11): the FEED_DEAD alert, the entry lock and its lift a whole slice after the feed
returns, REST quotes as the kernel's prices while the feed is dead, the bar builders
resumed over a dead spell, slices without data marked missing, and what a lost login adds.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN trades 10 units every 10 s, at :05, :15 and so on, at
500.00 unless a test moves the price. A strategy's first decision is at 14:42:01.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, BuyAndHold, Hold, alerts, ist, rig
from watch_samples import (
    Flipper,
    HoldFiveMinutes,
    HoldInfy,
    HoldSbin,
    HoldWithTarget,
    calls_between,
    events,
    run,
)

from openalgo.backtest._live._types import LOGGER_NAME

#: The F40 script: the feed dies at 14:43:30 (its last tick was 14:43:25, so it is dead
#: from 14:43:35), SBIN trades at 506.00 from 14:43:40, and the feed lives again at
#: 14:43:52 (its first tick is 14:43:55).
LAST_TICK, DEAD_AT, BACK_AT = ist(14, 43, 25), ist(14, 43, 35), ist(14, 43, 55)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


@pytest.fixture
def dead_feed(world):
    """The F40 script on the world, the session stopped at 14:46:30."""
    world.script.price = lambda now_ns: 506.0 if now_ns >= ist(14, 43, 40) else 500.0
    world.script.at(ist(14, 43, 30), world.fake.faults.feed_dead)
    world.script.at(ist(14, 43, 52), world.fake.faults.feed_alive)
    world.stop_at(ist(14, 46, 30))
    return world


def test_F40_a_dead_feed_is_alerted_once(dead_feed, caplog):
    run(dead_feed, HoldWithTarget)
    assert len(alerts(caplog, "feed_dead")) == 1


def test_F40_a_dead_feed_locks_entries_from_10_s_after_its_last_tick(dead_feed):
    outcome = run(dead_feed, HoldWithTarget)
    locks = [e for e in events(outcome, "lock") if e["reason"] == "feed_dead"]
    assert [e["ts_ns"] for e in locks] == [DEAD_AT]


def test_F40_the_lock_lifts_at_the_close_of_the_first_whole_slice_after_the_feed_returns(
    dead_feed,
):
    outcome = run(dead_feed, HoldWithTarget)
    unlocks = [e for e in events(outcome, "unlock") if e["reason"] == "feed_dead"]
    # The feed is back at 14:43:55: the first whole slice is 14:44, closing at 14:45:00.
    assert [e["ts_ns"] for e in unlocks] == [ist(14, 45)]


def test_a_feed_back_on_a_slice_open_lifts_the_lock_at_the_close_of_the_next_slice(world):
    # The feed dies at 14:43:30 and lives again at 14:43:58; its first tick back is a
    # trade at 14:44:00.000, taken by the loop at that instant. The 14:44 slice then only
    # seeds the bar builders and has no bar, so the first whole slice is 14:45.
    world.script.at(ist(14, 43, 30), world.fake.faults.feed_dead)
    world.script.at(ist(14, 43, 58), world.fake.faults.feed_alive)
    world.script.at(ist(14, 44), world.script.trade)
    world.stop_at(ist(14, 46, 30))
    outcome = run(world, Hold)
    assert [e["ts_ns"] for e in events(outcome, "feed_back")] == [ist(14, 44)]
    unlocks = [e for e in events(outcome, "unlock") if e["reason"] == "feed_dead"]
    assert [e["ts_ns"] for e in unlocks] == [ist(14, 46)]


def test_F40_quotes_are_read_every_2_s_while_the_feed_is_dead(dead_feed):
    run(dead_feed, HoldWithTarget)
    reads = [call.ts_ns for call in calls_between(dead_feed, "quotes", DEAD_AT, BACK_AT)]
    assert len(reads) >= 5
    assert {later - earlier for earlier, later in zip(reads, reads[1:], strict=False)} == {2 * NS}


def test_F40_a_target_reached_on_a_quote_exits_while_the_feed_is_dead(dead_feed):
    outcome = run(dead_feed, HoldWithTarget)
    sell = dead_feed.orders()[-1]
    assert sell.action == "SELL" and DEAD_AT < sell.placed_ns < BACK_AT
    trades = outcome.outputs["HoldWithTarget"]["trades"]
    assert (trades["exit_price_rupees"], trades["exit_reason"]) == ([506.0], ["leg_target"])


def test_F40_the_slice_of_the_dead_spell_makes_no_bar_and_is_marked_missing(dead_feed):
    outcome = run(dead_feed, HoldWithTarget)
    missing = [e["slice_open_ns"] for e in events(outcome, "missing")]
    assert missing == [ist(14, 43)]
    assert ist(14, 44) not in [close for close, _ in HoldWithTarget.seen]


def test_F40_the_first_bar_after_the_dead_spell_holds_none_of_its_volume(dead_feed):
    run(dead_feed, HoldWithTarget)
    # The 14:44 bar holds its own six trades of 10 units, none of the spell's.
    assert (ist(14, 45), 60.0) in HoldWithTarget.seen


def test_F40_no_candle_is_read_while_the_feed_is_dead(dead_feed):
    run(dead_feed, HoldWithTarget)
    assert calls_between(dead_feed, "history", DEAD_AT, BACK_AT) == []


def test_F40_the_decision_on_a_bar_holding_a_dead_minute_is_skipped(world):
    # A 5-minute strategy: the feed is dead from 14:47:55 (its last tick) to 14:49:15, so
    # the 14:48 and 14:49 slices have no bar when the 14:45 to 14:50 bar closes. (A
    # 1-minute strategy has no bar to decide on in a minute without data.)
    world.script.at(ist(14, 48), world.fake.faults.feed_dead)
    world.script.at(ist(14, 49, 10), world.fake.faults.feed_alive)
    world.stop_at(ist(14, 51))
    outcome = run(world, HoldFiveMinutes)
    assert [s.decided_at_ns for s in outcome.skipped_decisions] == [ist(14, 50)]


def test_a_symbol_that_ticks_on_raises_no_feed_dead_alert(world, caplog):
    world.stop_at(ist(14, 46))
    outcome = run(world, BuyAndHold)
    assert alerts(caplog, "feed_dead") == []
    assert events(outcome, "missing") == []


@pytest.fixture
def quiet(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """SBIN and INFY each held by a strategy; SBIN's four ticks from 14:43:35 to 14:44:05
    are lost while INFY ticks on; the session is stopped at 14:45."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.script.at(ist(14, 43, 30), lambda: world.fake.faults.tick_gap("SBIN", "NSE", 4))
    world.stop_at(ist(14, 45))
    return world


def test_a_quiet_symbol_with_a_position_gets_quotes_until_it_ticks_again(quiet):
    outcome = run(quiet, [HoldSbin, HoldInfy])
    asked = [(e["ts_ns"], e["instruments"]) for e in events(outcome, "quotes")]
    # SBIN's last tick is 14:43:25, its next 14:44:15.
    assert asked == [(ist(14, 43, 35), ["SBIN:NSE"]), (ist(14, 44, 15), [])]


def test_a_quiet_symbol_locks_no_entry(quiet, caplog):
    outcome = run(quiet, [HoldSbin, HoldInfy])
    assert [e for e in events(outcome, "lock") if e["reason"] == "feed_dead"] == []
    assert alerts(caplog, "feed_dead") == []


# -- F38: the broker's login expires while OpenAlgo still holds it -------------------------


@pytest.fixture
def expired(world):
    """The broker's login expires at 14:44:00 with 10 SBIN held and is restored at
    14:45:30; the session is stopped at 14:48."""
    world.script.at(ist(14, 44), world.fake.faults.expire_session)
    world.script.at(ist(14, 45, 30), world.fake.faults.restore_session)
    world.stop_at(ist(14, 48))
    return world


def test_F38_an_expired_login_kills_the_feed_and_locks_entries(expired, caplog):
    outcome = run(expired, Hold)
    assert len(alerts(caplog, "feed_dead")) == 1
    locks = [e for e in events(outcome, "lock") if e["reason"] == "feed_dead"]
    assert [e["ts_ns"] for e in locks] == [ist(14, 44, 5)]


def test_F38_refused_quotes_are_alerted_as_a_login_that_may_have_expired(expired, caplog):
    run(expired, Hold)
    errors = [text for text in alerts(caplog, "error") if "REST quotes of SBIN" in text]
    assert errors
    assert all("login may have expired" in text for text in errors)


def test_F38_the_refused_quotes_alert_is_repeated_at_most_once_a_minute(expired, caplog):
    run(expired, Hold)
    # The quotes fail every 2 s from 14:44:05 to the restore at 14:45:30.
    errors = [record for record in caplog.records if "REST quotes of SBIN" in record.getMessage()]
    alerted = [record for record in errors if record.getMessage().startswith("alert error")]
    assert len(alerted) == 2


def test_F38_after_the_login_is_restored_the_lock_lifts_a_whole_slice_later(expired):
    outcome = run(expired, Hold)
    unlocks = [e for e in events(outcome, "unlock") if e["reason"] == "feed_dead"]
    # The first tick after the restore is 14:45:35: the 14:46 slice closes at 14:47:00.
    assert [e["ts_ns"] for e in unlocks] == [ist(14, 47)]


def test_F38_the_session_ends_with_the_engine_and_the_broker_flat(expired, caplog):
    outcome = run(expired, Hold)
    assert expired.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    # Row F38's "reconciliation clean": no comparison with the broker's books differed,
    # and once the login is back the books are read and agree with the engine (unit 12).
    assert alerts(caplog, "reconcile_mismatch") == []
    assert outcome.reconciliations == ()
    after = [e["found"] for e in events(outcome, "reconcile") if e["ts_ns"] > ist(14, 45, 30)]
    assert after and set(after) == {"agree"}


def test_F38_a_square_off_refused_while_the_login_is_expired_is_retried_until_it_fills(world):
    world.script.at(ist(14, 58), world.fake.faults.expire_session)
    restored = ist(15, 0, 10)
    world.script.at(restored, world.fake.faults.restore_session)
    outcome = run(world, Hold)
    sells = [
        call
        for call in world.fake.log.calls("placeorder")
        if call.detail["request"].get("action") == "SELL"
    ]
    # The square-off and two retries are refused while the login is expired; the third
    # retry, after the restore, fills.
    assert [call.detail["status"] for call in sells] == [403, 403, 403, 200]
    assert sells[-1].ts_ns > restored
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "session_end"


# -- F39: OpenAlgo revokes the broker's login ------------------------------------------------


@pytest.fixture
def revoked(world):
    """OpenAlgo revokes the login at 14:45:30, after the 14:45:01 exit left the Flipper
    flat, and the user logs in again at 14:47:30; the session is stopped at 14:50:30."""
    world.script.at(ist(14, 45, 30), world.fake.faults.revoke_login)
    world.script.at(ist(14, 47, 30), world.fake.faults.restore_login)
    world.stop_at(ist(14, 50, 30))
    return world


def buys_placed(world, start_ns: int, end_ns: int) -> list[int]:
    return [
        order.placed_ns
        for order in world.orders()
        if order.action == "BUY" and start_ns <= order.placed_ns < end_ns
    ]


def test_F39_no_entry_is_sent_while_the_login_is_revoked(revoked):
    outcome = run(revoked, Flipper)
    assert buys_placed(revoked, ist(14, 45, 30), ist(14, 49)) == []
    assert "live_lock" in outcome.outputs["Flipper"]["orders"]["reject_code"]


def test_F39_the_guards_failed_reads_are_alerted_after_three(revoked, caplog):
    run(revoked, Flipper)
    errors = [text for text in alerts(caplog, "error") if "analyzerstatus 3 times" in text]
    assert len(errors) == 1


def test_F39_entries_resume_a_whole_slice_after_the_login_is_restored(revoked):
    run(revoked, Flipper)
    # The first tick after the restore is 14:47:35: entries unlock at 14:49:00, and the
    # Flipper buys at its decision of 14:49:01.
    assert len(buys_placed(revoked, ist(14, 49), ist(14, 49, 5))) == 1
