"""A closing order rejected on a shared instrument, on the fake OpenAlgo
(`_live/_margin_reject.py`; 18-multi-strategy 4.14 as the lead decided it in 11.0a items
1 and 2, 4.14a, rows F117 and F118, unit M6).

In a group of two or more members any rejection of a closing order holds its position:
no market exit, no stop placed again, no exit retry and no bare-leg protect; the member's
entries are locked, a book read is asked for at once and one ERROR alert says what to do.
Every automatic exit or protect of the held position waits for a book read that agrees
with the group, then passes the exit guard; the session's own square-off still closes it.
A strategy alone on its instrument keeps 2.3.0's path (the exit retries).

The market and the strategies are `shared_samples`': SBIN at 500.00, both buy at 14:42:01
IST, A first. Expected values come from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import NS, alerts, ist
from shared_samples import (
    A_UNITS,
    B_UNITS,
    OnceA,
    OnceB,
    SbinBreakout,
    SbinTrend,
    sbin_orders,
    terms,
)
from stops_samples import STOP_POINTS, StopHold, at_price, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._margin_reject import MARGIN_LOCK
from openalgo.backtest._live._session_news import RECONCILE_LOCK
from openalgo.backtest._live._stops_cover import CLOSE_REJECTED, REFUSED_FOR_MARGIN
from openalgo.backtest._live._types import LOGGER_NAME

#: The broker's words of the rejections (18-multi-strategy F118).
MARGIN_WORDS = "RMS: margin exceeds"
#: The square-off of NSE's MIS positions.
SQUARE_OFF = ist(15, 0)
#: When another client cancels B's stop, so the kernel places it again (R3).
CANCELLED_AT = ist(14, 44)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


def stop_placed_again_and_rejected(world, words: str = MARGIN_WORDS) -> None:
    """Another client cancels B's stop at `CANCELLED_AT`; the kernel places it again (R3)
    and the broker rejects that one right after accepting it, for `words`."""
    other = world.fake.client()

    def cancel_bs_stop():
        world.fake.reject_next(words, price_type="SL-M")
        stop = next(o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "SL-M")
        other.cancelorder(order_id=stop.order_id, strategy="SbinBreakout")

    world.script.at(CANCELLED_AT, cancel_bs_stop)


def test_F118_a_stop_rejected_on_a_shared_instrument_is_neither_exited_nor_placed_again(world):
    stop_placed_again_and_rejected(world)
    world.run([SbinTrend, SbinBreakout])

    orders = sbin_orders(world, "SbinBreakout")
    assert terms(orders) == [
        ("BUY", B_UNITS, "MARKET", "complete"),
        ("SELL", B_UNITS, "SL-M", "cancelled"),
        ("SELL", B_UNITS, "SL-M", "rejected"),
        # The session's square-off, once a read agreed with the group: no exit before it.
        ("SELL", B_UNITS, "MARKET", "complete"),
    ]
    assert SQUARE_OFF <= orders[-1].placed_ns < SQUARE_OFF + 10 * NS
    assert world.fake.position("SBIN", "NSE") == 0


def test_F118_the_held_member_is_locked_and_alerted_once_and_a_read_is_asked_at_once(world, caplog):
    stop_placed_again_and_rejected(world)
    outcome = world.run([SbinTrend, SbinBreakout])

    (held,) = events(outcome, "margin_reject")
    assert (held["strategy"], held["reason"]) == ("SbinBreakout", MARGIN_WORDS)
    locks = [(e["strategy"], e["reason"]) for e in events(outcome, "lock")]
    assert ("SbinBreakout", MARGIN_LOCK) in locks
    assert ("SbinTrend", MARGIN_LOCK) not in locks
    reads = [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]
    assert any(held["ts_ns"] <= read <= held["ts_ns"] + NS for read in reads)
    words = [t for t in alerts(caplog, "error") if "refused for margin by the broker" in t]
    assert len(words) == 1
    assert "SbinBreakout" in words[0] and MARGIN_WORDS in words[0]
    assert "end the session, check the position book and close by hand" in words[0]


def test_F118_the_held_leg_counts_unprotected_for_the_margin_refusal(world):
    stop_placed_again_and_rejected(world)
    outcome = world.run([SbinTrend, SbinBreakout])

    reasons = [w.reason for w in outcome.unprotected if w.strategy == "SbinBreakout"]
    assert REFUSED_FOR_MARGIN in reasons
    window = next(w for w in outcome.unprotected if w.reason == REFUSED_FOR_MARGIN)
    # From the rejection to B's square-off fill.
    assert CANCELLED_AT < window.from_ns < CANCELLED_AT + 5 * NS
    assert SQUARE_OFF <= window.to_ns < SQUARE_OFF + 10 * NS


def test_F118_the_other_member_is_untouched(world):
    stop_placed_again_and_rejected(world)
    world.run([SbinTrend, SbinBreakout])

    a_orders = sbin_orders(world, "SbinTrend")
    assert terms(a_orders) == [
        ("BUY", A_UNITS, "MARKET", "complete"),
        ("SELL", A_UNITS, "SL-M", "cancelled"),
        ("SELL", A_UNITS, "MARKET", "complete"),
    ]
    assert all(o.placed_ns < CANCELLED_AT or o.placed_ns >= SQUARE_OFF for o in a_orders)


def test_F117_any_rejection_of_a_stop_on_a_shared_instrument_is_held_and_worded_rejected(
    world, caplog
):
    # 11.0a item 2: the phrase table only words the alert; "trigger invalid" holds too.
    stop_placed_again_and_rejected(world, "trigger invalid")
    outcome = world.run([SbinTrend, SbinBreakout])

    sells = [o for o in sbin_orders(world, "SbinBreakout") if o.action == "SELL"]
    market = [o for o in sells if o.price_type == "MARKET"]
    assert [o.placed_ns >= SQUARE_OFF for o in market] == [True]
    reasons = [w.reason for w in outcome.unprotected if w.strategy == "SbinBreakout"]
    assert CLOSE_REJECTED in reasons
    words = [t for t in alerts(caplog, "error") if "was rejected by the broker" in t]
    assert len(words) == 1 and "trigger invalid" in words[0]


class TargetB(StopHold):
    """B of 50 units with a stop 3 points below and a target 2 points above."""

    qty = B_UNITS
    rules = {"stop": bt.points(STOP_POINTS), "target": bt.points(2)}


class StopA(StopHold):
    """A of 100 units with a stop 3 points below and no target."""

    qty = A_UNITS


def target_exit_rejected(world, *, at_placement: bool) -> None:
    """B's target is reached at 14:44 (SBIN at 502.00); the broker rejects B's exit."""
    at_price(world, ist(14, 44), 502.0)
    world.script.at(
        ist(14, 43, 59),
        lambda: world.fake.reject_next(
            MARGIN_WORDS, price_type="MARKET", at_placement=at_placement
        ),
    )


@pytest.mark.parametrize("at_placement", [False, True], ids=["after_acceptance", "at_placement"])
def test_an_exit_rejected_on_a_shared_instrument_is_not_asked_for_again(world, at_placement):
    target_exit_rejected(world, at_placement=at_placement)
    outcome = world.run([StopA, TargetB])

    # The rejected exit, then nothing until the session's square-off: no retry at 1, 2, 5
    # and 10 s, and no stop placed again for the bare leg.
    exits = [
        call.ts_ns
        for call in world.fake.log.calls("placeorder")
        if call.detail["request"]["strategy"] == "TargetB"
        and (call.detail["request"]["action"], call.detail["request"]["pricetype"])
        == ("SELL", "MARKET")
    ]
    assert [ts >= SQUARE_OFF for ts in exits] == [False, True]
    stops = [o for o in sbin_orders(world, "TargetB") if o.price_type == "SL-M"]
    assert len(stops) == 1
    assert [e["strategy"] for e in events(outcome, "exit_rejected_held")] == ["TargetB"]
    assert events(outcome, "exit_retried") == []


class TargetAlone(TargetB):
    """`TargetB` as the only strategy on SBIN."""


def test_a_strategy_alone_keeps_the_exit_retries_for_the_same_rejection(world):
    target_exit_rejected(world, at_placement=False)
    outcome = world.run(TargetAlone)

    exits = [o for o in world.orders() if o.price_type == "MARKET" and o.action == "SELL"]
    # A group of one is today's session: the exit is asked for again a second later.
    assert [o.status for o in exits] == ["rejected", "complete"]
    assert NS <= exits[1].placed_ns - exits[0].placed_ns <= 3 * NS
    assert events(outcome, "margin_reject") == []


def test_a_held_positions_exit_waits_for_an_agreeing_read_then_goes(world):
    stop_placed_again_and_rejected(world)
    # The position book cannot be read for a while after the rejection, so no comparison
    # agrees until 14:44:30 or so; Ctrl+C comes at 14:44:10.
    unreadable = world.fake.faults.gateway_error
    world.script.at(CANCELLED_AT, lambda: [unreadable("positionbook") for _ in range(8)])
    world.stop_at(ist(14, 44, 10))
    outcome = world.run([SbinTrend, SbinBreakout])

    held = [e for e in events(outcome, "held_for_read") if e["strategy"] == "SbinBreakout"]
    assert [(e["call"], e["ts_ns"]) for e in held] == [("operator_stop", ist(14, 44, 10))]
    (agreed,) = events(outcome, "margin_hold_agreed")
    exits = [o for o in sbin_orders(world, "SbinBreakout") if o.price_type == "MARKET"]
    exits = [o for o in exits if o.action == "SELL"]
    # B's exit, asked by the safe stop, went once a read agreed, within the stop's minute.
    assert len(exits) == 1 and agreed["ts_ns"] <= exits[0].placed_ns < ist(14, 45, 10)
    assert (world.fake.position("SBIN", "NSE"), outcome.stopped_by) == (0, "ctrl_c")


def test_a_held_leg_left_open_at_the_end_is_listed_with_its_reason(world):
    stop_placed_again_and_rejected(world)
    unreadable = world.fake.faults.gateway_error
    world.script.at(CANCELLED_AT, lambda: [unreadable("positionbook") for _ in range(20)])
    world.trace.left_unprotected_by_decision(
        "SbinBreakout",
        "18-multi-strategy 4.14: B's stop, rejected on a shared instrument, is placed again "
        "only after a book read agrees, and none can be read before the session ends",
    )
    world.stop_at(ist(14, 44, 10))
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")

    (b,) = [p for p in outcome.open_positions if p.strategy == "SbinBreakout"]
    assert (b.units, b.stop_confirmed) == (B_UNITS, False)
    assert b.note.startswith(f"{REFUSED_FOR_MARGIN}: the broker may not hold these units")


def restart_read_before_bs_order(world, restarted: int) -> bool:
    """Whether the restarted session read the position book before it placed any order of
    OnceB (by the order of the fake's log)."""
    calls = [e for e in world.fake.log.calls() if e.ts_ns >= restarted]
    first_read = next(i for i, e in enumerate(calls) if e.what == "positionbook")
    placed = next(
        i
        for i, e in enumerate(calls)
        if e.what == "placeorder" and e.detail["request"]["strategy"] == "OnceB"
    )
    return first_read < placed


def test_after_a_restart_the_hold_is_gone_and_the_restart_decides_first(world, monkeypatch):
    crashing(monkeypatch)
    other = world.fake.client()

    def cancel_bs_stop():
        world.fake.reject_next(MARGIN_WORDS, price_type="SL-M")
        stop = next(o for o in sbin_orders(world, "OnceB") if o.price_type == "SL-M")
        other.cancelorder(order_id=stop.order_id, strategy="OnceB")

    world.script.at(CANCELLED_AT, cancel_bs_stop)
    # The crash comes after the 14:45 slice's save, which knows the rejected stop ended.
    restarted = ist(14, 45, 30)
    crash_at(world, restarted)
    run_to_crash(world, [OnceA, OnceB])
    outcome = restart(world, [OnceA, OnceB], stop_ns=ist(14, 47), on_stop="keep")

    # The hold is not saved: the restart's comparison runs, then its protect places B's
    # stop again before trading resumes; nothing else is sent for B.
    # The day's event stream holds both runs: the rejection is the first run's.
    (held,) = events(outcome, "margin_reject")
    assert held["ts_ns"] < restarted
    assert [e for e in events(outcome, "held_for_read") if e["ts_ns"] >= restarted] == []
    after = [o for o in sbin_orders(world, "OnceB") if o.placed_ns >= restarted]
    assert terms(after) == [("SELL", B_UNITS, "SL-M", "trigger pending")]
    assert restart_read_before_bs_order(world, restarted)
    # Nor is the hold's entry lock set again: the restart's comparison decides afresh, and
    # with the group's sum agreeing with the broker's net it finds nothing to lock.
    locks = [
        (e["strategy"], e["reason"]) for e in events(outcome, "lock") if e["ts_ns"] >= restarted
    ]
    assert ("OnceB", MARGIN_LOCK) not in locks
    assert [lock for lock in locks if lock[1] == RECONCILE_LOCK] == []
    assert outcome.reconciliations == ()


def test_a_rejection_the_restart_learns_from_the_journal_holds_the_position_again(
    world, monkeypatch
):
    crashing(monkeypatch)
    other = world.fake.client()

    def cancel_bs_stop():
        world.fake.reject_next(MARGIN_WORDS, price_type="SL-M")
        stop = next(o for o in sbin_orders(world, "OnceB") if o.price_type == "SL-M")
        other.cancelorder(order_id=stop.order_id, strategy="OnceB")

    world.script.at(CANCELLED_AT, cancel_bs_stop)
    # The crash comes before any save after the rejection: the saved state still holds the
    # stop placed again, and the restart applies its REJECTED word from the journal.
    restarted = ist(14, 44, 30)
    crash_at(world, restarted)
    run_to_crash(world, [OnceA, OnceB])
    outcome = restart(world, [OnceA, OnceB], stop_ns=ist(14, 47), on_stop="keep")

    again = [e for e in events(outcome, "margin_reject") if e["ts_ns"] >= restarted]
    assert [e["strategy"] for e in again] == ["OnceB"]
    sells = [o for o in sbin_orders(world, "OnceB") if o.placed_ns >= restarted]
    # No market exit (R3 does not apply to a held position): the stop is placed again by
    # the restart's own protect, which its comparison of the books came before.
    assert terms(sells) == [("SELL", B_UNITS, "SL-M", "trigger pending")]
    assert restart_read_before_bs_order(world, restarted)
