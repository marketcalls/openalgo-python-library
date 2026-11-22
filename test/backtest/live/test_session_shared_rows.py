"""The group rows of the sum rule on the fake OpenAlgo (`_live/_reconcile_groups.py`,
`_live/_reconcile_act.py`, `_live/_reconcile_findings.py`; 18-multi-strategy 4.6, 4.7 and
4.11, rows F98 to F102 of 5.2, unit M7).

A broker nets both members into one position, so the session judges their sum (`E`)
against the broker's net (`B`). Each row acts when two comparisons in a row show it:

- **Absorbed** (F98, F100): other orders' trades explain the difference and the members on
  the side they close hold it: shared out, the most recently opened position first, each
  member adopting its share, its stop cut to what it is left with or cancelled.
- **Excess** (F101): the broker holds units beyond every member's position on their side:
  nothing is sent for them, every member's entries are locked, and the alert says the
  excess is no strategy's, to be closed by hand once the session has ended.
- **Other side** (F99): the broker holds the other side of the members' sum: nothing is
  adopted, every member's stop is cancelled, every member is locked, and one ERROR alert
  names the instrument.
- **Unexplained** (F102): no trade explains it: a lag of the broker's books, nothing done.

A difference of the group is one mismatch of the day, and each member's row of the day's
reconciliations carries the group, every member's units, the broker's net, the other
orders' units not taken in, and on the row of its act the allocation made and the stop it
cut.

The market and the strategies are `shared_samples`': SBIN at 500.00; A (`SbinTrend`) buys
100 at 14:42:01 IST, B 50 at 14:43:01 (`LateBreakout`), each with an SL-M stop 3 points
below (497.00); `ShortTrend` and `LateShort` sell instead, with SL-M buy stops at 503.00.
Another client trades SBIN by hand at 14:43:30. Expected values come from the fake's books
and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from shared_samples import (
    A_UNITS,
    B_UNITS,
    LateBreakout,
    LateShort,
    SbinBreakout,
    SbinTrend,
    ShortTrend,
    sbin_orders,
)
from stops_samples import stop_world

from openalgo.backtest._live._types import LOGGER_NAME

#: When another client trades by hand, and when the session is stopped (positions kept).
BY_HAND = ist(14, 43, 30)
STOPPED = ist(14, 45)
#: The members' sum of a long group (A and B).
HELD = A_UNITS + B_UNITS


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`, in order."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [event for event in lines if event["kind"] == kind]


def by_hand(world, action: str, units: int) -> None:
    """Another client trades `units` SBIN at market at `BY_HAND`."""
    other = world.fake.client()
    world.script.at(
        BY_HAND,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action=action,
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=units,
        ),
    )


def run(world, members, action: str, units: int):
    """`members` trade, another client trades `units` by hand, Ctrl+C at `STOPPED` keeps
    the positions: the outcome."""
    by_hand(world, action, units)
    world.stop_at(STOPPED)
    return world.run(members, on_stop="keep")


def acted(outcome) -> dict[str, object]:
    """Each strategy's row of its act (the second sighting), by name."""
    return {r.strategy: r for r in outcome.reconciliations if r.action != "noted"}


def stop_ids(world, *names: str) -> dict[str, str]:
    """Each strategy's first SL-M order at the fake, by name: its broker order ID."""
    return {
        name: next(o.order_id for o in sbin_orders(world, name) if o.price_type == "SL-M")
        for name in names
    }


def stop_terms(world, name: str) -> tuple[str, int, str]:
    """`name`'s last SL-M at the fake: action, units and status."""
    stop = [o for o in sbin_orders(world, name) if o.price_type == "SL-M"][-1]
    return stop.action, stop.quantity, stop.status


def placed_after_the_trade_by_hand(world) -> list:
    """The members' orders placed after the other client's trade."""
    return [o for o in world.orders() if o.placed_ns >= BY_HAND and o.strategy != "manual"]


# ----- F98: a sale by hand the members absorb ---------------------------------------------


def test_F98_a_sale_by_hand_is_shared_out_and_each_stop_cut_to_what_its_member_keeps(world):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    adopted = {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}
    assert adopted == {"LateBreakout": ("SELL", B_UNITS), "SbinTrend": ("SELL", 10)}
    assert stop_terms(world, "LateBreakout") == ("SELL", B_UNITS, "cancelled")
    assert stop_terms(world, "SbinTrend") == ("SELL", 90, "trigger pending")
    assert world.fake.position("SBIN", "NSE") == 90
    assert placed_after_the_trade_by_hand(world) == []


def test_F98_each_member_adopts_its_share_before_its_stop_is_cut(world):
    # 18-multi-strategy 4.14b (lead's ruling): the adoption comes first, and the kernel's cut
    # of the stop follows in the same correction.
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    adoptions = [e["ts_ns"] for e in events(outcome, "adopted")]
    cuts = [c.ts_ns for name in ("modifyorder", "cancelorder") for c in world.fake.log.calls(name)]
    assert adoptions and cuts and max(adoptions) <= min(cuts)


def test_F98_both_members_are_locked_and_the_alert_names_the_allocation(world, caplog):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    locks = {(e["strategy"], e["reason"]) for e in events(outcome, "lock")}
    assert {("SbinTrend", "reconciliation"), ("LateBreakout", "reconciliation")} <= locks
    texts = alerts(caplog, "reconcile_mismatch")
    assert any(
        "60 units sold by other orders are shared out, the most recently opened position "
        "first: taken from LateBreakout 50 (flat) and SbinTrend 10 (90 left)" in text
        for text in texts
    )


def test_F98_each_members_row_carries_the_group_the_allocation_and_its_own_stop_cut(world):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    stops = stop_ids(world, "SbinTrend", "LateBreakout")
    rows = acted(outcome)
    for name, units in (("SbinTrend", A_UNITS), ("LateBreakout", B_UNITS)):
        row = rows[name]
        assert (row.kind, row.engine_units, row.broker_units) == ("explained_shared", units, 90)
        assert (row.group, row.group_units) == ("SbinTrend, LateBreakout", HELD)
        assert row.member_units == f"SbinTrend {A_UNITS}, LateBreakout {B_UNITS}"
        assert row.external_units == -60
        assert row.allocation == "LateBreakout -50 to 0, SbinTrend -10 to 90"
    assert rows["SbinTrend"].stops_cut == f"stop {stops['SbinTrend']} cut to 90 units"
    assert rows["LateBreakout"].stops_cut == f"stop {stops['LateBreakout']} cancelled"


def test_F98_the_first_sighting_carries_the_group_but_no_allocation_yet(world):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    noted = [r for r in outcome.reconciliations if r.action == "noted"]
    assert {r.strategy for r in noted} == {"SbinTrend", "LateBreakout"}
    assert {(r.group_units, r.external_units, r.allocation, r.stops_cut) for r in noted} == {
        (HELD, -60, "", "")
    }


def test_F98_a_difference_of_the_group_is_one_mismatch_of_the_day(world):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 60)
    assert outcome.summary.reconcile_mismatches == 1


# ----- F100: a buy by hand on a short group -----------------------------------------------


def test_F100_a_buy_by_hand_on_a_short_group_is_taken_from_the_short_opened_last(world):
    # B sold its 50 after A sold its 100: the 30 bought by hand come off B, its stop cut to
    # the 20 it keeps; A and its stop are untouched.
    outcome = run(world, [ShortTrend, LateShort], "BUY", 30)
    adopted = {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}
    assert adopted == {"LateShort": ("BUY", 30)}
    assert stop_terms(world, "LateShort") == ("BUY", 20, "trigger pending")
    assert stop_terms(world, "ShortTrend") == ("BUY", A_UNITS, "trigger pending")
    assert world.fake.position("SBIN", "NSE") == -(A_UNITS + B_UNITS) + 30
    rows = acted(outcome)
    assert rows["ShortTrend"].allocation == "LateShort +30 to -20"
    assert rows["ShortTrend"].stops_cut == ""


# ----- F101: a buy by hand on an all-long group: an excess --------------------------------


def test_F101_an_excess_on_an_all_long_group_sends_nothing_and_locks_every_member(world):
    outcome = run(world, [SbinTrend, LateBreakout], "BUY", 10)
    assert placed_after_the_trade_by_hand(world) == []
    assert world.fake.position("SBIN", "NSE") == HELD + 10
    locks = {(e["strategy"], e["reason"]) for e in events(outcome, "lock")}
    assert {("SbinTrend", "reconciliation"), ("LateBreakout", "reconciliation")} <= locks
    assert events(outcome, "adopted") == []
    assert {(r.strategy, r.kind) for r in acted(outcome).values()} == {
        ("SbinTrend", "more"),
        ("LateBreakout", "more"),
    }


def test_F101_the_excess_is_named_no_strategys_to_close_by_hand_once_the_session_ends(
    world, caplog
):
    run(world, [SbinTrend, LateBreakout], "BUY", 10)
    texts = alerts(caplog, "reconcile_mismatch")
    assert len(texts) == 2
    for text in texts:
        assert f"OpenAlgo's books show {HELD + 10} for them together" in text
        assert (
            "the 10 units beyond what they hold are no strategy's of the session: nothing is "
            "sent for them and they have no stop at the broker. End the session, then close "
            "them by hand in OpenAlgo"
        ) in text
        assert "the excess is not the engine's" not in text


def test_F101_an_excess_does_not_guard_the_members_exits_and_keeps_every_stop(world):
    outcome = run(world, [SbinTrend, LateBreakout], "BUY", 10)
    assert events(outcome, "exit_guard") == []
    assert stop_terms(world, "SbinTrend")[2] == "trigger pending"
    assert stop_terms(world, "LateBreakout")[2] == "trigger pending"
    assert outcome.summary.reconcile_mismatches == 1


# ----- F99: a sale by hand beyond the members: the other side -----------------------------


def test_F99_the_other_side_cancels_every_members_stop_and_adopts_nothing(world):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 200)
    assert stop_terms(world, "SbinTrend") == ("SELL", A_UNITS, "cancelled")
    assert stop_terms(world, "LateBreakout") == ("SELL", B_UNITS, "cancelled")
    assert events(outcome, "adopted") == []
    assert placed_after_the_trade_by_hand(world) == []
    assert world.fake.position("SBIN", "NSE") == HELD - 200
    locks = {(e["strategy"], e["reason"]) for e in events(outcome, "lock")}
    assert {("SbinTrend", "reconciliation"), ("LateBreakout", "reconciliation")} <= locks


def test_F99_no_sell_stop_works_once_the_other_side_is_acted_on(world):
    # After the act the stops' sum on the sell side may be no more than the broker's long
    # units, none: from the first cancel on, no SL-M of a member is placed or raised.
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 200)
    (first_act,) = {r.at_ns for r in acted(outcome).values()}
    cancels = [c.ts_ns for c in world.fake.log.calls("cancelorder")]
    assert len(cancels) == 2 and min(cancels) >= first_act
    assert [c for c in world.fake.log.calls("modifyorder")] == []


def test_F99_one_error_alert_names_the_instrument_and_each_row_its_stop_cancelled(world, caplog):
    outcome = run(world, [SbinTrend, LateBreakout], "SELL", 200)
    (error,) = [t for t in alerts(caplog, "error") if "the other side" in t]
    assert error.endswith(
        f": SBIN NSE: OpenAlgo's books show {HELD - 200} units, the other side of the {HELD} "
        "units SbinTrend and LateBreakout hold together: end the session, then close the "
        "position by hand in OpenAlgo"
    )
    stops = stop_ids(world, "SbinTrend", "LateBreakout")
    rows = acted(outcome)
    for name in ("SbinTrend", "LateBreakout"):
        assert (rows[name].kind, rows[name].action) == ("other_side", "stop cancelled")
        assert rows[name].stops_cut == f"stop {stops[name]} cancelled"
        assert rows[name].allocation == ""
    assert outcome.summary.reconcile_mismatches == 1


def test_F99_each_members_alert_says_nothing_is_adopted_and_to_end_the_session(world, caplog):
    run(world, [SbinTrend, LateBreakout], "SELL", 200)
    texts = alerts(caplog, "reconcile_mismatch")
    assert len(texts) == 2
    for text in texts:
        assert (
            "the broker holds the other side of what they hold together: nothing is adopted, "
            "each member's stop is cancelled and nothing else is sent. End the session, then "
            "close the position by hand in OpenAlgo"
        ) in text


# ----- F102: a lagging position book ------------------------------------------------------


def test_F102_a_lagging_book_on_a_shared_instrument_is_alerted_as_a_lag_and_nothing_done(
    world, caplog
):
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    world.stop_at(STOPPED)
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")
    assert {r.kind for r in outcome.reconciliations} == {"unexplained"}
    assert events(outcome, "adopted") == []
    assert world.fake.log.calls("cancelorder") == []
    assert world.fake.log.calls("modifyorder") == []
    texts = alerts(caplog, "reconcile_mismatch")
    assert texts and all("taken as a lag of the broker's books" in text for text in texts)
    assert outcome.summary.reconcile_mismatches == 1
