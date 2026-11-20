"""A sale by hand on an instrument two strategies of the session share, on the fake
OpenAlgo (`_live/_reconcile_groups.py`, `_live/_reconcile_allocate.py`,
`_live/_reconcile_guard.py`, `_live/_kill_switch.py`; 18-multi-strategy 4.6 to 4.9 and
4.11).

A broker nets both members into one position, so the session judges their sum against the
broker's net, never one member against it. A sale by hand the other client's trade
explains guards every member (every closing order held until the books agree) from the
first comparison that shows it; the second shares it out (4.7): the member whose position
opened most recently takes first, up to its units, then the next, each adopting its share
as its own, so that each member's stop is cut to what the member is left with and the
stops kept working never cover more than the broker holds. A difference no trade explains
adopts nothing, keeps every stop and names the group's net. The kill switch exits a member
only when the broker holds the group's sum: after the allocation, each member what it
holds.

The market and the strategies are `shared_samples`': SBIN at 500.00; A (`SbinTrend`) buys
100 at 14:42:01 IST, and B 50 at 14:42:01 too (`SbinBreakout`) or at 14:43:01
(`LateBreakout`), each with a stop 3 points below (an SL-M at 497.00). Expected values come
from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from shared_samples import A_UNITS, B_UNITS, LateBreakout, SbinBreakout, SbinTrend, terms
from stops_samples import at_price, stop_world

from openalgo.backtest._live._types import LOGGER_NAME

#: When another client sells by hand, and how many units.
SOLD_AT = ist(14, 43, 30)
SOLD = 60
#: The broker's net after the sale.
NET = A_UNITS + B_UNITS - SOLD
#: Where SBIN trades from 14:46: below both stops (497.00).
FALLEN = 496.0
#: Why a sale by hand first seen by the close's final read leaves each member's stop at the
#: broker for more than the net (16-session-design 3.3 step 5 and 5.4 "As built"): no later
#: read confirms it, so nothing is adopted and no stop is cut; the close names each stop.
KEPT_AT_THE_CLOSE = (
    "3.3 step 5 and 5.4 As built: the close does not fit a stop to a sale by hand it has "
    "not settled, and names it"
)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def events(outcome, kind: str | None) -> list[dict]:
    """The run's event-stream lines of `kind` (all of them for None), in order."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [event for event in lines if kind is None or event["kind"] == kind]


def sell_by_hand(world, at_ns: int, units: int) -> None:
    """Another client sells `units` SBIN at market at `at_ns`."""
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=units,
        ),
    )


def run_sold(
    world,
    b=LateBreakout,
    *,
    sold: int = SOLD,
    sold_at: int = SOLD_AT,
    stop_ns: int = ist(14, 45),
):
    """The sale by hand of `sold` units at `sold_at` beside A and `b`, then Ctrl+C at
    `stop_ns` with the positions kept: the outcome."""
    sell_by_hand(world, sold_at, sold)
    world.stop_at(stop_ns)
    return world.run([SbinTrend, b], on_stop="keep")


def stops_at_the_fake(world) -> dict[str, tuple[str, int, str, str]]:
    """Each strategy's last SL-M at the fake, as `shared_samples.terms` gives it."""
    stops = [o for o in world.orders() if o.price_type == "SL-M"]
    return {o.strategy: terms([o])[0] for o in stops}


def test_a_sale_by_hand_guards_every_member_with_the_brokers_net(world):
    outcome = run_sold(world)
    guarded = [(e["strategy"], e["broker_units"]) for e in events(outcome, "exit_guard")]
    assert guarded[:2] == [("SbinTrend", NET), ("LateBreakout", NET)]
    # No closing order of either member is placed: the stops are cut, nothing is sold.
    after = [o for o in world.orders() if o.placed_ns > SOLD_AT and o.strategy != "manual"]
    assert after == []
    assert world.fake.position("SBIN", "NSE") == NET


def test_the_member_whose_position_opened_last_takes_a_sale_by_hand_first(world):
    # B opened at 14:43:01, after A: it takes 50 of the 60 sold, all it holds, and A the
    # 10 left.
    outcome = run_sold(world)
    adopted = {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}
    assert adopted == {"LateBreakout": ("SELL", B_UNITS), "SbinTrend": ("SELL", SOLD - B_UNITS)}
    # A's row, and the instrument's row with OpenAlgo's net (4.11).
    assert {p.strategy: p.units for p in outcome.open_positions} == {
        "SbinTrend": NET,
        "SbinTrend and LateBreakout": NET,
    }


def test_each_members_stop_is_cut_to_what_the_allocation_leaves_it(world):
    run_sold(world)
    assert stops_at_the_fake(world) == {
        "SbinTrend": ("SELL", NET, "SL-M", "trigger pending"),
        "LateBreakout": ("SELL", B_UNITS, "SL-M", "cancelled"),
    }


def test_members_whose_positions_opened_together_are_taken_in_call_order(world):
    # A and B both bought at 14:42:01: A, first in call order, takes the 60.
    outcome = run_sold(world, SbinBreakout)
    adopted = {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}
    assert adopted == {"SbinTrend": ("SELL", SOLD)}
    assert stops_at_the_fake(world) == {
        "SbinTrend": ("SELL", A_UNITS - SOLD, "SL-M", "trigger pending"),
        "SbinBreakout": ("SELL", B_UNITS, "SL-M", "trigger pending"),
    }


@pytest.mark.parametrize(
    ("b", "sold"),
    [(LateBreakout, SOLD), (SbinBreakout, SOLD), (LateBreakout, A_UNITS + B_UNITS)],
    ids=["60 sold, B opened last", "60 sold, opened together", "all 150 sold"],
)
def test_a_fall_through_the_stops_after_a_sale_by_hand_leaves_the_broker_flat(world, b, sold):
    # The stops kept working cover what the broker holds after the sale, never more: when
    # SBIN falls through them they sell it to flat, not through zero.
    at_price(world, ist(14, 46), FALLEN)
    run_sold(world, b, sold=sold, stop_ns=ist(14, 48))
    assert world.fake.position("SBIN", "NSE") == 0


def test_each_members_mismatch_alert_names_the_groups_sum_and_the_allocation(world, caplog):
    outcome = run_sold(world)
    kinds = {(r.strategy, r.kind, r.engine_units, r.broker_units) for r in outcome.reconciliations}
    assert kinds == {
        ("SbinTrend", "explained_shared", A_UNITS, NET),
        ("LateBreakout", "explained_shared", B_UNITS, NET),
    }
    texts = alerts(caplog, "reconcile_mismatch")
    (b,) = [text for text in texts if "LateBreakout SBIN NSE:" in text]
    assert (
        f"the engine holds {B_UNITS} units, {A_UNITS + B_UNITS} with SbinTrend's, OpenAlgo's "
        f"books show {NET} for them together"
    ) in b
    assert (
        f"{SOLD} units sold by other orders are shared out, the most recently opened position "
        f"first: taken from LateBreakout {B_UNITS} (flat) and SbinTrend {SOLD - B_UNITS} "
        f"({NET} left)"
    ) in b
    assert not any("lag of the broker's books" in text for text in texts)


def test_a_member_the_allocation_takes_nothing_from_is_locked_and_told_so(world, caplog):
    outcome = run_sold(world, SbinBreakout)
    texts = alerts(caplog, "reconcile_mismatch")
    (b,) = [text for text in texts if "SbinBreakout SBIN NSE:" in text]
    assert f"taken from SbinTrend {SOLD} ({A_UNITS - SOLD} left); SbinBreakout keeps its" in b
    locks = {(e["strategy"], e["reason"]) for e in events(outcome, "lock")}
    assert {("SbinTrend", "reconciliation"), ("SbinBreakout", "reconciliation")} <= locks


def test_a_sale_first_seen_by_the_closes_read_lists_each_member_at_its_own_units(world):
    # Ctrl+C at 14:45, two seconds after the sale: the close's final read is the first to
    # see it, and no later read confirms it. OpenAlgo's figure is the group's net, so each
    # member is listed at its own units, its note naming the net and the members' sum.
    for member in ("SbinTrend", "SbinBreakout"):
        world.trace.kept_stop_by_decision(member, KEPT_AT_THE_CLOSE)
    outcome = run_sold(world, SbinBreakout, sold_at=ist(14, 44, 58))
    rows = {p.strategy: p for p in outcome.open_positions}
    instrument = rows.pop("SbinTrend and SbinBreakout")
    assert {name: row.units for name, row in rows.items()} == {
        "SbinTrend": A_UNITS,
        "SbinBreakout": B_UNITS,
    }
    for row in rows.values():
        assert (
            f"OpenAlgo's books showed {NET} units of SBIN for SbinTrend and SbinBreakout "
            f"together, who held {A_UNITS + B_UNITS}, at the last read"
        ) in row.note
    # The instrument's own row: OpenAlgo's net, against the members' sum (4.11).
    assert (instrument.units, instrument.stop_confirmed) == (NET, False)
    assert f"their sum is {A_UNITS + B_UNITS}, so they differ" in instrument.note
    assert events(outcome, "adopted") == []


def test_a_sale_of_every_unit_first_seen_by_the_closes_read_lists_no_position(world, caplog):
    # The net the close's read saw is flat: nothing is left to close by hand, and each
    # member's alert says so, naming the stops still working.
    for member in ("SbinTrend", "SbinBreakout"):
        world.trace.kept_stop_by_decision(member, KEPT_AT_THE_CLOSE)
    outcome = run_sold(world, SbinBreakout, sold=A_UNITS + B_UNITS, sold_at=ist(14, 44, 58))
    assert outcome.open_positions == ()
    texts = alerts(caplog, "error")
    (b,) = [text for text in texts if "SbinBreakout SBIN NSE:" in text]
    assert "OpenAlgo's books showed no position of SBIN for SbinTrend and SbinBreakout" in b
    assert "no position is left to close by hand" in b


def test_a_difference_no_trade_explains_keeps_every_stop_and_names_the_groups_net(world):
    # The position book stops showing fills from 14:41:30: it shows none of the members'
    # 150 units, and no trade of another order explains it, so nothing is adopted, every
    # stop works on, and each open position names the group's net at the last read.
    world.script.at(ist(14, 41, 30), world.fake.lag_positionbook)
    world.stop_at(ist(14, 45))
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")
    assert events(outcome, "adopted") == []
    rows = {p.strategy: p for p in outcome.open_positions}
    assert (rows["SbinTrend"].units, rows["SbinBreakout"].units) == (A_UNITS, B_UNITS)
    for row in rows.values():
        assert (
            "OpenAlgo's books showed 0 units of SBIN for SbinTrend and SbinBreakout together "
            "at the last read"
        ) in row.note
    assert {s[3] for s in stops_at_the_fake(world).values()} == {"trigger pending"}


def test_the_kill_switch_exits_each_member_what_it_holds_once_the_sale_is_shared_out(world):
    # The fall to 497.40 at 14:44 takes the account's loss past its limit at the first
    # trade there (14:44:05); a second before, another client sold 30 by hand, so the kill
    # switch's first reads show 120 against the members' 150 and send nothing. Once the
    # allocation takes the 30 from A (first in call order: both opened together), each
    # member's exit sells what it holds, 120 together, and the broker ends flat.
    at_price(world, ist(14, 44), 497.4)
    sell_by_hand(world, ist(14, 44, 4), 30)
    world.stop_at(ist(14, 48))
    outcome = world.run([SbinTrend, SbinBreakout], account_max_daily_loss=300, on_stop="keep")

    assert {e["strategy"] for e in events(outcome, "kill_switch")} == {"SbinTrend", "SbinBreakout"}
    kinds = [e["kind"] for e in events(outcome, None)]
    assert kinds.count("adopted") == 1
    assert kinds.index("adopted") < kinds.index("kill_exit")
    exits = events(outcome, "kill_exit")
    assert {e["strategy"]: e["units"] for e in exits} == {
        "SbinTrend": A_UNITS - 30,
        "SbinBreakout": B_UNITS,
    }
    assert world.fake.position("SBIN", "NSE") == 0
