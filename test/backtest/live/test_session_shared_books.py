"""A sale by hand on an instrument two strategies of the session share, on the fake
OpenAlgo (`_live/_reconcile_groups.py`, `_live/_reconcile_guard.py`, `_live/_kill_switch.py`;
18-multi-strategy 4.6, 4.8, 4.9 and 4.11 as unit M6 builds them).

A broker nets both members into one position, so the session judges their sum against the
broker's net, never one member against it: a sale by hand that leaves the net below the
sum guards every member (every closing order held until the books agree), each alert and
note names the group's net and sum, and nothing is adopted before unit M7's allocation,
though the other client's trade explains it. The kill switch exits a member only when the
broker holds the group's sum.

The market and the strategies are `shared_samples`': SBIN at 500.00, A (`SbinTrend`) buys
100 and B (`SbinBreakout`) 50 at 14:42:01 IST, each with a stop 3 points below. Expected
values come from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from session_trace import OTHER_SIDE, WITHIN
from shared_samples import A_UNITS, B_UNITS, SbinBreakout, SbinTrend
from stops_samples import at_price, stop_world

from openalgo.backtest._live._types import LOGGER_NAME

#: When another client sells by hand, and how many units.
SOLD_AT = ist(14, 43, 30)
SOLD = 60
#: The broker's net after the sale.
NET = A_UNITS + B_UNITS - SOLD
#: Why the checks of a stop beyond the position are waived: the cut of kept stops after a
#: sale by hand on a shared instrument is unit M7's (18-multi-strategy 4.7).
KEPT = "18-multi-strategy 4.7: the cut of kept stops after a sale by hand is unit M7's"


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive(WITHIN, KEPT)
    world.trace.waive(OTHER_SIDE, KEPT)
    return world


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]


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


def run_sold(world):
    """The sale by hand, then Ctrl+C at 14:45 with the positions kept: the outcome."""
    sell_by_hand(world, SOLD_AT, SOLD)
    world.stop_at(ist(14, 45))
    return world.run([SbinTrend, SbinBreakout], on_stop="keep")


def test_a_sale_by_hand_guards_every_member_with_the_brokers_net(world):
    outcome = run_sold(world)
    guarded = [(e["strategy"], e["broker_units"]) for e in events(outcome, "exit_guard")]
    assert guarded[:2] == [("SbinTrend", NET), ("SbinBreakout", NET)]
    # No closing order of either member goes while the group differs.
    after = [o for o in world.orders() if o.placed_ns > SOLD_AT and o.strategy != "manual"]
    assert after == []
    assert world.fake.position("SBIN", "NSE") == NET


def test_each_open_position_names_the_groups_net_at_the_last_read(world):
    outcome = run_sold(world)
    rows = {p.strategy: p for p in outcome.open_positions}
    assert (rows["SbinTrend"].units, rows["SbinBreakout"].units) == (A_UNITS, B_UNITS)
    for row in rows.values():
        assert (
            f"OpenAlgo's books showed {NET} units of SBIN for SbinTrend and SbinBreakout "
            "together at the last read"
        ) in row.note


def test_the_mismatch_names_the_groups_sum_and_adopts_what_explains_it_for_no_member(world, caplog):
    outcome = run_sold(world)
    kinds = {(r.strategy, r.kind, r.engine_units, r.broker_units) for r in outcome.reconciliations}
    assert kinds == {
        ("SbinTrend", "explained_shared", A_UNITS, NET),
        ("SbinBreakout", "explained_shared", B_UNITS, NET),
    }
    assert events(outcome, "adopted") == []
    texts = alerts(caplog, "reconcile_mismatch")
    (b,) = [text for text in texts if "SbinBreakout SBIN NSE:" in text]
    assert (
        f"the engine holds {B_UNITS} units, {A_UNITS + B_UNITS} with SbinTrend's, OpenAlgo's "
        f"books show {NET} for them together"
    ) in b
    assert "trades of other orders explain it" in b
    assert not any("lag of the broker's books" in text for text in texts)


def test_the_kill_switch_exits_no_member_while_the_broker_holds_less_than_their_sum(world):
    # The fall to 497.40 at 14:44 takes the account's loss past its limit at the first
    # trade there (14:44:05); a second before, another client sold 30 by hand, so the kill
    # switch's read shows 120 against the members' 150: each member alone would agree
    # with it, and their exits together would sell 30 more than the broker holds.
    at_price(world, ist(14, 44), 497.4)
    sell_by_hand(world, ist(14, 44, 4), 30)
    world.stop_at(ist(14, 48))
    outcome = world.run([SbinTrend, SbinBreakout], account_max_daily_loss=300, on_stop="keep")

    assert {e["strategy"] for e in events(outcome, "kill_switch")} == {"SbinTrend", "SbinBreakout"}
    assert events(outcome, "kill_exit") == []
    assert world.fake.position("SBIN", "NSE") == A_UNITS + B_UNITS - 30
