"""A restart of a session two strategies share an instrument in, on the fake OpenAlgo
(`_live/_restart_groups.py`, `_live/_restart_books.py`; 18-multi-strategy 4.10 as unit M6
needs it, and 4.14a item 3).

A broker nets the members of an instrument into one position, so a restart compares the
sum of the members' journal positions with the broker's net, never one member's with it:
a sum that agrees resumes every member with nothing locked; a broker position no member's
journal holds, or a journal the broker holds nothing of with no trade to explain it,
refuses the start naming the group; any other difference resumes every member with its
entries locked. A sale by hand while the session was down, which the trade book explains,
is shared out among the members as in a session (4.7, `_reconcile_allocate`), the position
that opened most recently by the saved state first, each member adopting its share with
its stop cut to what is left.

The market and the strategies are `shared_samples`': SBIN at 500.00; `OnceA` buys 100 and
`OnceB` 50 at their first decision (14:42:01 IST, A first), `LateOnceB` 50 at its second
(14:43:01), each with a stop 3 points below, and `NeverB` never trades. The session
crashes at 14:44:30 and starts again at once. Expected values come from the fake's books
and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import alerts, ist
from shared_samples import A_UNITS, B_UNITS, NeverB, OnceA, OnceB, sbin_orders, terms
from stops_samples import STOP_POINTS, at_price, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._session_news import RECONCILE_LOCK
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import LiveError

#: When the first session dies.
CRASHED = ist(14, 44, 30)
#: The units another client sells by hand while the session is down, and what is left.
SOLD_BY_HAND = 60
NET = A_UNITS + B_UNITS - SOLD_BY_HAND


class LateOnceB(OnceB):
    """`OnceB` buying at its second decision of the day (14:43:01), so that its position
    opens after A's; never again (it counts its decisions in `self.state`)."""

    def on_bar(self, bar):
        self.state["decisions"] = self.state.get("decisions", 0) + 1
        if self.state["decisions"] == 2 and not self.position.qty:
            self.buy(type(self).qty, stop=bt.points(STOP_POINTS))


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    crash_at(world, CRASHED)
    return world


def events(outcome, kind: str) -> list[dict]:
    """The day's event-stream lines of `kind` written from the restart on."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [e for e in lines if e["kind"] == kind and e["ts_ns"] >= CRASHED]


def locks(outcome) -> list[tuple[str, str]]:
    return [(e["strategy"], e["reason"]) for e in events(outcome, "lock")]


def adopted(outcome) -> dict[str, tuple[str, int]]:
    """Each strategy's adoption from the restart on: (side, units)."""
    return {e["strategy"]: (e["side"], e["units"]) for e in events(outcome, "adopted")}


def stops(world, strategy: str) -> list[tuple[str, int, str, str]]:
    """`strategy`'s SL-M orders at the fake, oldest first."""
    return terms([o for o in sbin_orders(world, strategy) if o.price_type == "SL-M"])


def sell_by_hand(world, units: int) -> None:
    """Another client sells `units` SBIN at market, while the session is down."""
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=units,
    )


def test_a_restart_resumes_a_shared_instrument_one_member_of_which_is_flat(world):
    run_to_crash(world, [OnceA, NeverB])
    outcome = restart(world, [OnceA, NeverB], stop_ns=ist(14, 47), on_stop="keep")

    assert outcome.reconciliations == ()
    assert locks(outcome) == []
    # A resumes with its stop, and nothing new is sent.
    assert [o for o in world.orders() if o.placed_ns >= CRASHED] == []
    assert terms(sbin_orders(world, "OnceA"))[-1] == ("SELL", A_UNITS, "SL-M", "trigger pending")


def test_a_restart_resumes_both_members_when_their_sum_is_the_brokers_net(world, caplog):
    run_to_crash(world, [OnceA, OnceB])
    outcome = restart(world, [OnceA, OnceB], stop_ns=ist(14, 47), on_stop="keep")

    assert world.fake.position("SBIN", "NSE") == A_UNITS + B_UNITS
    assert outcome.reconciliations == ()
    assert [lock for lock in locks(outcome) if lock[1] == RECONCILE_LOCK] == []
    assert alerts(caplog, "reconcile_mismatch") == []
    assert [o for o in world.orders() if o.placed_ns >= CRASHED] == []


def test_a_broker_position_no_member_of_the_group_holds_refuses_the_restart(world):
    run_to_crash(world, [NeverB, type("NeverA", (NeverB,), {})])
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=30,
    )
    with pytest.raises(LiveError, match="30 units of SBIN NSE, which NeverB and NeverA trade"):
        restart(world, [NeverB, type("NeverA", (NeverB,), {})], stop_ns=ist(14, 47))


def test_a_restart_shares_out_a_sale_by_hand_made_while_it_was_down(world):
    # A and B opened together (14:42:01): A, first in call order, takes the 60 sold while
    # the session was down, its stop cut to the 40 it is left with; B keeps its 50 and its
    # stop. Both are locked, and nothing new is placed.
    run_to_crash(world, [OnceA, OnceB])
    sell_by_hand(world, SOLD_BY_HAND)
    outcome = restart(world, [OnceA, OnceB], stop_ns=ist(14, 47), on_stop="keep")

    rows = [(r.strategy, r.kind, r.engine_units, r.broker_units) for r in outcome.reconciliations]
    assert rows == [
        ("OnceA", "explained_shared", A_UNITS, NET),
        ("OnceB", "explained_shared", B_UNITS, NET),
    ]
    assert {("OnceA", RECONCILE_LOCK), ("OnceB", RECONCILE_LOCK)} <= set(locks(outcome))
    assert adopted(outcome) == {"OnceA": ("SELL", SOLD_BY_HAND)}
    assert stops(world, "OnceA")[-1] == ("SELL", A_UNITS - SOLD_BY_HAND, "SL-M", "trigger pending")
    assert stops(world, "OnceB")[-1] == ("SELL", B_UNITS, "SL-M", "trigger pending")
    assert [o for o in world.orders() if o.placed_ns >= CRASHED and o.strategy != "manual"] == []
    assert world.fake.position("SBIN", "NSE") == NET


def test_a_restart_takes_a_sale_by_hand_from_the_position_its_saved_state_opened_last(world):
    # B's saved position opened at 14:43:01, after A's: it takes 50 of the 60 sold, all it
    # holds, its stop cancelled, and A the 10 left, its stop cut to 90.
    run_to_crash(world, [OnceA, LateOnceB])
    sell_by_hand(world, SOLD_BY_HAND)
    outcome = restart(world, [OnceA, LateOnceB], stop_ns=ist(14, 47), on_stop="keep")

    assert adopted(outcome) == {
        "LateOnceB": ("SELL", B_UNITS),
        "OnceA": ("SELL", SOLD_BY_HAND - B_UNITS),
    }
    assert stops(world, "OnceA")[-1] == ("SELL", NET, "SL-M", "trigger pending")
    assert stops(world, "LateOnceB")[-1] == ("SELL", B_UNITS, "SL-M", "cancelled")
    assert {p.strategy: p.units for p in outcome.open_positions} == {"OnceA": NET}


def test_after_a_restart_a_fall_through_the_stops_leaves_the_broker_flat(world):
    # The stops kept working after the restart cover the 90 the broker holds, never more.
    run_to_crash(world, [OnceA, OnceB])
    sell_by_hand(world, SOLD_BY_HAND)
    at_price(world, ist(14, 46), 496.0)
    restart(world, [OnceA, OnceB], stop_ns=ist(14, 48), on_stop="keep")
    assert world.fake.position("SBIN", "NSE") == 0
