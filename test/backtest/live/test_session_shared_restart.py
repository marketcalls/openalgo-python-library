"""A restart of a session two strategies share an instrument in, on the fake OpenAlgo
(`_live/_restart_groups.py`, `_live/_restart_books.py`; 18-multi-strategy 4.10 as unit M6
needs it, and 4.14a item 3).

A broker nets the members of an instrument into one position, so a restart compares the
sum of the members' journal positions with the broker's net, never one member's with it:
a sum that agrees resumes every member with nothing locked; a broker position no member's
journal holds, or a journal the broker holds nothing of with no trade to explain it,
refuses the start naming the group; any other difference resumes every member with its
entries locked and its exits held by the exit guard (nothing is adopted before unit M7).

The market and the strategies are `shared_samples`': SBIN at 500.00; `OnceA` buys 100 and
`OnceB` 50 at their first decision (14:42:01 IST, A first), each with a stop 3 points below,
and `NeverB` never trades. The session crashes at 14:44:30 and starts again at once.
Expected values come from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import alerts, ist
from session_trace import OTHER_SIDE, WITHIN
from shared_samples import A_UNITS, B_UNITS, NeverB, OnceA, OnceB, sbin_orders, terms
from stops_samples import stop_world

from openalgo.backtest._live._session_news import RECONCILE_LOCK
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import LiveError

#: When the first session dies.
CRASHED = ist(14, 44, 30)
#: The units another client sells by hand while the session is down.
SOLD_BY_HAND = 60


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


def test_a_restart_whose_sum_the_broker_holds_fewer_of_locks_every_member(world, caplog):
    # A's 100 and B's 50 keep their stops at the broker while it holds 90: the session keeps
    # the stops as they are, as for a leg the books show fewer units of (5.5; the cut of
    # kept stops is unit M7's).
    kept = "18-multi-strategy 4.7: the cut of kept stops after a sale by hand is unit M7's"
    world.trace.waive(WITHIN, kept)
    world.trace.waive(OTHER_SIDE, kept)
    run_to_crash(world, [OnceA, OnceB])
    sell_by_hand(world, SOLD_BY_HAND)
    outcome = restart(world, [OnceA, OnceB], stop_ns=ist(14, 47), on_stop="keep")

    net = A_UNITS + B_UNITS - SOLD_BY_HAND
    rows = [(r.strategy, r.engine_units, r.broker_units) for r in outcome.reconciliations]
    assert rows[:2] == [("OnceA", A_UNITS, net), ("OnceB", B_UNITS, net)]
    assert {("OnceA", RECONCILE_LOCK), ("OnceB", RECONCILE_LOCK)} <= set(locks(outcome))
    # The exit guard holds every member's exits while the group differs.
    guarded = [(e["strategy"], e["broker_units"]) for e in events(outcome, "exit_guard")]
    assert guarded[:2] == [("OnceA", net), ("OnceB", net)]
    assert [o for o in world.orders() if o.placed_ns >= CRASHED and o.strategy != "manual"] == []
