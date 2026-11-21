"""The fault row of a restart of a shared instrument whose members trail their stops, on the
fake OpenAlgo (18-multi-strategy 5.2 row F109, "trailing resumes per member", 4.10; unit
M8): after a crash each member's restored stop trails again from its own kernel, by its
own distance, with no second entry.

The market is `stops_samples`': SBIN at 500.00 from 14:40:05 IST, 10 units every 10 s.
`TrailingA` buys 100 units and `TrailingB` 50 at their first decision (14:42:01, A first),
once a day, each with a stop that trails 3 and 4 points below the best price. The session
crashes at 14:44:30 and starts again at once; SBIN then rises 0.50 a trade from 14:45 to
14:47 (13 rises, to 506.50) and holds; Ctrl+C at 14:48 keeps the positions. Every test
runs the session invariants, those of groups and those restated per member among them.
Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import NS, TRADE_EVERY_S, ist
from shared_samples import A_UNITS, B_UNITS, sbin_orders
from stops_samples import stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

CRASHED = ist(14, 44, 30)
RISE_FROM, RISE_UNTIL, STEP = ist(14, 45), ist(14, 47), 0.5
#: The best price: 500.00 and 13 rises of 0.50 (`rising`).
BEST = 506.5
#: A broker's trigger is modified only after a move of at least 0.25 percent
#: (16-session-design R6): the trigger may lag the kernel's level by less than that.
MODIFY_STEP = 0.0025


class TrailingA(bt.Strategy):
    """Buys `qty` SBIN once a day (it remembers it in `self.state`) with a stop `distance`
    points below that trails the best price by as much."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty, distance = A_UNITS, 3.0

    def on_bar(self, bar):
        if not self.state.get("bought") and not self.position.qty:
            self.state["bought"] = True
            kind = type(self)
            self.buy(kind.qty, stop=bt.points(kind.distance), trail=bt.trail_points(kind.distance))


class TrailingB(TrailingA):
    """B: 50 units trailing 4 points below."""

    qty, distance = B_UNITS, 4.0


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    crash_at(world, CRASHED)
    return world


def rising(world) -> None:
    """SBIN rises by `STEP` each trade from `RISE_FROM` until `RISE_UNTIL`, then holds."""

    def price(now_ns: int) -> float:
        if now_ns < RISE_FROM:
            return 500.0
        trades = (min(now_ns, RISE_UNTIL) - RISE_FROM) // (TRADE_EVERY_S * NS)
        return round(500.0 + STEP * (trades + 1), 2)

    world.script.price = price


def run_restarted(world):
    run_to_crash(world, [TrailingA, TrailingB])
    rising(world)
    return restart(world, [TrailingA, TrailingB], stop_ns=ist(14, 48), on_stop="keep")


def test_F109_each_members_stop_trails_again_after_the_restart_by_its_own_distance(world):
    outcome = run_restarted(world)

    for name, distance in (("TrailingA", 3.0), ("TrailingB", 4.0)):
        (stop,) = [o for o in sbin_orders(world, name) if o.price_type == "SL-M"]
        modified = [
            c
            for c in world.fake.log.calls("modifyorder")
            if c.ts_ns > CRASHED and c.detail["request"].get("orderid") == stop.order_id
        ]
        assert modified and {c.detail["request"]["strategy"] for c in modified} == {name}
        level = BEST - distance
        assert 0 <= level - stop.trigger_price < level * MODIFY_STEP
        # The broker's trigger is the one the member's kernel holds for its stop.
        orders = outcome.outputs[name]["orders"]
        kernel = [
            t
            for t, kind in zip(orders["trigger_rupees"], orders["order_type"], strict=True)
            if kind == "SL-M"
        ]
        assert kernel[-1] == stop.trigger_price


def test_F109_no_member_enters_again_after_the_restart(world):
    run_restarted(world)

    assert [o for o in world.orders() if o.placed_ns >= CRASHED] == []
    assert world.fake.position("SBIN", "NSE") == A_UNITS + B_UNITS
