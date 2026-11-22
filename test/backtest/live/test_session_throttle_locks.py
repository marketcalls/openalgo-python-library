"""An entry or add the order rate throttle holds in the worker's queue is never sent once
its strategy's entries lock (18-multi-strategy 2.5a rules 5, 6 and 12): the kernel cancels
it, and the worker withdraws it from its queue instead of placing it when the window frees.

The market is `session_samples`': SBIN and INFY on NSE on Monday 2026-11-16 from 14:40:05
IST, trading 10 units every 10 s at 500.00. `InfyAdds` and `SbinAdds` buy 10 units at
their first decision (14:42:01) and add 10 at their second (14:43:01), in that call order.
With at most one order call a second, INFY's add takes the window at 14:43:01 and SBIN's
waits in the queue. A trade of SBIN at 508.00 half a second later puts `SbinAdds` at
10 x 8.00 = +80.00 while its add still waits; each test locks its entries then. Nothing
contacts a server. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, ist, rig
from watch_samples import events, run

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

#: When SBIN's add waits for the window, and the trade at `RISEN` that locks its entries.
ADD_DECIDED = ist(14, 43, 1)
LOCKED_AT = ADD_DECIDED + 500_000_000
RISEN = 508.0


class SbinAdds(bt.Strategy):
    """Buys 10 SBIN at its first decision and adds 10 at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.decisions = 0

    def on_bar(self, bar):
        self.decisions += 1
        if self.decisions in (1, 2):
            self.buy(10)


class InfyAdds(SbinAdds):
    """Buys 10 INFY at its first decision and adds 10 at its second."""

    symbol = "INFY"


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.script.at(LOCKED_AT, lambda: world.fake.trade("SBIN", "NSE", RISEN, volume=10.0))
    world.stop_at(ist(14, 46))
    return world


def sbin_buys_after(world, instant_ns: int) -> list:
    return [
        o
        for o in world.orders()
        if o.symbol == "SBIN" and o.action == "BUY" and o.placed_ns > instant_ns
    ]


def withdrawn(outcome, reason: str = "cancelled") -> list[tuple[str, str]]:
    """(strategy, reason) of each request the worker did not send for `reason`."""
    return [
        (e["strategy"], e["reason"]) for e in events(outcome, "not_sent") if e["reason"] == reason
    ]


def test_an_add_held_by_the_order_rate_is_not_sent_after_the_profit_target_locks(world):
    outcome = run(
        world,
        [InfyAdds, SbinAdds],
        daily_profit_target={"InfyAdds": 100_000, "SbinAdds": 50},
        max_orders_per_second=1,
        on_stop="keep",
    )
    assert sbin_buys_after(world, LOCKED_AT) == []
    assert world.fake.position("SBIN", "NSE") == 0
    assert withdrawn(outcome) == [("SbinAdds", "cancelled")]


def test_an_add_held_by_the_order_rate_is_not_sent_after_the_account_kill_switch(world):
    outcome = run(
        world,
        [InfyAdds, SbinAdds],
        account_daily_profit_target=50,
        max_orders_per_second=1,
        on_stop="keep",
    )
    assert sbin_buys_after(world, LOCKED_AT) == []
    sbin_sells = [o for o in world.orders() if o.symbol == "SBIN" and o.action == "SELL"]
    assert [(o.quantity, o.filled) for o in sbin_sells] == [(10, 10)]
    assert (world.fake.position("SBIN", "NSE"), world.fake.position("INFY", "NSE")) == (0, 0)
    assert withdrawn(outcome) == [("SbinAdds", "cancelled")]


def test_an_add_held_by_the_order_rate_is_not_sent_after_a_feed_dead_lock(world):
    # 2.4.0 item M8-X2. The last tick comes at 14:42:51.5, so the feed is dead (10 s with no
    # tick on any symbol) and every strategy's entries lock at `LOCKED_AT`, while SBIN's add
    # still waits for the window: no kernel cancel withdraws it, the lock must.
    def last_tick_then_dead():
        world.fake.trade("SBIN", "NSE", 500.0, volume=10.0)
        world.fake.faults.feed_dead()

    world.script.at(LOCKED_AT - 10 * NS, last_tick_then_dead)
    outcome = run(world, [InfyAdds, SbinAdds], max_orders_per_second=1, on_stop="keep")
    assert sbin_buys_after(world, LOCKED_AT) == []
    assert world.fake.position("SBIN", "NSE") == 10
    assert withdrawn(outcome, "entries_locked") == [("SbinAdds", "entries_locked")]


def test_the_add_withdrawn_for_the_lock_is_rejected_with_the_locks_code(world):
    def last_tick_then_dead():
        world.fake.trade("SBIN", "NSE", 500.0, volume=10.0)
        world.fake.faults.feed_dead()

    world.script.at(LOCKED_AT - 10 * NS, last_tick_then_dead)
    outcome = run(world, [InfyAdds, SbinAdds], max_orders_per_second=1, on_stop="keep")
    orders = outcome.outputs["SbinAdds"]["orders"]
    codes = [code for code in orders["reject_code"] if code is not None]
    # The code an entry submitted while entries are locked gets (the kernel's `live_lock`).
    assert codes == ["live_lock"]


def test_an_add_held_by_the_order_rate_is_not_sent_after_a_safe_stop(world):
    # Ctrl+C at the instant of the trade: the safe stop locks entries and cancels the
    # working ones, and with on_stop="keep" leaves the 10 units of each strategy open.
    world.stop_at(LOCKED_AT)
    outcome = run(world, [InfyAdds, SbinAdds], max_orders_per_second=1, on_stop="keep")
    assert sbin_buys_after(world, LOCKED_AT) == []
    assert world.fake.position("SBIN", "NSE") == 10
    assert withdrawn(outcome) == [("SbinAdds", "cancelled")]
