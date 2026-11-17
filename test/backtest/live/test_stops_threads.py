"""The stops supervisor while the order worker is blocked in a call, on the session's real
threads and the fake OpenAlgo (`_live/stops.py`; 16-session-design 4.2, 4.3 and 7.2 rows
F71 and F92).

As in `test_session_threads`: the fake's clock runs `SPEED` times faster than the wall
clock, moved on by a driver thread that plays the script, and a client double holds one
call of the worker for a span of fake seconds. The start runs on the sped-up clock too,
so each hold is triggered by what happened (a stop placed), never by a time of day.
`StopHold` buys 10 SBIN at its first decision with a stop 3 points below
(`stops_samples`); the session is stopped with Ctrl+C at 14:47 IST, which flattens.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist, rig
from stops_samples import STOP_POINTS, StopHold, Timeline, check_invariants, events
from test_session_threads import BLOCK_LIMIT_S, run_threaded

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME


class Holds:
    """A client that holds its first call of `method` for which `when(kwargs)` is true,
    for `hold_s` fake seconds (by `now`), running `on_hold` as the hold begins; every
    call of `method` runs `before(kwargs)` first when given."""

    def __init__(
        self,
        client: Any,
        method: str,
        *,
        when: Any,
        hold_s: float,
        now: Any,
        on_hold: Any = None,
        before: Any = None,
    ) -> None:
        self._client, self._method, self._when = client, method, when
        self._hold_ns, self._now = round(hold_s * NS), now
        self._on_hold, self._before = on_hold, before
        self.held_at: int | None = None
        self.freed = threading.Event()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if name != self._method:
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            if self._before is not None:
                self._before(kwargs)
            if self.held_at is None and self._when(kwargs):
                self.held_at = self._now()
                if self._on_hold is not None:
                    self._on_hold()
                limit = time.perf_counter() + BLOCK_LIMIT_S
                while self._now() < self.held_at + self._hold_ns and not self.freed.is_set():
                    if time.perf_counter() > limit:
                        break
                    time.sleep(0.002)
            return attribute(*args, **kwargs)

        return call


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 5))
    made.timeline = Timeline(made.fake)
    made.stop_at(ist(14, 47))
    return made


def stop_order(world) -> Any:
    """The fake's stop order, or None before one is placed."""
    return next((o for o in world.fake.orders() if o.price_type == "SL-M"), None)


def test_F92_no_deadline_exit_while_the_worker_is_in_a_call_with_the_confirmation_waiting(
    world, caplog
):
    # The stop's working word is held at the fake from its placement; the worker's first
    # read of the stop (its poll, 2 s after it was sent) then hangs for 30 s, and the
    # held word reaches the worker's queue as the hang begins: well past the 10 s
    # deadline, the confirmation waits behind a busy worker.
    fake = world.fake

    def hold_the_stops_word(kwargs):
        if kwargs.get("price_type") == "SL-M":
            fake.faults.hold_order_updates()

    placing = Holds(
        fake.client(),
        "placeorder",
        when=lambda kwargs: False,
        hold_s=0,
        now=fake.clock.now,
        before=hold_the_stops_word,
    )
    stop_polled = Holds(
        placing,
        "orderstatus",
        when=lambda kwargs: (
            stop_order(world) is not None and kwargs.get("order_id") == stop_order(world).order_id
        ),
        hold_s=30,
        now=fake.clock.now,
        on_hold=fake.faults.release_order_updates,
    )
    outcome = run_threaded(world, stop_polled, "orders", (StopHold,))

    assert stop_polled.held_at is not None
    # The only cancel is the Ctrl+C's at 14:47: no exit for an unconfirmed stop.
    assert all(call.ts_ns >= ist(14, 47) for call in fake.log.calls("cancelorder"))
    assert events(outcome, "stop_unconfirmed") == []
    assert not any("not confirmed working" in text for text in alerts(caplog, "error"))
    # The stop's 10 s deadline fell inside the 30 s hang.
    stop = stop_order(world)
    assert stop.placed_ns + 10 * NS < stop_polled.held_at + 30 * NS
    assert stop_polled.held_at < stop.placed_ns + 10 * NS
    check_invariants(world.timeline, strategy="StopHold")


class LimitTrail(bt.Strategy):
    """Buys 10 SBIN with a LIMIT at 500.00 at its first decision, with a stop 3 points
    below that trails 3 points behind the best price."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(
                10, price=500.0, stop=bt.points(STOP_POINTS), trail=bt.trail_points(STOP_POINTS)
            )


def test_F71_a_quantity_change_behind_a_queued_trailing_modify_goes_after_it_in_order(world):
    # The entry fills 4 units as it arrives, a stop for 4 is placed, and the price stays
    # above the LIMIT. Then a guard read hangs for 30 s: meanwhile the price rises to
    # 505.00 (the trail wants the stop at 502.00) and the rest of the entry fills.
    fake = world.fake
    fake.split_next_fill(4)
    rose = threading.Event()

    def price(now_ns: int) -> float:
        if rose.is_set():
            return 505.0
        return 501.0 if fake.fills() else 500.0

    world.script.price = price

    def the_rest_fills_and_the_price_rises():
        rose.set()
        entry = next(o for o in fake.orders() if o.action == "BUY")
        fake.fill(entry.order_id, price=500.0)

    guard_hangs = Holds(
        fake.client(),
        "analyzerstatus",
        when=lambda kwargs: stop_order(world) is not None,
        hold_s=30,
        now=fake.clock.now,
        on_hold=the_rest_fills_and_the_price_rises,
    )
    outcome = run_threaded(world, guard_hangs, "guard", (LimitTrail,))

    assert guard_hangs.held_at is not None
    modifies = [
        (int(c.detail["request"]["quantity"]), float(c.detail["request"]["trigger_price"]))
        for c in fake.log.calls("modifyorder")
    ]
    assert modifies[:2] == [(4, 502.0), (10, 502.0)]
    stop = stop_order(world)
    assert (stop.quantity, stop.trigger_price) == (10, 502.0)
    assert fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"
    check_invariants(world.timeline, strategy="LimitTrail")
