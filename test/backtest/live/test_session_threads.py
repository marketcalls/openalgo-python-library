"""A live session on its real threads while the order worker is blocked in a call, on the
fake OpenAlgo (`_live/session.py`; 16-session-design 2.1, 2.4, 2.9 and 7.2 rows F63 and
F64, unit 8, run without leg rules).

These tests run the session's own threads (`threads=True`): the loop on the test's
thread, the order worker, the data worker and the alert sender on theirs. The fake's
clock runs `SPEED` times faster than the wall clock, moved on by a driver thread that
also plays the script (the trades, and the release of a blocked call), so a fake minute
lasts a couple of real seconds. A client double holds one call of the order worker on
an event until the script releases it.

Two strategies trade: `Early` buys 10 SBIN at its first decision (14:42:01 IST) and
sells them at its second (14:43:01); `Late` buys 10 INFY at its second decision
(14:43:01). The session starts at 14:40:05 and is stopped with Ctrl+C at 14:47. The
start runs on the wall clock too, so the first decision is read from what happened.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist, rig

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.session import run_session

#: Fake seconds per real second.
SPEED = 30
#: The loop's longest wait, real seconds: a fraction of a fake second.
WAIT_MAX_S = 0.01
#: Real seconds a blocked call waits at most, so a broken test cannot hang.
BLOCK_LIMIT_S = 20.0


class Early(bt.Strategy):
    """Buys SBIN at its first decision, sells at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2 and self.position.qty:
            self.exit()


class Late(bt.Strategy):
    """Buys INFY at its second decision; keeps (bar close, the fake's time) of each."""

    symbol, exchange, interval = "INFY", "NSE", "1m"
    decided: list[tuple[int, int]] = []
    clock: Any = None

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        type(self).decided.append((bar.ts_close, type(self).clock()))
        if self.calls == 2:
            self.buy(10)


class Blocking:
    """A client that holds its first call of `method` made once `when()` is true for
    `hold_s` fake seconds (by `now`), then makes it."""

    def __init__(self, client: Any, method: str, *, when: Any, hold_s: float, now: Any) -> None:
        self._client, self._method, self._when = client, method, when
        self._hold_ns, self._now = round(hold_s * NS), now
        self.held_at: int | None = None
        self.freed = threading.Event()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if name != self._method:
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            if self.held_at is None and self._when():
                self.held_at = self._now()
                limit = time.perf_counter() + BLOCK_LIMIT_S
                while self._now() < self.held_at + self._hold_ns and not self.freed.is_set():
                    if time.perf_counter() > limit:
                        break
                    time.sleep(0.002)
            return attribute(*args, **kwargs)

        return call


class Driver(threading.Thread):
    """Moves the fake's clock on in step with the wall clock, `SPEED` times faster,
    playing the script on the way."""

    def __init__(self, script: Any) -> None:
        super().__init__(name="test-clock", daemon=True)
        self._script = script
        self.done = threading.Event()

    def run(self) -> None:
        last = time.perf_counter()
        while not self.done.is_set():
            time.sleep(0.002)
            now = time.perf_counter()
            self._script.sleep((now - last) * SPEED)
            last = now


def run_threaded(world: Any, blocking: Blocking, role: str) -> Any:
    """Run `Early` and `Late` with the session's threads, `blocking` standing in for the
    session's `role` client."""
    clients = world.clients()
    clients = type(clients)(**{**clients.__dict__, role: blocking})
    clock = world.fake.clock

    def wait(seconds: float) -> None:
        until = clock.now() + round(seconds * NS)
        while clock.now() < until:
            time.sleep(0.001)

    driver = Driver(world.script)
    driver.start()
    try:
        return run_session(
            [Early, Late],
            world.fake.client(),
            clients=clients,
            feed=world.fake.direct_feed(),
            now=clock.now,
            sleep=wait,
            threads=True,
            signals=False,
            stop=world.stop,
            emit=world.lines.append,
            capital=100_000,
            max_daily_loss=5_000,
            alerts_to="trader1",
            live_dir=world.folder / "live",
            wait_max_s=WAIT_MAX_S,
        )
    finally:
        blocking.freed.set()
        driver.done.set()
        driver.join(5)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(
        make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 5), symbols=("SBIN", "INFY")
    )
    made.stop_at(ist(14, 47))
    Late.decided, Late.clock = [], made.fake.clock.now
    return made


def first_decision(world) -> int:
    """The close of the strategies' first decision."""
    return min(close for close, _ in Late.decided)


def test_F63_while_an_order_call_hangs_slices_close_and_decisions_are_made_on_time(world):
    blocking = Blocking(
        world.fake.client(), "placeorder", when=lambda: True, hold_s=150, now=world.fake.clock.now
    )
    outcome = run_threaded(world, blocking, "orders")
    first = first_decision(world)
    assert blocking.held_at is not None and blocking.held_at - first < 5 * NS
    late = dict(Late.decided)
    # Decided within a few fake seconds of each close while the call was held.
    for close in (first + 60 * NS, first + 120 * NS):
        assert late[close] - close < 5 * NS
    buys = [order for order in world.orders() if order.action == "BUY"]
    assert buys[0].symbol == "SBIN"
    assert buys[0].placed_ns >= blocking.held_at + 150 * NS
    # Late's entry, decided while the call was held, waited too long and was dropped.
    assert list(outcome.outputs["Late"]["orders"]["status"]) == ["Rejected"]


def test_F64_an_entry_queued_behind_a_blocked_worker_expires_and_the_exit_goes(world, caplog):
    fake = world.fake

    def a_while_after_the_first_fill() -> bool:
        fills = fake.fills()
        return bool(fills) and fake.clock.now() >= fills[0].ts_ns + 50 * NS

    blocking = Blocking(
        fake.client(),
        "analyzerstatus",
        when=a_while_after_the_first_fill,
        hold_s=30,
        now=fake.clock.now,
    )
    outcome = run_threaded(world, blocking, "guard")
    sells = [order for order in world.orders() if order.action == "SELL"]
    assert sells[0].symbol == "SBIN"
    assert sells[0].placed_ns >= blocking.held_at + 30 * NS
    assert [order.symbol for order in world.orders() if order.action == "BUY"] == ["SBIN"]
    assert list(outcome.outputs["Late"]["orders"]["status"]) == ["Rejected"]
    assert any("stale" in text for text in alerts(caplog, "reject"))
