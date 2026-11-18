"""A live session on the fake OpenAlgo, shared by the session's tests (`test_session*.py`;
`_live/session.py`, 16-session-design unit 8).

The market is `plan_samples`': SBIN on NSE on Monday 2026-11-16, its broker serving the
candles of the two sessions before and of today up to the session's start. From the start
on the test's `Script` plays the market on the fake's manual clock: SBIN trades every
`TRADE_EVERY_S` seconds at the price the script gives, and the actions a test schedules
(faults, a stop request, another client's order) run at their instants. The session
runs under its test seam (16-session-design 2.11): the fake's clients, its direct feed,
`threads=False`, no signal handlers, the script's `sleep` as every wait, so the whole
session runs on the test's thread and nothing waits for real. Nothing contacts a server.

Every rig records its fake's order changes and session runs (`Rig.trace`, `session_trace`),
and the session invariants of 16-session-design 7.1 run over them when the test ends
(`session_invariants`): every session test is checked. A test whose row leaves a leg
unprotected says so (`trace.leave_unprotected`, or `trace.left_unprotected_by_decision`
for a decided end); one whose other client sells by hand names what the decided end it
plays leaves at the fake (`trace.kept_stop_by_decision`, `trace.oversold_by_decision`);
one whose session cannot keep a check waives that check alone and says why
(`trace.waive`).

Expected values come from the fake's books and the scenario, never from the session's
code.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fake_openalgo import USERNAME, FakeOpenAlgo
from plan_samples import DAY, candles, ist, market
from session_hooks import trace_sessions
from session_invariants import assert_session_invariants
from session_trace import SessionTrace

import openalgo.backtest as bt
from openalgo.backtest._live._session_parts import StopFlag
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.clients import (
    ALERTS_CLIENT_TIMEOUT_S,
    DATA_CLIENT_TIMEOUT_S,
    GUARD_CLIENT_TIMEOUT_S,
    SessionClients,
)
from openalgo.backtest._live.session import run_session

NS = 10**9
#: Seconds between two trades of SBIN in the script.
TRADE_EVERY_S = 10
#: SBIN's price in the script, rupees.
PRICE = 500.0
#: The longest wait of the loop in these tests, seconds: the loop runs its duties at
#: least this often, which is what the tests' timings rest on.
WAIT_MAX_S = 1.0
#: The session's limits, rupees.
CAPITAL = 100_000
MAX_DAILY_LOSS = 5_000
#: The session's start in most tests: 14:40:05 IST, ten minutes before NSE's MIS entry
#: cut-off (14:50) and twenty before its square-off (15:00).
START = (14, 40, 5)


class Script:
    """The fake's time as a test plays it: `at` schedules an action at an instant (UTC
    nanoseconds), `trades` SBIN's trades; `sleep` (the session's wait) moves the fake's
    clock on, running each action due on the way at its instant."""

    def __init__(self, fake: FakeOpenAlgo) -> None:
        self.fake = fake
        self._actions: list[tuple[int, int, Callable[[], None]]] = []
        self._order = itertools.count()
        self.price: Callable[[int], float] = lambda now_ns: PRICE

    def at(self, instant_ns: int, action: Callable[[], None]) -> None:
        """Run `action` when the clock reaches `instant_ns`."""
        heapq.heappush(self._actions, (instant_ns, next(self._order), action))

    def trades(self, start_ns: int, end_ns: int, symbols: tuple[str, ...] = ("SBIN",)) -> None:
        """Each of `symbols` (on NSE) trades 10 units every `TRADE_EVERY_S` seconds from
        `start_ns` to `end_ns`, at `self.price(now)`."""
        for instant in range(start_ns, end_ns, TRADE_EVERY_S * NS):
            for symbol in symbols:
                self.at(instant, lambda symbol=symbol: self.trade(symbol))

    def trade(self, symbol: str = "SBIN") -> None:
        """One trade of 10 units of `symbol` on NSE now, at `self.price(now)`."""
        now_ns = self.fake.clock.now()
        self.fake.trade(symbol, "NSE", self.price(now_ns), volume=10.0)

    def sleep(self, seconds: float) -> None:
        """Move the fake's clock on by `seconds`, running the actions due meanwhile."""
        clock = self.fake.clock
        target = clock.now() + round(seconds * NS)
        while self._actions and self._actions[0][0] <= target:
            instant, _, action = heapq.heappop(self._actions)
            if instant > clock.now():
                clock.advance((instant - clock.now()) / NS)
            action()
        if target > clock.now():
            clock.advance((target - clock.now()) / NS)


@dataclass
class Rig:
    """A session's world: the fake, its script, the stop flag, the console's lines, and
    the trace the session invariants are checked over."""

    fake: FakeOpenAlgo
    script: Script
    folder: Any
    trace: SessionTrace
    stop: StopFlag = field(default_factory=StopFlag)
    lines: list[str] = field(default_factory=list)
    feed: Any = None

    def clients(self) -> SessionClients:
        """The session's five clients from the fake, with the timeouts `derive` gives."""
        made = {role: self.fake.client() for role in ("orders", "guard", "data", "alerts", "feed")}
        made["guard"].timeout = GUARD_CLIENT_TIMEOUT_S
        made["data"].timeout = DATA_CLIENT_TIMEOUT_S
        made["alerts"].timeout = ALERTS_CLIENT_TIMEOUT_S
        return SessionClients(**made)

    def run(self, strategy: Any, **arguments: Any) -> Any:
        """Run a session of `strategy` on the fake to its end, under the test seam."""
        self.feed = self.fake.direct_feed() if self.feed is None else self.feed
        limits = {
            "capital": CAPITAL,
            "max_daily_loss": MAX_DAILY_LOSS,
            "alerts_to": USERNAME,
            "live_dir": self.folder / "live",
        }
        return run_session(
            strategy,
            self.fake.client(),
            clients=self.clients(),
            feed=self.feed,
            now=self.fake.clock.now,
            sleep=self.script.sleep,
            threads=False,
            signals=False,
            stop=self.stop,
            emit=self.lines.append,
            wait_max_s=WAIT_MAX_S,
            **(limits | arguments),
        )

    def stop_at(self, instant_ns: int) -> None:
        """Ask the session to stop (Ctrl+C) at `instant_ns`."""
        self.script.at(instant_ns, self.stop.request)

    def orders(self) -> list[Any]:
        """The fake's orders of the selected book, oldest first."""
        return self.fake.orders()


def alerts(caplog: Any, kind: str) -> list[str]:
    """The texts of the session's alerts of `kind` ("reject", "error", ...), as the
    alerts log each one they queue."""
    tag = f"alert {kind}: "
    return [
        record.getMessage()[len(tag) :]
        for record in caplog.records
        if record.name == LOGGER_NAME and record.getMessage().startswith(tag)
    ]


def rig(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    start: tuple[int, int, int] = START,
    trades_until: tuple[int, int] | None = (15, 2),
    analyzer: bool = False,
    symbols: tuple[str, ...] = ("SBIN",),
) -> Rig:
    """A rig whose fake starts at the IST time `start` with the candles of `symbols` (on
    NSE) of today up to that minute, a first trade of each at the start itself (so the
    broker quotes them to the preflight), and trades from then to `trades_until` (None:
    none at all)."""
    at = ist(*start)
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=at, today_until=start[:2])
    for symbol in symbols[1:]:
        fake.add_instrument(symbol, "NSE")
        fake.seed_candles(symbol, "NSE", candles(today_until=start[:2]))
    if analyzer:
        fake.set_analyzer(True)
    script = Script(fake)
    if trades_until is not None:
        for symbol in symbols:
            script.trade(symbol)
        script.trades(at + TRADE_EVERY_S * NS, ist(*trades_until), symbols)
    trace = trace_sessions(fake, monkeypatch, assert_session_invariants)
    return Rig(fake, script, tmp_path, trace)


class BuyAndHold(bt.Strategy):
    """A 1-minute SBIN strategy: buys `qty` units at its first decision, sells them all
    after `hold` bars (0: never), then trades no more."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty = 10
    hold = 3

    def init(self):
        self.state.setdefault("bars", 0)
        self.decisions: list[int] = []
        self.done = False

    def on_bar(self, bar):
        self.decisions.append(bar.ts_close)
        if self.done:
            return
        if not self.position.qty:
            if not self.decisions[:-1]:
                self.buy(self.qty)
            return
        self.state["bars"] += 1
        if self.hold and self.state["bars"] >= self.hold:
            self.exit()
            self.done = True


class Hold(BuyAndHold):
    """Buys at its first decision and never sells (the square-off or a stop does)."""

    hold = 0


class BuyWhenFlat(bt.Strategy):
    """Buys `qty` units at every decision at which it holds none."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty = 10

    def on_bar(self, bar):
        if not self.position.qty:
            self.buy(self.qty)


class Raises(bt.Strategy):
    """Buys at its first decision, raises at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
            self.state["bought"] = True
        else:
            raise RuntimeError("the strategy's own error")


__all__ = [
    "CAPITAL",
    "DAY",
    "MAX_DAILY_LOSS",
    "NS",
    "PRICE",
    "START",
    "TRADE_EVERY_S",
    "WAIT_MAX_S",
    "BuyAndHold",
    "Hold",
    "StopFlag",
    "alerts",
    "BuyWhenFlat",
    "Raises",
    "Rig",
    "Script",
    "ist",
    "rig",
]
