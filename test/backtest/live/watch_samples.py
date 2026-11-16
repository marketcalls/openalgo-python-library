"""The strategies and helpers of the risk watch's session tests (`test_watch_*.py`;
`_live/watch.py`, `_live/_clock_skew.py`, `_live/_kill_switch.py`, 16-session-design unit
11), on `session_samples`' market: SBIN on NSE on Monday 2026-11-16, the session started
at 14:40:05 IST, trading 10 units every 10 s at the price the script gives.

`run` runs a session as `Rig.run` does, with what these tests change: the session's own
clock (one ahead of the fake's), a server address on this computer, and the SDK's real
feed in place of the fake's direct one. Nothing contacts a server: every REST call is
answered in-process by the fake, whatever address the clients carry.

Expected values come from the scenario and the fake's books, never from the watch's code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from fake_openalgo import USERNAME
from feed_fakes import wait_for
from session_samples import CAPITAL, MAX_DAILY_LOSS, NS, WAIT_MAX_S, Rig

import openalgo.backtest as bt
from openalgo.backtest._live.clients import SessionClients
from openalgo.backtest._live.session import run_session

#: An address on this computer the clients are given to stand for a server here: a
#: loopback host on which nothing listens (the fake answers every call in-process).
LOOPBACK_HOST = "http://127.0.0.2:5999"
#: The option of the NFO tests: a NIFTY call of the reference tables (lot 65 units).
OPTION = "NIFTY01DEC2626000CE"
OPTION_LOT = 65


def run(
    rig: Rig,
    strategy: Any,
    *,
    ahead_s: float = 0.0,
    clients: SessionClients | None = None,
    sdk_feed: bool = False,
    **arguments: Any,
) -> Any:
    """Run a session of `strategy` on the rig's fake to its end, under the test seam, its
    clock `ahead_s` seconds ahead of the fake's, with `clients` (the rig's by default),
    and the SDK's real feed of `clients.feed` when `sdk_feed` (else the rig's direct
    feed)."""
    clients = rig.clients() if clients is None else clients
    feed = None
    if not sdk_feed:
        rig.feed = rig.fake.direct_feed() if rig.feed is None else rig.feed
        feed = rig.feed
    clock = rig.fake.clock
    limits = {
        "capital": CAPITAL,
        "max_daily_loss": MAX_DAILY_LOSS,
        "alerts_to": USERNAME,
        "live_dir": rig.folder / "live",
    }
    return run_session(
        strategy,
        rig.fake.client(),
        clients=clients,
        feed=feed,
        now=lambda: clock.now() + round(ahead_s * NS),
        sleep=rig.script.sleep,
        threads=False,
        signals=False,
        stop=rig.stop,
        emit=rig.lines.append,
        wait_max_s=WAIT_MAX_S,
        **(limits | arguments),
    )


def on_this_computer(clients: SessionClients) -> SessionClients:
    """`clients` addressed to a server on this computer (`LOOPBACK_HOST`)."""
    for client in (clients.orders, clients.guard, clients.data, clients.alerts, clients.feed):
        client.base_url = f"{LOOPBACK_HOST}/api/v1/"
    return clients


def events(outcome: Any, kind: str | None = None) -> list[dict[str, Any]]:
    """The session's event stream (of `kind` when given), oldest first."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [line for line in lines if kind is None or line["kind"] == kind]


def calls_between(rig: Rig, endpoint: str, start_ns: int, end_ns: int) -> list[Any]:
    """The fake's calls of `endpoint` answered in [start_ns, end_ns)."""
    return [call for call in rig.fake.log.calls(endpoint) if start_ns <= call.ts_ns < end_ns]


class AlertTimes(logging.Handler):
    """Notes, for each alert the session sends, the time on `clock` (UTC nanoseconds)
    and its text: `Alerts.send` logs each alert at once, on the calling thread."""

    def __init__(self, clock: Callable[[], int]) -> None:
        super().__init__(level=logging.DEBUG)
        self._clock = clock
        self.sent: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("alert "):
            self.sent.append((self._clock(), message))

    def of(self, kind: str, words: str) -> list[int]:
        """The times of the alerts of `kind` whose text holds `words`."""
        tag = f"alert {kind}: "
        return [ts for ts, text in self.sent if text.startswith(tag) and words in text]


class HoldWithTarget(bt.Strategy):
    """Buys 10 SBIN at its first decision with a target 5 rupees above the fill, never
    buys again; keeps the (bar close, bar volume) of every decision in the class's `seen`
    (the session builds the strategy, so the test reads the class)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    seen: list[tuple[int, float]] = []

    def init(self):
        type(self).seen = []

    def on_bar(self, bar):
        type(self).seen.append((bar.ts_close, bar.volume))
        if len(type(self).seen) == 1:
            self.buy(10, target=bt.points(5))


class HoldFiveMinutes(bt.Strategy):
    """A 5-minute strategy: buys 10 SBIN at its first decision and holds them."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        self.decisions = 0

    def on_bar(self, bar):
        self.decisions += 1
        if self.decisions == 1:
            self.buy(10)


class Flipper(bt.Strategy):
    """Buys 10 SBIN at each decision at which it holds none, sells them at the next."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if self.position.qty:
            self.exit()
        else:
            self.buy(10)


class HoldSbin(bt.Strategy):
    """Buys 10 SBIN at each decision at which it holds none."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.position.qty:
            self.buy(10)


class HoldInfy(HoldSbin):
    """Buys 10 INFY at each decision at which it holds none."""

    symbol = "INFY"


class RestingBuy(bt.Strategy):
    """Rests a LIMIT buy of 10 SBIN far below the market at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.orders and not self.position.qty:
            self.buy(10, price=400.0)


class HookWitness:
    """Hooks registered on an SDK feed client after the session's own, so once one has
    run, the session's hook for the same event has run too (the SDK delivers a hook's
    listeners in order): `wait` waits, in real time and bounded, for an event."""

    def __init__(self, client: Any) -> None:
        self.seen: list[str] = []
        for event in ("disconnect", "reconnect"):
            register = getattr(client, f"on_{event}")
            register(lambda payload, event=event: self.seen.append(event))

    def wait(self, event: str) -> None:
        """Wait until the hook of `event` has run. Raises AssertionError when it does not
        within the feed tests' wait."""
        assert wait_for(lambda: event in self.seen), f"the feed's {event} hook never ran"


class HoldOption(bt.Strategy):
    """Buys one lot of the NFO call at its first decision and holds it."""

    symbol, exchange, interval = OPTION, "NFO", "1m"

    def init(self):
        self.bought = False

    def on_bar(self, bar):
        if not self.bought:
            self.bought = True
            self.buy(OPTION_LOT)


__all__ = [
    "LOOPBACK_HOST",
    "OPTION",
    "OPTION_LOT",
    "AlertTimes",
    "Flipper",
    "HoldFiveMinutes",
    "HoldInfy",
    "HoldOption",
    "HoldSbin",
    "HoldWithTarget",
    "HookWitness",
    "RestingBuy",
    "calls_between",
    "events",
    "on_this_computer",
    "run",
]
