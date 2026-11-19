"""In-process fakes for the recorder's tests: manual clocks, a feed with the SDK's feed
calls whose ticks and outages a test plays, a script that plays them while the recorder
waits, and the recording a test runs in a child process and kills. Nothing here opens a
socket.

The feed's calls and hook payloads are shaped as the SDK's `FeedAPI` makes them
(openalgo/feed.py): `connect` returns whether the server accepted the API key,
`subscribe_<mode>(instruments, on_data_received)` keeps the mode's data callback, a
disconnect hook receives {"event": "disconnect", ..., "requested": ..., "will_reconnect":
...}, and `disconnect()` fires it with "requested": True for a connection that was up.
The data messages are those of the SDK's quote callback (openalgo/_feed_ticks.py).
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

NS = 1_000_000_000
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
#: Monday 2026-11-16: NSE trades 09:15 to 15:30 IST.
DAY = dt.date(2026, 11, 16)
SBIN = {"exchange": "NSE", "symbol": "SBIN"}
INFY = {"exchange": "NSE", "symbol": "INFY"}
#: The exit code of the child process the kill test ends with `os._exit`.
KILLED_EXIT_CODE = 9
_MODES = {1: "ltp", 2: "quote", 3: "depth"}


def ist_ns(clock: str, day: dt.date = DAY) -> int:
    """UTC nanoseconds of an IST clock time ("10:00:01.250") on `day`."""
    moment = dt.datetime.combine(day, dt.time.fromisoformat(clock), tzinfo=IST)
    return (moment - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)) // dt.timedelta(microseconds=1) * 1000


def quote(symbol: str, ltp: float, *, volume: int = 0, ltt: Any = 0, exchange: str = "NSE"):
    """A quote callback message of the SDK: every quote field, 0 where none was sent."""
    data = {
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
        "ltp": ltp,
        "volume": volume,
        "last_trade_quantity": 0,
        "avg_trade_price": 0,
        "change": 0,
        "change_percent": 0,
        "ltt": ltt,
        "total_buy_quantity": 0,
        "total_sell_quantity": 0,
        "oi": 0,
        "timestamp": 0,
    }
    return {"type": "market_data", "symbol": symbol, "exchange": exchange, "mode": 2, "data": data}


def raw_files(cache_dir: Path) -> list[Path]:
    """Every raw tick file under a cache directory."""
    return sorted(Path(cache_dir).rglob("*.raw"))


class Clocks:
    """The injected clocks of a recording: `wall`, the UTC clock (`Now`), which a test may
    step as a system clock is corrected, and `mono`, a monotonic clock that never steps.
    Both start at `start_ns` and move together unless the wall clock is stepped."""

    def __init__(self, start_ns: int) -> None:
        self.wall_ns = start_ns
        self.mono_ns = start_ns

    def wall(self) -> int:
        return self.wall_ns

    def mono(self) -> int:
        return self.mono_ns

    def advance(self, ns: int) -> None:
        """Time passes: both clocks move on."""
        self.wall_ns += ns
        self.mono_ns += ns

    def step_wall(self, ns: int) -> None:
        """The system clock is corrected by `ns` (negative: back)."""
        self.wall_ns += ns


class FakeFeed:
    """The feed side of an `openalgo.api` client, played by the test.

    `calls` lists the client calls in order. `push` hands a message to the data callback
    of its mode, as the feed's thread does; `drop` and `restore` fire the disconnect and
    reconnect hooks of a lost and a restored connection.
    """

    def __init__(self, *, connects: bool = True) -> None:
        self.connects = connects
        self.connected = False
        self.calls: list[str] = []
        self.callbacks: dict[str, Callable] = {}
        self.requests: dict[str, list] = {}
        self.hooks: dict[str, list[Callable]] = {"disconnect": [], "reconnect": []}
        self._seq = 0

    # ----- the SDK's calls -----

    def connect(self) -> bool:
        self.calls.append("connect")
        self._seq += 1
        self.connected = self.connects
        return self.connects

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        if self.connected:
            self.connected = False
            self._fire("disconnect", self._down(requested=True))

    def on_disconnect(self, callback: Callable) -> Callable:
        self.hooks["disconnect"].append(callback)
        return callback

    def on_reconnect(self, callback: Callable) -> Callable:
        self.hooks["reconnect"].append(callback)
        return callback

    def remove_hook(self, callback: Callable) -> bool:
        self.calls.append("remove_hook")
        found = [hooks for hooks in self.hooks.values() if callback in hooks]
        for hooks in found:
            hooks.remove(callback)
        return bool(found)

    def subscribe_ltp(self, instruments, on_data_received=None) -> bool:
        return self._subscribe("ltp", instruments, on_data_received)

    def subscribe_quote(self, instruments, on_data_received=None) -> bool:
        return self._subscribe("quote", instruments, on_data_received)

    def subscribe_depth(self, instruments, on_data_received=None) -> bool:
        return self._subscribe("depth", instruments, on_data_received)

    def unsubscribe_ltp(self, instruments) -> bool:
        return self._unsubscribe("ltp", instruments)

    def unsubscribe_quote(self, instruments) -> bool:
        return self._unsubscribe("quote", instruments)

    def unsubscribe_depth(self, instruments) -> bool:
        return self._unsubscribe("depth", instruments)

    # ----- what the test plays -----

    def push(self, message: dict) -> None:
        """Hand a message to its mode's data callback, as the feed's thread does."""
        self.callbacks[_MODES[message["mode"]]](message)

    def drop(self) -> None:
        """The connection is lost; the SDK reconnects by itself."""
        self.connected = False
        self._fire("disconnect", self._down(requested=False))

    def restore(self) -> None:
        """A new connection is up and the subscriptions are replayed."""
        self._seq += 1
        self.connected = True
        self._fire("reconnect", {"event": "reconnect", "seq": self._seq, "reconnect": True})

    # ----- inside -----

    def _subscribe(self, mode: str, instruments, callback) -> bool:
        self.calls.append(f"subscribe_{mode}")
        self.requests[f"subscribe_{mode}"] = list(instruments)
        if callback is not None:
            self.callbacks[mode] = callback
        return self.connected

    def _unsubscribe(self, mode: str, instruments) -> bool:
        self.calls.append(f"unsubscribe_{mode}")
        self.requests[f"unsubscribe_{mode}"] = list(instruments)
        return self.connected

    def _down(self, *, requested: bool) -> dict:
        return {
            "event": "disconnect",
            "seq": self._seq,
            "url": "ws://127.0.0.1:8765",
            "code": None,
            "reason": "disconnect() called" if requested else "connection lost",
            "error": None,
            "requested": requested,
            "will_reconnect": not requested,
        }

    def _fire(self, event: str, payload: dict) -> None:
        for hook in list(self.hooks[event]):
            hook(dict(payload))


class Script:
    """The injected `wait` of a recording: it plays scripted events at their instants on
    the monotonic clock (the wall clock's instants while it is not stepped). A wait
    returns right after the first event due before it would end, as the recorder's own
    wait wakes when a tick arrives; else time passes by what it was given."""

    def __init__(self, clocks: Clocks) -> None:
        self.clocks = clocks
        self._events: list[tuple[int, int, Callable[[], None]]] = []

    def at(self, at_ns: int, action: Callable[[], None]) -> None:
        """Play `action` at `at_ns`; events of one instant play in the order given."""
        self._events.append((at_ns, len(self._events), action))
        self._events.sort(key=lambda event: event[:2])

    def __call__(self, seconds: float) -> None:
        until_ns = self.clocks.mono_ns + round(seconds * NS)
        if not self._events or self._events[0][0] > until_ns:
            self.clocks.advance(until_ns - self.clocks.mono_ns)
            return
        due_ns = self._events[0][0]
        self.clocks.advance(max(due_ns - self.clocks.mono_ns, 0))
        while self._events and self._events[0][0] == due_ns:
            self._events.pop(0)[2]()


def interrupt() -> None:
    """Ctrl+C, as Python raises it in the waiting thread."""
    raise KeyboardInterrupt


def killed_recording(cache_dir: str) -> None:
    """A recording of SBIN from 10:00:00 that receives two ticks (taken at 10:00:00.200
    and 10:00:00.500, on disk by the flush at 10:00:01) and is killed at 10:00:02 without
    a chance to clean up. Run in a child process: it never returns."""
    from openalgo.backtest._live.recorder import record

    clocks = Clocks(ist_ns("10:00:00"))
    feed = FakeFeed()
    script = Script(clocks)
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.500"), lambda: feed.push(quote("SBIN", 812.5)))
    script.at(ist_ns("10:00:02"), lambda: os._exit(KILLED_EXIT_CODE))
    record(
        feed,
        [SBIN],
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        cache_dir=cache_dir,
        stop_at_ns=ist_ns("10:01:00"),
    )
