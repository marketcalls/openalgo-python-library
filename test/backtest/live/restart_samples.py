"""A crash and a restart of a live session on the fake OpenAlgo, shared by the restart's
tests (`test_restart*.py`; `_live/_restart_books.py`, `_live/_session_restore.py`,
16-session-design 3.6, 5.5 and 7.2, unit 12).

A crash drops the session where it stands, as a process that dies would: nothing more is
sent and nothing of the safe stop or the common close runs, but what the operating system
would release (the journal's files, the account lock) is released, so a restart on the
same journal can open it (`crash`). A restart is a new session of the same rig, on the
same journal and the fake's same clock (`restart`).

The market is `stops_samples`': SBIN on NSE from 14:40:05 IST, trading 10 units every
10 s at 500.00. `StopWhenFlat` buys 10 SBIN with a stop 3 points below (an SL-M at
497.00) at each decision at which it holds none: a restarted strategy that resumes its
position buys no more. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

from typing import Any

import pytest
from session_samples import StopFlag
from stops_samples import STOP_POINTS

import openalgo.backtest as bt
from openalgo.backtest._live import session as session_module


class Crash(BaseException):
    """The process dies here (a test's hook raises it)."""


class StopWhenFlat(bt.Strategy):
    """Buys 10 SBIN with a stop 3 points below at each decision at which it holds none."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.position.qty and not self.orders:
            self.buy(10, stop=bt.points(STOP_POINTS))


class TrailsWhenFlat(bt.Strategy):
    """StopWhenFlat whose stop trails 3 points below the best price."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.position.qty and not self.orders:
            self.buy(10, stop=bt.points(STOP_POINTS), trail=bt.trail_points(3))


def crashing(monkeypatch: Any) -> None:
    """From now on a `Crash` inside a session drops it where it stands: its stack (the
    journal's files, the account lock, the threads) is closed, and nothing is sent."""
    abort = session_module._abort

    def drop(session: Any, error: BaseException, *, journal_works: bool) -> None:
        if isinstance(error, Crash):
            session.parts.worker.close()
            session.parts.stack.close()
            return
        abort(session, error, journal_works=journal_works)

    monkeypatch.setattr(session_module, "_abort", drop)


def crash_at(world: Any, instant_ns: int) -> None:
    """The session crashes at `instant_ns`."""

    def die() -> None:
        raise Crash(f"crash at {instant_ns}")

    world.script.at(instant_ns, die)


def run_to_crash(world: Any, strategy: Any, **arguments: Any) -> None:
    """Run a session of `strategy` until it crashes."""
    with pytest.raises(Crash):
        world.run(strategy, **arguments)


def restart(world: Any, strategy: Any, *, stop_ns: int | None = None, **arguments: Any) -> Any:
    """Start a new session of `strategy` on the rig's journal, as after a crash, asked to
    stop (Ctrl+C) at `stop_ns` when given."""
    world.stop, world.feed = StopFlag(), None
    if stop_ns is not None:
        world.stop_at(stop_ns)
    return world.run(strategy, **arguments)


def stops_working(world: Any) -> list[Any]:
    """The SL-M stops working at the fake."""
    return [o for o in world.orders() if o.price_type == "SL-M" and o.status == "trigger pending"]


__all__ = [
    "Crash",
    "StopWhenFlat",
    "TrailsWhenFlat",
    "crash_at",
    "crashing",
    "restart",
    "run_to_crash",
    "stops_working",
]
