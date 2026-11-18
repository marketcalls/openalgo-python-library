"""The public `bt.live` on the fake OpenAlgo, shared by its tests (`test_live_*.py`; `run.live`,
`_live/_result_build.py`, 16-session-design 1.2 and 1.4, unit 13).

`bt.live` is called as a user calls it. Only what the session takes from its process is
the fake's: its five clients, its feed, its clock and waits, its Ctrl+C and its console,
bound into the session's own test seam (`run_session`'s keywords), since clients derived
from the fake's would not reach it and the wall clock would not be the fake's. Nothing
contacts a server. The market and the script are `session_samples`'.
"""

from __future__ import annotations

import functools
from typing import Any

from fake_openalgo import USERNAME
from session_samples import CAPITAL, MAX_DAILY_LOSS, WAIT_MAX_S, Rig

import openalgo.backtest as bt
from openalgo.backtest._live import session as session_module


def seam(world: Rig, monkeypatch: Any) -> None:
    """From now on a `bt.live` call runs its session on `world`'s fake, under the test
    seam: a new set of the fake's clients and a new direct feed for each call."""
    run_session = session_module.run_session

    def on_the_fake(strategy: Any, client: Any, **arguments: Any) -> Any:
        world.feed = world.fake.direct_feed()
        bound = functools.partial(
            run_session,
            clients=world.clients(),
            feed=world.feed,
            now=world.fake.clock.now,
            sleep=world.script.sleep,
            threads=False,
            signals=False,
            stop=world.stop,
            emit=world.lines.append,
            wait_max_s=WAIT_MAX_S,
        )
        return bound(strategy, client, **arguments)

    monkeypatch.setattr(session_module, "run_session", on_the_fake)


def live(world: Rig, strategy: Any, **arguments: Any) -> bt.SessionResult:
    """`bt.live(strategy, client, ...)` with the session samples' limits and alerts, the
    rig's folder, and `arguments`."""
    limits = {
        "capital": CAPITAL,
        "max_daily_loss": MAX_DAILY_LOSS,
        "alerts_to": USERNAME,
        "live_dir": world.folder / "live",
    }
    return bt.live(strategy, world.fake.client(), **(limits | arguments))


__all__ = ["live", "seam"]
