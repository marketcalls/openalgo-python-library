"""A session that crashes before its day-close record, continued on a later trading day, on
the fake OpenAlgo: shared by the tests of the rebuild (`test_session_carry_crash*.py`;
20-positional-live 4.2 step 2; the lead's rulings N5c-2 and N5c-3).

A crash is N5a's: the process ends inside the session (`restart_samples.crashing`), with no
close and no record. The days and the strategies are `carry_samples`': Monday's session
starts at 14:40:05 IST, SBIN trades every 10 s at 500.00 until 15:32, and a strategy buys 10
SBIN CNC at its first decision (14:42:01, filled at 500.00) with a stop 3 points below held
by a SINGLE GTT (sell 10 at 497.00 or below, the child's limit 492.00).

**The banner's line of a rebuild** names when the session stopped and when it last saved:
a session saves at each slice it releases, one second after the slice's close
(`_plan_release.SETTLE_NS`: the 14:44 slice, closing at 14:45:00, is released at 14:45:01),
and before each batch of commands; with no update after the save, it stopped then.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from carry_samples import DAY1, add_future, on
from positional_samples import gtt_world
from restart_samples import crash_at, crashing, run_to_crash
from session_samples import StopFlag

#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)
#: The calls that place, change or cancel a GTT.
GTT_CALLS = ("placegttorder", "modifygttorder", "cancelgttorder")


def crashed_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    strategy: Any,
    *,
    crash: tuple[int, int, int] = (14, 46, 0),
    future: str | None = None,
    **options: Any,
) -> Any:
    """A `positional_samples.gtt_world` (its `history` and `analyzer` among `options`) whose
    Monday session of `strategy` crashed at the IST time `crash`: the process ended with no
    close and no record (module docstring). With `future` (a symbol of `carry_samples`), the
    fake also trades it, as `carry_samples.carried_world` does. The world stays on Monday."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **options)
    world.future = future
    if future is not None:
        add_future(world, future)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, *crash))
    run_to_crash(world, strategy)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    return world


def rebuilt_line(name: str, day: dt.date, stopped: str, saved: str, updates: int) -> str:
    """The banner's line of a session rebuilt (module docstring)."""
    return (
        f"{name}: its session of {day} stopped at {stopped} IST without its day-close record; "
        f"its position is continued from its last save of {saved} and {updates} later updates"
    )


__all__ = ["GTT_CALLS", "SENDS", "crashed_world", "rebuilt_line"]
