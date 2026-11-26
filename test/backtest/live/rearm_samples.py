"""A CNC leg carried into a live session under `positional_stops="rearm_daily"`, shared by
the rearm_daily session tests (`test_session_rearm*.py`; design 20-positional-live 6, the
lead's ruling N5b-2).

The market is `session_samples`': SBIN on NSE on Monday 2026-11-16. The session starts
before the open, at 09:05 IST, and SBIN trades 10 units every 10 seconds from the open at
the script's price (500.00 unless a test moves it). The fake broker holds 10 SBIN for
delivery at 500.00, bought on Friday 2026-11-13.

Until the restart across days (unit N5a, built beside this one) restores a strategy's
day-close record, the leg is carried through the binding's `restore_carried_state`, as the
ruling asks: a kernel session of Friday, built from the live strategy's own setup, buys 10
SBIN CNC at 500.00 with the strategy's 3-point stop (497.00) at Friday's open and holds them
to the close; its carried state is restored into the live strategy's kernel session as the
start restores today's state (`carry_into`). Expected values come from the scenario, never
from the session's code.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
from positional_samples import CncHold
from session_samples import NS, ist, rig
from stops_samples import STOP_POINTS, Timeline

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest._engine import instrument_dict
from openalgo.backtest._live import _session_start
from openalgo.backtest.calendar import default_calendar

#: The trading day before the session's, on which the carried position was bought.
FRIDAY = dt.date(2026, 11, 13)
#: The session's start, before NSE's open (09:15 IST).
START = (9, 5, 0)
#: The carried leg: units, entry price and stop level, rupees.
UNITS, ENTRY, STOP = 10, 500.0, 497.0
#: NSE's 09:15 open, Monday.
OPEN = ist(9, 15)

_MINUTE_NS = 60 * NS
_NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))


class RearmHold(CncHold):
    """CncHold (10 SBIN CNC, a 3-point stop) that never enters in the session: it only
    holds the leg carried from Friday."""

    def on_bar(self, bar):
        self.calls += 1


def _friday_close(setup: Any) -> dict[str, Any]:
    """Friday's carried state of a kernel session built from the live strategy's `setup`
    (module docstring), through JSON as a day-close record keeps it."""
    config = dict(setup.config) | {"first_day": to_day(FRIDAY), "last_day": to_day(FRIDAY)}
    specs = [instrument_dict(instrument) for instrument in setup.instruments]
    engine = native_attr("Engine").session(default_calendar()._cal, config, specs, [_NO_BARS])
    opened, closed = default_calendar().session("NSE", FRIDAY)
    slices = (closed - opened) // _MINUTE_NS
    request = {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": float(UNITS)},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "CNC",
        "reduce_only": False,
        "reverse": False,
        "rules": {"stop": ("points", STOP_POINTS, None)},
    }
    for minute in range(slices):
        assert engine.advance()[0] == "waiting"
        engine.add_bar(0, (opened + minute * _MINUTE_NS, ENTRY, ENTRY, ENTRY, ENTRY, 1000.0))
        engine.release(minute + 1)
        if engine.advance()[0] == "bar_close" and minute == 0:
            assert engine.submit(request)[0] == "submitted"
            assert engine.advance()[0] == "waiting"
            (entry,) = engine.take_commands()
            engine.apply_fill(entry["order_id"], float(UNITS), ENTRY, opened + _MINUTE_NS)
    if engine.advance()[0] == "waiting":
        for command in engine.take_commands():
            if command["command"] == "cancel":
                engine.apply_cancel(command["order_id"])
        engine.advance()
    return json.loads(json.dumps(engine.carried_state()))


def carry_into(monkeypatch: Any) -> list[dict[str, Any]]:
    """From now on, the start restores Friday's carried leg into each rearm_daily strategy's
    kernel session right after it restores today's state (module docstring); returns the
    restore's answers, one per strategy, in order."""
    answers: list[dict[str, Any]] = []
    restore = _session_start.restore

    def restored(session: Any, restart: Any, sleep: Any) -> None:
        restore(session, restart, sleep)
        for trader in session.traders:
            carried = _friday_close(trader.part.setup)
            answers.append(trader.engine.restore_carried_state(carried))

    monkeypatch.setattr(_session_start, "restore", restored)
    return answers


def rearm_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    gtt: bool = True,
    analyzer: bool = False,
    open_price: float = ENTRY,
    start: tuple[int, int, int] = START,
) -> Any:
    """A rig started at the IST time `start` (`START` by default) holding Friday's 10 SBIN
    (module docstring), the broker holding GTT orders or not (`gtt`), in the sandbox when
    `analyzer`; SBIN trades from the open (or the start, after it) at `open_price` until a
    test moves it (`world.script.price`)."""
    world = rig(
        lambda **o: make_fake_openalgo(gtt=gtt, **o),
        monkeypatch,
        tmp_path,
        start=start,
        trades_until=None,
        analyzer=analyzer,
    )
    world.timeline = Timeline(world.fake)
    world.fake.set_holding("SBIN", "NSE", UNITS, average_price=ENTRY)
    world.trace.carried(world.fake.mode, "SBIN", "NSE", UNITS)
    world.script.price = lambda now_ns: open_price
    world.script.trade()
    world.script.trades(max(OPEN, ist(*start) + 10 * NS), ist(15, 2))
    world.carried = carry_into(monkeypatch)
    return world


__all__ = [
    "ENTRY",
    "FRIDAY",
    "OPEN",
    "START",
    "STOP",
    "UNITS",
    "RearmHold",
    "carry_into",
    "rearm_world",
]
