"""A live session on the last trading day of an NFO contract, shared by the expiry tests
(`test_session_expiry.py`; design 20-positional-live section 9, the lead's ruling N5b-3).

The market: one NFO contract on the fake OpenAlgo, its broker serving flat 1-minute candles
(at `PRICE`) of the two NFO sessions before the day and of the day up to the session's
start, and trading one lot every 10 seconds at `world.price` from the start until 15:31
IST. Two contracts, each on its last trading day: the NIFTY weekly call of Tuesday
2026-11-17 at 24000 (an option: exited by the engine at the square-off of that day), and
the NIFTY future of Monday 2026-11-23 (an index future settled in cash: left to the
exchange). The NFO square-off is 15:10 IST and the close 15:30 (the calendar's dated rows).
Lot size, tick and freeze quantity are the reference tables'. Expected values come from
the scenario.
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import Any

from fake_openalgo import ist_ns
from session_hooks import trace_sessions
from session_invariants import assert_session_invariants
from session_samples import NS, TRADE_EVERY_S, Rig, Script

from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data.fetching import HistoryFetcher

MINUTE_NS = 60 * NS
#: The contracts' lot (units), as the reference tables give NIFTY's in November 2026.
LOT = 65
#: The contracts' price all day, rupees.
PRICE = 120.0
#: The option and its last trading day.
OPTION_DAY = dt.date(2026, 11, 17)
OPTION = "NIFTY17NOV2624000CE"
#: The index future and its last trading day.
FUTURE_DAY = dt.date(2026, 11, 23)
FUTURE = "NIFTY23NOV26FUT"


def ist(day: dt.date, hour: int, minute: int, second: int = 0) -> int:
    """UTC nanoseconds of an IST wall-clock time of `day`."""
    return ist_ns(day.year, day.month, day.day, hour, minute, second)


Candle = tuple[int, float, float, float, float, float]


def flat_candles(day: dt.date, until_ns: int) -> list[Candle]:
    """Flat 1-minute candles at `PRICE` over the two NFO sessions before `day` and over
    `day` up to `until_ns` (UTC nanoseconds)."""
    calendar = default_calendar()
    sessions = list(calendar.sessions("NFO", day - dt.timedelta(days=7), day)["date"])
    rows = []
    for session_day in [*sessions[-3:-1], day]:
        opens, closes = calendar.session("NFO", session_day)
        end = min(closes, until_ns) if session_day == day else closes
        rows += [(m, PRICE, PRICE, PRICE, PRICE, 1_000.0) for m in range(opens, end, MINUTE_NS)]
    return rows


def expiry_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    symbol: str,
    day: dt.date,
    start: tuple[int, int, int],
    analyzer: bool = False,
) -> Rig:
    """A rig on `day` started at the IST time `start`, trading `symbol` (module
    docstring), in the sandbox when `analyzer`."""
    at = ist(day, *start)
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(HistoryFetcher, "_wait", lambda fetcher, seconds: None)
    build = functools.partial(make_fake_openalgo, gtt_history=True)
    fake = build(start_ns=at)
    option = symbol == OPTION
    fake.add_instrument(
        symbol,
        "NFO",
        lot_size=LOT,
        tick_size=0.05,
        freeze_qty=1800,
        instrument_type="CE" if option else "FUT",
        expiry=f"{day:%d-%b-%y}".upper(),
        strike=24000.0 if option else 0.0,
    )
    fake.seed_candles(symbol, "NFO", flat_candles(day, ist(day, *start[:2])))
    if analyzer:
        fake.set_analyzer(True)
    script = Script(fake)
    trace = trace_sessions(fake, monkeypatch, assert_session_invariants)
    world = Rig(fake, script, tmp_path, trace)
    world.price = PRICE

    def trade() -> None:
        fake.trade(symbol, "NFO", world.price, volume=LOT)

    trade()
    for instant in range(at + TRADE_EVERY_S * NS, ist(day, 15, 31), TRADE_EVERY_S * NS):
        script.at(instant, trade)
    return world


__all__ = [
    "FUTURE",
    "FUTURE_DAY",
    "LOT",
    "OPTION",
    "OPTION_DAY",
    "PRICE",
    "expiry_world",
    "flat_candles",
    "ist",
]
