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

from openalgo.backtest import _market, _positional_exit, reference
from openalgo.backtest._live import _session_versions
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
#: The stock future of the same last trading day (Tuesday 2026-11-24 is a holiday) and its
#: lot in the test's own reference rows (`stock_future_tables`).
STOCK_FUTURE = "SBIN23NOV26FUT"
STOCK_LOT = 750
#: The test's reference rows of SBIN's derivatives on NFO, by table: the built-in tables
#: list index underlyings only, and a stock's are a user file's (the tables' docs).
_STOCK_ROWS = {
    "underlyings": (
        "underlying,exchange,valid_from,valid_to,kind,source,status,note",
        "SBIN,NFO,2019-01-01,,stock,test row,user,",
    ),
    "expiry_rules": (
        "underlying,exchange,series,valid_from,valid_to,weekday,listed_from,source,status,note",
        "SBIN,NFO,weekly,2019-01-01,,NONE,,test row,user,",
        "SBIN,NFO,monthly,2025-09-01,,TUE,,test row,user,",
    ),
    "lot_sizes": (
        "underlying,exchange,valid_from,valid_to,expiry_from,expiry_to,lot_size,source,status,note",
        f"SBIN,NFO,2025-01-01,,,,{STOCK_LOT},test row,user,",
    ),
    "freeze_qty": (
        "underlying,exchange,valid_from,valid_to,freeze_qty_futures,freeze_qty_options,source,"
        "status,note",
        "SBIN,NFO,2025-01-01,,15000,15000,test row,user,",
    ),
}


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


def stock_future_tables(monkeypatch: Any, folder: Any) -> None:
    """From now on the reference tables are the built-in ones with `_STOCK_ROWS` as user
    files written to `folder`: neither `bt.live` nor `bt.backtest` takes reference tables, so
    every module that reads the default tables is given these (a stock's derivatives cannot
    be planned on the built-in rows alone)."""
    files = {}
    for table, lines in _STOCK_ROWS.items():
        path = folder / f"{table}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        files[table] = path
    tables = reference.load_tables(user_files=files)
    for module in (reference, _market, _positional_exit, _session_versions):
        monkeypatch.setattr(module, "default_tables", lambda: tables)


def expiry_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    symbol: str,
    day: dt.date,
    start: tuple[int, int, int],
    analyzer: bool = False,
    lot: int = LOT,
) -> Rig:
    """A rig on `day` started at the IST time `start`, trading `symbol` (module
    docstring), its lot `lot` units, in the sandbox when `analyzer`."""
    at = ist(day, *start)
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(HistoryFetcher, "_wait", lambda fetcher, seconds: None)
    build = functools.partial(make_fake_openalgo, gtt_history=True)
    fake = build(start_ns=at)
    option = symbol == OPTION
    fake.add_instrument(
        symbol,
        "NFO",
        lot_size=lot,
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
        fake.trade(symbol, "NFO", world.price, volume=lot)

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
    "STOCK_FUTURE",
    "STOCK_LOT",
    "expiry_world",
    "flat_candles",
    "ist",
    "stock_future_tables",
]
