"""Two or more trading days of a CNC strategy on the fake OpenAlgo, shared by the tests of
the restart across days (`test_session_carry*.py`; 20-positional-live 4; the lead's rulings
N5a-1 to N5a-6).

Day 1 is `positional_samples`' Monday 2026-11-16: the session starts at 14:40:05 IST, SBIN
trades every 10 s at 500.00 until 15:32; `CncKept` buys 10 SBIN CNC at its first decision
(14:42:01, filled at 500.00) with a stop 3 points below held by the broker as a SINGLE GTT
(sell 10 at 497.00, the child's limit 492.00), and the session ends at NSE's close with the
leg carried and its day-close record written. The fake's day then ends (`end_day`: the CNC
units move into the holdings, the GTT stays). A later trading day (`to_day`) starts its
session at 14:40:05 IST with SBIN's broker candles flat at 500.00 from the earlier days'
close to the start, SBIN trading every 10 s at the script's price until 15:32; a day no
session runs (`missed_day`) has the fake's market trade at the prices a test gives, and
ends.

Expected values come from the fake's books and the scenario, never from the session's code.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fake_openalgo import BROKER
from fake_openalgo.clock import ist_ns
from plan_samples import candles
from positional_samples import CncHold, gtt_world
from session_samples import NS, TRADE_EVERY_S
from stops_samples import STOP_POINTS

import openalgo.backtest as bt
from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal
from openalgo.backtest.calendar import default_calendar

#: The trading days of the scenarios: Monday, Tuesday, Wednesday and Thursday.
DAY1, DAY2, DAY3, DAY4 = (dt.date(2026, 11, d) for d in (16, 17, 18, 19))
#: The trading days after them to November's last NIFTY future's last trading day
#: (Monday 2026-11-23), and the next trading day (Wednesday 2026-11-25: Tuesday 2026-11-24
#: is a holiday).
TO_NOVEMBERS_LAST = tuple(dt.date(2026, 11, d) for d in (20, 23))
AFTER_NOVEMBERS_LAST = dt.date(2026, 11, 25)
#: Where a later day's session starts, and until when SBIN trades then (IST).
START, TRADES_UNTIL = (14, 40, 5), (15, 32)
MINUTE_NS = 60 * NS


def on(day: dt.date, hour: int, minute: int, second: int = 0) -> int:
    """UTC nanoseconds of an IST wall-clock time of `day`."""
    return ist_ns(day.year, day.month, day.day, hour, minute, second)


class CncKept(CncHold):
    """CncHold whose entry is kept in `self.state`: a later start buys nothing more."""

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(self.qty, **type(self).rules)


class CncKeptTrail(CncKept):
    """CncKept whose stop trails 3 points below the best price since the entry, with a
    target 6 points above the entry (506.00), held by the session on ticks."""

    rules = {
        "stop": bt.points(STOP_POINTS),
        "trail": bt.trail_points(STOP_POINTS),
        "target": bt.points(6),
    }


#: A NIFTY future of the scenarios that hold one, NRML: one lot of 65 units, priced as the
#: stops tests price the option (about 120.00), so its margin is small; and November's,
#: whose last trading day is Monday 2026-11-23 (the month's last Tuesday, 2026-11-24, is a
#: holiday).
FUTURE, FUTURE_LOT, FUTURE_PRICE = "NIFTY29DEC26FUT", 65, 120.0
NOVEMBER_FUTURE = "NIFTY23NOV26FUT"
#: Each future's expiry as OpenAlgo's master contract writes it.
_EXPIRIES = {FUTURE: "29-DEC-26", NOVEMBER_FUTURE: "23-NOV-26"}


class FutKept(bt.Strategy):
    """Buys one lot of `FUTURE` in NRML at its first decision with a stop 10 points below,
    the entry kept in `self.state`."""

    symbol, exchange, interval, product = FUTURE, "NFO", "1m", "NRML"

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(FUTURE_LOT, stop=bt.points(10))


class NovemberKept(FutKept):
    """FutKept on November's future."""

    symbol = NOVEMBER_FUTURE


def carried_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    strategy: Any = CncKept,
    *,
    future: str | None = None,
    **options: Any,
) -> Any:
    """A `positional_samples.gtt_world` (its `history` and `analyzer` among `options`)
    whose day-1 session of `strategy` (one or a list) has run to NSE's close and carried
    its legs; the outcome is `world.first`. With `future` (a symbol of this module), the
    fake also trades it at `FUTURE_PRICE` every 10 s from the start, as it does SBIN."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **options)
    world.future = future
    if future is not None:
        add_future(world, future)
    world.first = world.run(strategy)
    world.day = DAY1
    return world


def add_future(world: Any, symbol: str) -> None:
    """The fake's future `symbol` (`list_future`) and its trades every 10 s from Monday's
    start to 15:32."""
    list_future(world, symbol)
    _future_trades(world, on(DAY1, *START), on(DAY1, *TRADES_UNTIL))


def list_future(world: Any, symbol: str) -> None:
    """The fake lists the future `symbol` with its candles of the history days and of
    Monday to the start, at a quarter of SBIN's; from now on the days `to_day` and
    `missed_day` move to seed and trade it as the world's future."""
    world.fake.add_instrument(
        symbol,
        "NFO",
        lot_size=FUTURE_LOT,
        tick_size=0.05,
        instrument_type="FUT",
        expiry=_EXPIRIES[symbol],
    )
    seeded = candles(today_until=(14, 40))
    rows = [(t, o / 4, h / 4, low / 4, c / 4, v) for t, o, h, low, c, v in seeded]
    world.fake.seed_candles(symbol, "NFO", rows)
    world.future = symbol


def _future_trades(world: Any, start_ns: int, end_ns: int) -> None:
    fake, symbol = world.fake, world.future
    fake.trade(symbol, "NFO", FUTURE_PRICE, volume=FUTURE_LOT)
    for instant in range(start_ns + TRADE_EVERY_S * NS, end_ns, TRADE_EVERY_S * NS):
        world.script.at(instant, lambda: fake.trade(symbol, "NFO", FUTURE_PRICE, volume=FUTURE_LOT))


def flat_minutes(start_ns: int, end_ns: int, price: float = 500.0) -> list[tuple]:
    """Broker candles of SBIN at `price` for each minute from `start_ns` to `end_ns`."""
    return [(t, price, price, price, price, 100.0) for t in range(start_ns, end_ns, MINUTE_NS)]


def missed_day(world: Any, day: dt.date, prices: tuple[tuple[tuple[int, int], float], ...]) -> None:
    """End the fake's current day, then play `day` with no session: SBIN trades at each
    (IST (hour, minute), price) of `prices`, which may fire a GTT, and the day's broker
    candles are flat at 500.00 from the open to the close."""
    _next(world, day)
    fake = world.fake
    for (hour, minute), price in prices:
        world.script.sleep((on(day, hour, minute) - fake.clock.now()) / NS)
        fake.trade("SBIN", "NSE", price, volume=10.0)


def to_day(world: Any, day: dt.date, *, start: tuple[int, int, int] = START) -> None:
    """End the fake's current day and move to `day`'s session start (module docstring),
    or to the IST time `start`, SBIN trading every 10 s from then at the script's price."""
    _next(world, day, start)
    fake = world.fake
    start_ns = on(day, *start)
    world.script.sleep((start_ns - fake.clock.now()) / NS)
    world.script.trade()
    world.script.trades(start_ns + TRADE_EVERY_S * NS, on(day, *TRADES_UNTIL))
    if getattr(world, "future", None) is not None:
        _future_trades(world, start_ns, on(day, *TRADES_UNTIL))


def _next(world: Any, day: dt.date, start: tuple[int, int, int] = START) -> None:
    """End the fake's day and seed flat candles of SBIN (and of `FUTURE`, which the
    world may trade) from the last day's start to `day`'s session start (none of `day`
    for a start before the open)."""
    fake, calendar = world.fake, default_calendar()
    fake.end_day()
    _, last_close = calendar.session("NSE", world.day)
    opens, _ = calendar.session("NSE", day)
    priced = [("SBIN", "NSE", 500.0)]
    if getattr(world, "future", None) is not None:
        priced.append((world.future, "NFO", FUTURE_PRICE))
    for symbol, exchange, price in priced:
        rows = flat_minutes(on(world.day, 14, 40), last_close, price)
        rows += flat_minutes(opens, on(day, *start[:2]), price)
        fake.seed_candles(symbol, exchange, rows)
    world.day = day


def held(world: Any) -> int:
    """SBIN's units at the fake, as the broker shows a CNC leg's: its holdings and its CNC
    position of the day together."""
    return world.fake.holding("SBIN", "NSE") + world.fake.position("SBIN", "NSE")


def journal_of(world: Any, mode: Mode | None = None) -> Journal:
    """The account's journal in the ledger of `mode` (the fake's by default)."""
    folder = world.folder / "live" / BROKER
    return Journal.open(
        folder / JOURNAL_FILE_NAME,
        account=BROKER,
        broker=BROKER,
        mode=world.fake.mode if mode is None else mode,
        now=world.fake.clock.now,
    )


__all__ = [
    "AFTER_NOVEMBERS_LAST",
    "DAY1",
    "DAY2",
    "DAY3",
    "DAY4",
    "FUTURE",
    "FUTURE_LOT",
    "FUTURE_PRICE",
    "NOVEMBER_FUTURE",
    "TO_NOVEMBERS_LAST",
    "CncKept",
    "CncKeptTrail",
    "FutKept",
    "NovemberKept",
    "carried_world",
    "flat_minutes",
    "held",
    "journal_of",
    "add_future",
    "list_future",
    "missed_day",
    "on",
    "to_day",
]
