"""The fixed strategy set of the golden fixtures (NFR-COR-02), and how a result is
written down for them.

Four runs, each a different path through the engine:

    intraday_mis       an `on_bar` EMA crossover on 5-minute bars of a 1-minute base,
                       long and short on MIS with a stop and a target, two sessions
    condition_cnc      a condition strategy (`entry_when`, `exit_when`) on daily bars,
                       delivery (CNC), about seven months after thirteen months of
                       history that its indicators are computed over
    futures_roll       an `on_bar` strategy on the NIFTY futures series, daily, across
                       the roll from the September to the October 2024 contract
    gtt_positional     a positional strategy on daily bars that buys a dip with a GTT
                       order and holds its stop and target as GTTs (CNC)

Every run charges the default costs and slippage. The bars are made from integers only
(a 64-bit linear congruential generator walking in ticks), so they are the same bits on
every platform; prices are ticks divided by 20 (tick 0.05).

A result is written down as plain JSON types: the equity table, the trades table and
the metrics. Times are UTC nanoseconds or ISO strings; floats are kept as floats, which
`json` writes in the shortest form that reads back to the same bits. The pins live in
`golden/<run>.json`; only `golden/regenerate.py --write` writes them.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.reference import Instrument

# A full-period 64-bit linear congruential generator (the widely published multiplier
# and increment for modulus 2**64): integer arithmetic only.
LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MODULUS = 2**64
LCG_DROPPED_BITS = 33  # the low bits of an LCG are poor; use the high ones
TICKS_PER_RUPEE = 20  # tick 0.05
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
# (platform.system(), platform.machine()) of the platform the pins are made on.
REFERENCE_PLATFORM = ("Windows", "AMD64")


class TickWalk:
    """Bars as a random walk in ticks, from a seed, with integers only."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def draw(self, low: int, high: int) -> int:
        """A whole number from `low` to `high`, inclusive."""
        self.state = (LCG_MULTIPLIER * self.state + LCG_INCREMENT) % LCG_MODULUS
        return low + (self.state >> LCG_DROPPED_BITS) % (high - low + 1)

    def bars(self, stamps: list, start_ticks: int, step: int, drift_every: int) -> pd.DataFrame:
        """One bar per stamp: the close moves up to `step` ticks a bar with a drift of
        one tick that changes sign every `drift_every` bars."""
        rows, close = [], start_ticks
        for k in range(len(stamps)):
            drift = 1 if (k // drift_every) % 2 == 0 else -1
            open_ = close + self.draw(-2, 2)
            close = open_ + drift + self.draw(-step, step)
            high = max(open_, close) + self.draw(0, step)
            low = min(open_, close) - self.draw(0, step)
            rows.append([value / TICKS_PER_RUPEE for value in (open_, high, low, close)])
        frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        frame["volume"] = 10_000.0
        frame.index = pd.DatetimeIndex(stamps, name="timestamp")
        return frame


def minute_stamps(days: list[dt.date]) -> list[dt.datetime]:
    """Every 1-minute bar open of the NSE session (09:15 to 15:29 IST) on `days`."""
    return [
        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=k)
        for day in days
        for k in range(375)
    ]


def session_dates(exchange: str, start: str, end: str) -> list[dt.datetime]:
    """The session dates of the built-in calendar, as IST midnights."""
    sessions = bt.calendar.default_calendar().sessions(exchange, start, end)
    return [dt.datetime.combine(day, dt.time(0), tzinfo=IST) for day in sessions["date"]]


def tick_price(rupees: float) -> float:
    """`rupees` rounded to the nearest tick."""
    return round(rupees * TICKS_PER_RUPEE) / TICKS_PER_RUPEE


# ----- intraday_mis --------------------------------------------------------------------


class IntradayEmaCross(bt.Strategy):
    """Long when the 5-bar EMA crosses above the 15-bar EMA, short when it crosses
    below, 100 shares with a 3-point stop and a 6-point target, flat by the square-off."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    product = "MIS"

    def init(self):
        self.fast = self.I(ta.ema, self.close, 5)
        self.slow = self.I(ta.ema, self.close, 15)

    def on_bar(self, bar):
        if len(self.close) < 15 or not self.position.is_flat:
            return
        protect = {"stop": bt.points(3), "target": bt.points(6)}
        if self.fast.crossed_above(self.slow):
            self.buy(qty=100, **protect)
        elif self.fast.crossed_below(self.slow):
            self.sell(qty=100, **protect)


def intraday_mis() -> bt.Result:
    # The 15-bar EMA needs 128 five-minute bars before the strategy decides (`_warm_up`):
    # the two sessions before the run, 2026-10-30 and 11-02 (a walk of their own from the
    # same price), are history, so it trades from the first bar of 11-03.
    before = [dt.date(2026, 10, 30), dt.date(2026, 11, 2)]
    history = TickWalk(seed=12).bars(minute_stamps(before), 16_000, 20, 45)
    days = [dt.date(2026, 11, 3), dt.date(2026, 11, 4)]
    bars = TickWalk(seed=11).bars(minute_stamps(days), 16_000, 20, 45)
    data = pd.concat([history, bars])
    return bt.backtest(IntradayEmaCross, data=data, start=days[0], capital=CAPITAL, base="1m")


# ----- condition_cnc -------------------------------------------------------------------


class ConditionEmaCross(bt.Strategy):
    """Conditions: buy 20 shares when the 5-day EMA crosses above the 20-day EMA, exit
    when it crosses below."""

    symbol, exchange, interval = "TCS", "NSE", "D"
    product = "CNC"

    def init(self):
        fast = self.I(ta.ema, self.close, 5)
        slow = self.I(ta.ema, self.close, 20)
        self.entry_when(fast.crossed_above(slow), qty=20)
        self.exit_when(fast.crossed_below(slow))


def condition_cnc() -> bt.Result:
    # The 20-day EMA needs 256 bars before the strategy decides (`_warm_up`): the 266
    # sessions from 2025-03-03 are history, so it trades from 2026-04-01. The walk of
    # seed 24 crosses often enough in the run to pin seven round trips.
    stamps = session_dates("NSE", "2025-03-03", "2026-10-30")
    bars = TickWalk(seed=24).bars(stamps, 70_000, 400, 25)
    return bt.backtest(ConditionEmaCross, data=bars, start=dt.date(2026, 4, 1), capital=CAPITAL)


# ----- futures_roll --------------------------------------------------------------------

SEPTEMBER_2024 = "NIFTY26SEP24FUT"
OCTOBER_2024 = "NIFTY31OCT24FUT"
# The October contract trades about 100 points (2,000 ticks) above September.
OCTOBER_PREMIUM_TICKS = 2_000


def nifty_contract(symbol: str, last_trading_day: dt.date) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="NFO",
        kind="future",
        lot_size=25,
        tick_size=0.05,
        expiry=last_trading_day,
        underlying="NIFTY",
        settlement="cash",
    )


class NiftyTrend(bt.Strategy):
    """Long one lot of the current-month NIFTY future while the close is above its
    3-session mean; rolls one session before expiry."""

    underlying, exchange, interval = "NIFTY", "NFO", "D"

    def init(self):
        self.mean = self.I(ta.sma, self.close, 3)

    def on_bar(self, bar):
        if len(self.close) < 3:
            return
        if self.position.is_flat and bar.close > self.mean[-1]:
            self.buy(lots=1)
        elif self.position.is_long and bar.close < self.mean[-1]:
            self.exit()


def futures_roll() -> bt.Result:
    stamps = session_dates("NFO", "2024-09-02", "2024-10-25")
    walk = TickWalk(seed=37).bars(stamps, 500_000, 2_000, 6)
    september = walk.loc[: dt.datetime(2024, 9, 26, tzinfo=IST)]
    october = walk.loc[dt.datetime(2024, 9, 25, tzinfo=IST) :].copy()
    prices = ["open", "high", "low", "close"]
    october[prices] += OCTOBER_PREMIUM_TICKS / TICKS_PER_RUPEE
    with warnings.catch_warnings():
        # Expected, and listed in the result's warnings: some of the rates charged in
        # September and October 2024 are not verified against their source.
        warnings.simplefilter("ignore", bt.costs.UnverifiedScheduleWarning)
        return bt.backtest(
            NiftyTrend,
            data={SEPTEMBER_2024: september, OCTOBER_2024: october},
            instrument={
                SEPTEMBER_2024: nifty_contract(SEPTEMBER_2024, dt.date(2024, 9, 26)),
                OCTOBER_2024: nifty_contract(OCTOBER_2024, dt.date(2024, 10, 31)),
            },
            start=dt.date(2024, 9, 2),
            end=dt.date(2024, 10, 25),
            capital=CAPITAL,
        )


# ----- gtt_positional ------------------------------------------------------------------


class GttDipBuyer(bt.Strategy):
    """While flat and above its 10-day mean, rests a GTT buy of 50 shares 1 percent
    below the close, with a 3 percent stop and a 5 percent target held as GTTs; a GTT
    that has not fired in 5 sessions is cancelled and placed again."""

    symbol, exchange, interval = "INFY", "NSE", "D"
    product = "CNC"

    def init(self):
        self.mean = self.I(ta.sma, self.close, 10)

    def on_bar(self, bar):
        if len(self.close) < 10 or not self.position.is_flat:
            return
        waiting = self.state.get("placed_at")
        if self.orders and waiting is not None and len(self.close) - waiting < 5:
            return
        self.cancel_all()
        self.state["placed_at"] = None
        if bar.close > self.mean[-1]:
            dip = tick_price(bar.close * 0.99)
            self.buy(qty=50, price=dip, validity="gtt", stop=bt.pct(3), target=bt.pct(5))
            self.state["placed_at"] = len(self.close)


def gtt_positional() -> bt.Result:
    stamps = session_dates("NSE", "2026-04-01", "2026-10-30")
    bars = TickWalk(seed=41).bars(stamps, 30_000, 500, 20)
    return bt.backtest(GttDipBuyer, data=bars, capital=CAPITAL)


RUNS: dict[str, Callable[[], bt.Result]] = {
    "intraday_mis": intraday_mis,
    "condition_cnc": condition_cnc,
    "futures_roll": futures_roll,
    "gtt_positional": gtt_positional,
}

# ----- writing a result down ------------------------------------------------------------

EQUITY_COLUMNS = (
    "equity_rupees",
    "realized_rupees",
    "unrealized_rupees",
    "costs_rupees",
    "margin_rupees",
    "exposure_rupees",
)
TRADE_COLUMNS = (
    "symbol",
    "product",
    "direction",
    "entry_ts",
    "exit_ts",
    "entry_units",
    "max_units",
    "entry_price_rupees",
    "exit_price_rupees",
    "gross_pnl_rupees",
    "costs_rupees",
    "net_pnl_rupees",
    "exit_reason",
    "fills",
)


def plain(value: Any) -> Any:
    """A value as a JSON type: times as ISO strings, NumPy scalars as Python ones."""
    if isinstance(value, pd.Timestamp | dt.datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def snapshot(result: bt.Result) -> dict[str, Any]:
    """The pinned part of a result: equity, trades and metrics, as JSON types."""
    equity = result.equity
    trades = result.trades
    return {
        "equity": {
            "ts": [int(stamp.value) for stamp in equity.index],
            **{name: [plain(v) for v in equity[name]] for name in EQUITY_COLUMNS},
        },
        "trades": {name: [plain(v) for v in trades[name]] for name in TRADE_COLUMNS},
        "metrics": {key: plain(value) for key, value in result.metrics.items()},
    }


def on_reference_platform() -> bool:
    """True on the platform the pins were made on: Windows on x86_64."""
    return (platform.system(), platform.machine()) == REFERENCE_PLATFORM


def pins_path(name: str) -> Path:
    """The file that pins run `name`."""
    return GOLDEN_DIR / f"{name}.json"


def read_pins(name: str) -> dict[str, Any]:
    """The pinned snapshot of run `name`."""
    return json.loads(pins_path(name).read_text(encoding="utf-8"))


def write_pins(name: str, pinned: dict[str, Any]) -> Path:
    """Write the snapshot of run `name` (only `golden/regenerate.py` calls this). NaN
    and infinities are written as the JSON extension tokens `json` reads back."""
    path = pins_path(name)
    path.write_text(json.dumps(pinned, indent=1) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CAPITAL",
    "GOLDEN_DIR",
    "REFERENCE_PLATFORM",
    "RUNS",
    "TickWalk",
    "on_reference_platform",
    "pins_path",
    "read_pins",
    "snapshot",
    "write_pins",
]
