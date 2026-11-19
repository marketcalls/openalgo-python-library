"""The three seeded synthetic runs of the golden tearsheet pages (`test_report_golden.py`),
and how a page is written down for them.

    equity      an `on_bar` EMA crossover on daily SBIN bars, long only, with a stop
                (so the page has R-multiples), about a year of sessions after the
                history its indicators warm up on, against a benchmark given as closes
    futures     an `on_bar` trend strategy on the NIFTY futures series, daily, across
                the rolls of ten monthly contracts of 2024
    combined    two daily strategies (the SBIN crossover and one on INFY) on fixed
                allocations, combined with `metrics.combine_equity` and handed to
                `report.tearsheet` with their `strategies`, as a combined result is

The bars are those of the golden runs (`golden_runs.TickWalk`: a 64-bit linear
congruential walk in ticks, integers only), so they are the same bits on every platform.
Each run is small (a few hundred sessions, a few dozen trades) so the twelve pages, one
per run and report mode, stay small (`MAX_TOTAL_BYTES`).

A page is written down whole, with the SDK's version in its footer replaced by
`VERSION_TEXT`, so a version bump does not move it. The pages live in
`golden/tearsheets/<run>.<mode>.html`; only `golden/tearsheets/regenerate.py --write`
writes them.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from golden_runs import IST, TICKS_PER_RUPEE, TickWalk, nifty_contract, session_dates

import openalgo.backtest as bt
from openalgo import __version__, ta
from openalgo.backtest import metrics
from openalgo.backtest.report import MODES
from openalgo.backtest.report._performance import Performance

CAPITAL = 1_000_000
#: The text that stands for the SDK's version in a golden page's footer.
VERSION_TEXT = "golden"
#: Where the pages live.
PAGES_DIR = Path(__file__).resolve().parent / "golden" / "tearsheets"
#: The twelve pages together stay under this many bytes (the lead's limit, 2 MB).
MAX_TOTAL_BYTES = 2_000_000


# ----- equity --------------------------------------------------------------------------


class SbinCross(bt.Strategy):
    """Long 1,000 SBIN when the 3-day EMA crosses above the 8-day EMA, with a 1.5 percent
    stop; out when it crosses below."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    product = "CNC"

    def init(self):
        self.fast = self.I(ta.ema, self.close, 3)
        self.slow = self.I(ta.ema, self.close, 8)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast.crossed_above(self.slow):
            self.buy(qty=1_000, stop=bt.pct(1.5))
        elif self.position.is_long and self.fast.crossed_below(self.slow):
            self.exit()


class InfyCross(bt.Strategy):
    """Long 300 INFY when the 4-day SMA crosses above the 10-day SMA; out when it crosses
    below."""

    symbol, exchange, interval = "INFY", "NSE", "D"
    product = "CNC"

    def init(self):
        self.fast = self.I(ta.sma, self.close, 4)
        self.slow = self.I(ta.sma, self.close, 10)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast.crossed_above(self.slow):
            self.buy(qty=300)
        elif self.position.is_long and self.fast.crossed_below(self.slow):
            self.exit()


#: The equity runs' history starts here; trading starts at `EQUITY_START`.
EQUITY_HISTORY = ("2024-10-01", "2026-03-31")
EQUITY_START = dt.date(2025, 4, 1)


def nifty_benchmark() -> pd.Series:
    """NIFTY closes as a walk of their own, indexed by naive session dates."""
    stamps = session_dates("NSE", *EQUITY_HISTORY)
    walk = TickWalk(seed=53).bars(stamps, 460_000, 1_500, 30)
    dates = pd.DatetimeIndex([stamp.date() for stamp in stamps])
    return pd.Series(walk["close"].to_numpy(), index=dates, name="close")


def sbin_run(benchmark: pd.Series | None) -> bt.Result:
    stamps = session_dates("NSE", *EQUITY_HISTORY)
    bars = TickWalk(seed=69).bars(stamps, 16_000, 160, 9)
    return bt.backtest(
        SbinCross, data=bars, start=EQUITY_START, capital=CAPITAL, benchmark=benchmark
    )


def infy_run() -> bt.Result:
    stamps = session_dates("NSE", *EQUITY_HISTORY)
    bars = TickWalk(seed=67).bars(stamps, 30_000, 260, 11)
    return bt.backtest(InfyCross, data=bars, start=EQUITY_START, capital=CAPITAL)


def equity() -> bt.Result:
    return sbin_run(nifty_benchmark())


# ----- futures -------------------------------------------------------------------------


class NiftyTrend(bt.Strategy):
    """Long one lot of the current-month NIFTY future while the close is above its
    5-session mean; rolls one session before expiry."""

    underlying, exchange, interval = "NIFTY", "NFO", "D"

    def init(self):
        self.mean = self.I(ta.sma, self.close, 5)

    def on_bar(self, bar):
        if self.position.is_flat and bar.close > self.mean[-1]:
            self.buy(lots=1)
        elif self.position.is_long and bar.close < self.mean[-1]:
            self.exit()


#: The ten monthly contracts of 2024, by their last trading day (the last Thursday).
EXPIRIES = (
    dt.date(2024, 1, 25),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 28),
    dt.date(2024, 4, 25),
    dt.date(2024, 5, 30),
    dt.date(2024, 6, 27),
    dt.date(2024, 7, 25),
    dt.date(2024, 8, 29),
    dt.date(2024, 9, 26),
    dt.date(2024, 10, 31),
)
#: Each later contract trades this many ticks above the one before (25 rupees).
PREMIUM_TICKS = 500
FUTURES_SPAN = ("2024-01-01", "2024-10-25")


def contract_symbol(expiry: dt.date) -> str:
    """The contract's symbol, as "NIFTY25JAN24FUT"."""
    return f"NIFTY{expiry:%d%b%y}FUT".upper()


def futures() -> bt.Result:
    stamps = session_dates("NFO", *FUTURES_SPAN)
    walk = TickWalk(seed=71).bars(stamps, 430_000, 1_600, 7)
    prices = ["open", "high", "low", "close"]
    data, instruments = {}, {}
    since = stamps[0]
    for k, expiry in enumerate(EXPIRIES):
        symbol = contract_symbol(expiry)
        last = dt.datetime.combine(expiry, dt.time(0), tzinfo=IST)
        bars = walk.loc[since:last].copy()
        bars[prices] += k * PREMIUM_TICKS / TICKS_PER_RUPEE
        data[symbol] = bars
        instruments[symbol] = nifty_contract(symbol, expiry)
        # The next contract's bars start on the session before this one's expiry, the
        # roll session.
        since = walk.loc[:last].index[-2]
    with warnings.catch_warnings():
        # Expected, and listed in the result's warnings: some of the rates charged in
        # 2024 are not verified against their source, and a futures series is built from
        # `start`, so the strategy waits inside the run for its mean.
        warnings.simplefilter("ignore", bt.costs.UnverifiedScheduleWarning)
        warnings.simplefilter("ignore", bt.errors.WarmUpWarning)
        return bt.backtest(
            NiftyTrend,
            data=data,
            instrument=instruments,
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 10, 25),
            capital=CAPITAL,
        )


# ----- combined ------------------------------------------------------------------------


@dataclass(frozen=True)
class Combined:
    """A combined result as `report.tearsheet` reads it: the account's performance, its
    name, the strategies' performances and the assumption lines."""

    performance: Performance
    name: str
    strategies: dict[str, Performance]
    assumptions: list[str] = field(default_factory=list)


def combined() -> Combined:
    runs = {"SBIN cross": sbin_run(None), "INFY cross": infy_run()}
    equities = {name: run.equity["equity_rupees"] for name, run in runs.items()}
    capitals = {name: CAPITAL for name in runs}
    account = metrics.combine_equity(equities, opening_wealth=capitals)
    trades = pd.concat([run.performance.trades for run in runs.values()], ignore_index=True)
    performance = metrics.analyse(
        account, exchanges="NSE", opening_wealth=sum(capitals.values()), trades=trades
    )
    return Combined(
        performance=performance,
        name="Two crossovers",
        strategies={name: run.performance for name, run in runs.items()},
        assumptions=["allocation: Rs 10,00,000 to each strategy"],
    )


RUNS: dict[str, Callable[[], Any]] = {
    "equity": equity,
    "futures": futures,
    "combined": combined,
}


# ----- writing a page down --------------------------------------------------------------


def default_page(run: Any, mode: str) -> str:
    """The run's default page in `mode`: a `Result`'s own tearsheet, a combined result's
    through `report.tearsheet`."""
    if isinstance(run, bt.Result):
        return run.tearsheet(mode=mode)
    return bt.report.tearsheet(run, mode=mode)


def pinned(page: str) -> str:
    """A page as it is pinned: the SDK's version in its footer replaced by
    `VERSION_TEXT`."""
    footer = f"<span class='nums'>{__version__}</span>"
    if page.count(footer) != 1:
        raise AssertionError("a tearsheet's footer carries the SDK's version exactly once")
    return page.replace(footer, f"<span class='nums'>{VERSION_TEXT}</span>")


def page_path(name: str, mode: str) -> Path:
    """The golden page of run `name` in `mode`."""
    return PAGES_DIR / f"{name}.{mode}.html"


def read_page(name: str, mode: str) -> str:
    """The golden page of run `name` in `mode`, as text."""
    return page_path(name, mode).read_bytes().decode("utf-8")


def write_page(name: str, mode: str, page: str) -> Path:
    """Write the golden page (only `golden/tearsheets/regenerate.py` calls this)."""
    path = page_path(name, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(page.encode("utf-8"))
    return path


__all__ = [
    "MAX_TOTAL_BYTES",
    "MODES",
    "PAGES_DIR",
    "RUNS",
    "VERSION_TEXT",
    "default_page",
    "page_path",
    "pinned",
    "read_page",
    "write_page",
]
