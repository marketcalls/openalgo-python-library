"""The market, strategies and helpers shared by the session plan's tests
(`test_plan*.py`; `_live/plan.py`, 16-session-design unit 7).

The market: SBIN on NSE, the live day Monday 2026-11-16, the fake OpenAlgo's day. Its
broker serves 1-minute candles of the two sessions before it (Thursday 2026-11-12 and
Friday 2026-11-13) and of today from the open up to a time the test names: a seeded
random walk from 500.00 (each candle opens at the last close, moves by a normal step of
0.40 rupees rounded to the paisa, high and low 0.30 beyond open and close, 1,000 units).
The candles are put in as the broker's own (`FakeOpenAlgo.seed_candles`), so `history`
answers them whatever the fake's clock. Expected values come from a backtest of the same
day on the same fake, never from the plan's code.
"""

from __future__ import annotations

import datetime as dt
import inspect
from typing import Any

import numpy as np
from fake_openalgo import FakeOpenAlgo, ist_ns

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live import plan as plan_module
from openalgo.backtest._live._preflight_plan import InstrumentFacts
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data.fetching import HistoryFetcher

DAY = dt.date(2026, 11, 16)
HISTORY_DAYS = (dt.date(2026, 11, 12), dt.date(2026, 11, 13))
MINUTE_NS = 60 * 10**9
NS = 10**9
#: OpenAlgo's facts of SBIN as the preflight reads them: those of the generic
#: cash-equity instrument the reference tables assume, so a backtest's equal them.
SBIN_FACTS = InstrumentFacts("SBIN", "NSE", lot_size=1, tick_size=0.05, freeze_quantity=None)
LIMITS = {"capital": 1_000_000, "max_daily_loss": 5_000}


def ist(hour: int, minute: int, second: int = 0, day: dt.date = DAY) -> int:
    """UTC nanoseconds of an IST wall-clock time of `day`."""
    return ist_ns(day.year, day.month, day.day, hour, minute, second)


def candles(
    days: tuple[dt.date, ...] = HISTORY_DAYS,
    *,
    today_until: tuple[int, int] | None = (10, 15),
    skip: tuple[int, ...] = (),
    seed: int = 3,
) -> list[tuple[int, float, float, float, float, float]]:
    """The candles of the module docstring: every minute of each of `days`, then of
    today from the open to the IST (hour, minute) `today_until` (excluded; None for
    none), leaving out the minutes opening at `skip` (UTC nanoseconds). Each (start UTC
    nanoseconds, open, high, low, close, volume)."""
    calendar = default_calendar()
    rng = np.random.default_rng(seed)
    rows, price = [], 500.0
    sessions = [(day, None) for day in days]
    if today_until is not None:
        sessions.append((DAY, ist(*today_until)))
    for day, until in sessions:
        opens, closes = calendar.session("NSE", day)
        end = closes if until is None else until
        for start in range(opens, end, MINUTE_NS):
            open_ = price
            close = round(open_ + rng.normal(0.0, 0.4), 2)
            price = close
            if start not in skip:
                rows.append(
                    (start, open_, max(open_, close) + 0.3, min(open_, close) - 0.3, close, 1_000.0)
                )
    return rows


def market(
    make_fake_openalgo, monkeypatch, tmp_path, *, at: int, rows: list | None = None, **seeded: Any
) -> FakeOpenAlgo:
    """A fake OpenAlgo at the instant `at` (UTC nanoseconds) serving SBIN's candles
    (`rows`, by default `candles(**seeded)`), the data layer's cache in `tmp_path`. The
    data layer's pacing between two history requests waits for nothing: the fake's clock
    is the test's, and its rate limits are counted on it."""
    monkeypatch.setenv("OPENALGO_BT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(HistoryFetcher, "_wait", lambda fetcher, seconds: None)
    fake = make_fake_openalgo(start_ns=at)
    fake.add_instrument("SBIN", "NSE")
    fake.seed_candles("SBIN", "NSE", candles(**seeded) if rows is None else rows)
    return fake


class EmaTrend(bt.Strategy):
    """A 5-minute strategy with a 10-bar EMA, which needs 128 bars of warm-up on the
    engine's reference bars: the two sessions before the day hold 150."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        self.fast = self.I(ta.ema, self.close, 10)

    def on_bar(self, bar):
        if self.fast[-1] < bar.close:
            self.buy(quantity=1)


def specify(strategy: Any = EmaTrend, fake: FakeOpenAlgo | None = None, **arguments: Any):
    """`plan.specify` of `strategy` with the session's limits, on the fake's clock (a
    clock at the module's day's 10:15 without one)."""
    clock = fake.clock.now if fake is not None else lambda: ist(10, 15)
    return plan_module.specify(strategy, clock=clock, **(LIMITS | arguments))


def build(
    fake: FakeOpenAlgo,
    strategy: Any = EmaTrend,
    *,
    facts=(SBIN_FACTS,),
    user_states=None,
    **arguments: Any,
):
    """The plan of `strategy` built on the fake, with OpenAlgo's `facts` and the saved
    `user_states`."""
    spec = specify(strategy, fake, **arguments)
    return plan_module.build(
        spec, fake.client(), facts=facts, clock=fake.clock.now, user_states=user_states
    )


def backtest_arguments(strategy: Any, client: Any, present_ns: int, **settings: Any) -> dict:
    """The arguments `bt.backtest` runs with for `strategy` on the module's day, read from
    its own signature with its own defaults (as `bt.backtest` passes `dict(locals())` on),
    with broker history, a 1-minute base and the present `present_ns`."""
    bound = inspect.signature(bt.backtest).bind(
        strategy,
        client,
        start=DAY,
        end=DAY,
        source="api",
        base="1m",
        now=present_ns,
        **(LIMITS | settings),
    )
    bound.apply_defaults()
    return dict(bound.arguments)


__all__ = [
    "DAY",
    "HISTORY_DAYS",
    "LIMITS",
    "MINUTE_NS",
    "NS",
    "SBIN_FACTS",
    "EmaTrend",
    "backtest_arguments",
    "build",
    "candles",
    "ist",
    "market",
    "specify",
]
