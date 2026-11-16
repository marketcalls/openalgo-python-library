"""A live strategy series equals a backtest's (16-session-design unit 6): the same
strategy class, run once by `bt.backtest` over the bars of `series_samples` and once in
a live run context on a kernel session of the live day fed the same bars slice by
slice, sees the same bars, the same history, the same indicator values and the same
views built in `init`, at every decision. A broker candle that replaces a tick bar feeds
every later close, as a backtest of the candles would.

The strategy decides on 5-minute bars. Its 10-bar EMA needs 128 bars of warm-up on the
engine's reference bars, so the backtest keeps the two sessions before the live day
(150 bars) and decides from its first bar; the live run gets those sessions as its
history.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from series_samples import (
    IST,
    LIVE_DAY,
    LiveRun,
    backtest_views,
    before,
    live_bar,
    minute_bars,
    on_day,
)

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live._types import LiveBar

BARS = minute_bars()
TODAY = on_day(BARS, LIVE_DAY)


class EmaTrend(bt.Strategy):
    """Indicators and views of the kinds a strategy reads: a single output, several
    outputs, an indicator of high, low and close, views built in init, and another
    interval."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        self.fast = self.I(ta.ema, self.close, 10)
        self.slow = self.I(ta.sma, self.close, 20)
        self.strength = self.I(ta.rsi, self.close, 14)
        self.macd_line, self.signal_line, self.histogram = self.I(ta.macd, self.close, 12, 26, 9)
        self.band = self.I(ta.atr, self.high, self.low, self.close, 14)
        self.gap = self.fast - self.slow
        self.crossed = self.fast.crossed_above(self.slow)
        self.recent_high = self.high.rolling_max(6).shift(1)


VIEWS = ("close", "fast", "slow", "strength", "macd_line", "histogram", "band", "gap", "crossed")


def reading(strategy: bt.Strategy) -> dict:
    """What a decision of `EmaTrend` sees."""
    seen = {name: getattr(strategy, name).values for name in VIEWS}
    seen["recent_high"] = strategy.recent_high.values
    quarter_hours = strategy.data["SBIN", "15m"].close
    seen["quarter_hours"] = quarter_hours.values
    seen["labels"] = (strategy.fast.label, strategy.gap.label, quarter_hours.label)
    store = strategy._ctx.stores["SBIN"]
    seen["history"] = (store.history_bars, int(store.column("ts_open")[0]))
    return seen


def live_history(first_open: int) -> pd.DataFrame:
    """The base bars of the sessions the backtest kept, from the day of `first_open`."""
    history = before(BARS, LIVE_DAY)
    dates = pd.to_datetime(history["ts_open"], utc=True).dt.tz_convert(IST).dt.date
    first_day = pd.Timestamp(first_open, tz="UTC").tz_convert(IST).date()
    return history[dates >= first_day].reset_index(drop=True)


def live_views(strategy: type, rows: pd.DataFrame, history: pd.DataFrame) -> list[tuple]:
    """(decision instant, bar, reading) at every `on_bar` of a live run fed `rows` as
    tick bars."""
    seen: list[tuple] = []

    class Recording(strategy):
        def on_bar(self, bar):
            seen.append((self._ctx.decided_at, tuple(bar), reading(self)))

    run = LiveRun(Recording, history, interval_minutes=5)
    for row in rows.itertuples():
        run.close_slice(live_bar(row))
    return seen


@pytest.fixture(scope="module")
def backtest_seen() -> list[tuple]:
    return backtest_views(EmaTrend, BARS, reading)


@pytest.fixture(scope="module")
def live_seen(backtest_seen) -> list[tuple]:
    history = live_history(backtest_seen[0][2]["history"][1])
    return live_views(EmaTrend, TODAY, history)


def test_the_backtest_decides_at_every_bar_of_the_live_day(backtest_seen):
    assert len(backtest_seen) == len(TODAY) // 5


def test_the_live_run_decides_at_the_same_instants(backtest_seen, live_seen):
    assert [seen[0] for seen in live_seen] == [seen[0] for seen in backtest_seen]


def test_the_live_run_is_handed_the_same_bars(backtest_seen, live_seen):
    assert [seen[1] for seen in live_seen] == [seen[1] for seen in backtest_seen]


def test_the_live_store_holds_the_same_history(backtest_seen, live_seen):
    assert live_seen[0][2]["history"] == backtest_seen[0][2]["history"]


def test_views_are_labelled_as_in_the_backtest(backtest_seen, live_seen):
    # The indicators' warm-ups are keyed on these labels.
    assert live_seen[0][2]["labels"] == backtest_seen[0][2]["labels"]


@pytest.mark.parametrize("name", [*VIEWS, "recent_high", "quarter_hours"])
def test_every_view_equals_the_backtests_at_every_decision(backtest_seen, live_seen, name):
    for (when, _, backtest), (_, _, live) in zip(backtest_seen, live_seen, strict=True):
        np.testing.assert_array_equal(live[name], backtest[name], err_msg=f"at {when}")


def test_indicators_recomputed_per_close_equal_their_values_over_the_whole_series(live_seen):
    # The EMA over every close the live run saw by the end of the day, prefix by prefix.
    final = live_seen[-1][2]
    whole = ta.ema(final["close"], 10)
    for _, _, seen in live_seen:
        np.testing.assert_array_equal(seen["fast"], whole[: len(seen["fast"])])


def tick_bar_of(row) -> LiveBar:
    """The bar the ticks of a slice made: its close five paise above the broker's candle
    (its row of `BARS`), as when the feed missed the slice's last trade."""
    bar = live_bar(row)
    close = round(bar.close + 0.05, 2)
    return dataclasses.replace(bar, close=close, high=max(bar.high, close))


def test_candles_feed_every_later_close_as_a_backtest_of_the_candles(backtest_seen):
    """Each slice closes on its tick bar and its candle arrives after the decision of
    that close, as the candle check reads it at the close plus 2 s. At every decision
    the bars before the latest are the candles' bars, with the indicator values a
    backtest of the candles computes; the latest bar still ends on its tick bar."""
    seen: list[tuple] = []

    class Recording(EmaTrend):
        def on_bar(self, bar):
            seen.append((tuple(bar), self.close.values, self.fast.values))

    run = LiveRun(Recording, live_history(backtest_seen[0][2]["history"][1]), 5)
    for row in TODAY.itertuples():
        run.close_slice(tick_bar_of(row))
        run.series.take(live_bar(row, "candle"))
    for (bar, closes, fast), (_, _, backtest) in zip(seen, backtest_seen, strict=True):
        np.testing.assert_array_equal(closes[:-1], backtest["close"][:-1])
        np.testing.assert_array_equal(fast[:-1], backtest["fast"][:-1])
        assert closes[-1] == bar[6] == round(backtest["close"][-1] + 0.05, 2)
