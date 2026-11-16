"""`self.I` indicators and views on live stores (`_live/_series_indicators.py`;
16-session-design unit 6): registered in `init` only, causal outputs only (the registry
`openalgo.indicators.causality`), recomputed once per change of the store, and refused
over values of the user's own, which a live session has no counterpart for.

The runs are `series_samples.LiveRun`: a 1-minute strategy on a kernel session of the
live day, with the session before it as history.
"""

from __future__ import annotations

import numpy as np
import pytest
from series_samples import LIVE_DAY, LiveRun, before, live_bar, minute_bars, on_day

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live._series_indicators import LiveView
from openalgo.backtest.errors import ConfigError, IndicatorOutsideInitError

BARS = minute_bars()
TODAY = on_day(BARS, LIVE_DAY)
HISTORY = before(BARS, LIVE_DAY)


class OneMinute(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"


def live_run(strategy: type, slices: int = 0, history=HISTORY) -> LiveRun:
    """A live run of `strategy` with today's first `slices` slices closed."""
    run = LiveRun(strategy, history, interval_minutes=1)
    for row in TODAY.iloc[:slices].itertuples():
        run.close_slice(live_bar(row))
    return run


def test_an_indicator_registered_after_init_is_refused():
    class Late(OneMinute):
        def on_bar(self, bar):
            self.I(ta.ema, self.close, 10)

    with pytest.raises(IndicatorOutsideInitError, match="after init"):
        live_run(Late, slices=1)


def test_an_indicator_registered_in_init_is_a_live_view():
    class Ema(OneMinute):
        def init(self):
            self.line = self.I(ta.ema, self.close, 10)

    assert isinstance(live_run(Ema).strategy.line, LiveView)


def test_an_indicator_whose_outputs_all_look_ahead_is_refused():
    class Fractals(OneMinute):
        def init(self):
            self.I(ta.fractals, self.high, self.low)

    with pytest.raises(ConfigError, match="looks ahead"):
        live_run(Fractals)


def test_a_look_ahead_output_of_an_indicator_raises_on_any_use():
    class Cloud(OneMinute):
        def init(self):
            *self.causal, self.lagging = self.I(ta.ichimoku, self.high, self.low, self.close)

        def on_bar(self, bar):
            self.lagging[-1]

    with pytest.raises(ConfigError, match="lagging_span"):
        live_run(Cloud, slices=1)


def test_the_causal_outputs_of_an_indicator_with_a_look_ahead_one_follow_the_bars():
    class Cloud(OneMinute):
        def init(self):
            *self.causal, self.lagging = self.I(ta.ichimoku, self.high, self.low, self.close)

    run = live_run(Cloud, slices=3)
    conversion = run.strategy.causal[0]
    assert isinstance(conversion, LiveView)
    assert len(conversion) == len(HISTORY) + 3


def test_an_indicator_over_values_of_your_own_is_refused():
    class Own(OneMinute):
        def init(self):
            self.I(ta.crossover, self.close, np.ones(len(HISTORY)))

    with pytest.raises(NotImplementedError, match="values of your own"):
        live_run(Own)


def test_an_indicator_too_short_for_the_history_stops_the_start():
    class Long(OneMinute):
        def init(self):
            self.line = self.I(ta.sma, self.close, 100)

    with pytest.raises(ConfigError, match="too short to compute sma"):
        live_run(Long, slices=1, history=HISTORY.iloc[:30])


def test_an_indicator_is_recomputed_once_per_change_of_the_store():
    class Ema(OneMinute):
        def init(self):
            self.line = self.I(ta.ema, self.close, 10)

    run = live_run(Ema, slices=2)
    line = run.strategy.line
    assert line._values is line._values
    first = line._values
    run.close_slice(live_bar(TODAY.iloc[2]))
    assert line._values is not first


def test_an_indicator_reads_the_new_bar_after_each_close():
    class Ema(OneMinute):
        def init(self):
            self.line = self.I(ta.ema, self.close, 10)

    run = live_run(Ema, slices=3)
    closes = np.concatenate((HISTORY.close.to_numpy(), TODAY.close.to_numpy()[:3]))
    assert run.strategy.line[-1] == ta.ema(closes, 10)[-1]


def test_a_view_built_in_init_follows_the_bars():
    class Gap(OneMinute):
        def init(self):
            self.gap = self.close - self.open

    run = live_run(Gap, slices=2)
    assert run.strategy.gap[-1] == TODAY.close.iloc[1] - TODAY.open.iloc[1]


def test_a_clock_condition_meeting_a_live_view_follows_the_bars():
    class Morning(OneMinute):
        def init(self):
            self.morning = (self.close > 0) & bt.time_between("09:15", "09:17")

    run = live_run(Morning, slices=3)
    assert list(run.strategy.morning.values[-3:]) == [True, False, False]
