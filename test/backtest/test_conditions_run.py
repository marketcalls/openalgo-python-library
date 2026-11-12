"""Condition strategies through `bt.backtest` (14-strategy-api 4): the declaration rules,
the run in Rust with no Python call per bar, and `bt.from_arrays` and `bt.from_signals`
giving the same result as the same logic written as conditions.

The hand-computed runs use nine 5-minute SBIN bars on 2025-07-01 from 09:15 IST: bar k
opens at 500 + k rupees and closes half a rupee higher (see `steps`).
"""

import datetime as dt
import sys

import numpy as np
import pandas as pd
import pytest
from condition_runs import IST, assert_same_result, bars_from_closes, run, walk_bars

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.errors import ConfigError

STEPS = 9


def steps() -> pd.DataFrame:
    """Bar k (k = 0..8) opens at 09:15 + 5k minutes at 500 + k and closes at 500.5 + k."""
    times = [
        dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST) + dt.timedelta(minutes=5 * k)
        for k in range(STEPS)
    ]
    opens = [500.0 + k for k in range(STEPS)]
    return bars_from_closes(opens, [price + 0.5 for price in opens], times)


def at(hour: int, minute: int) -> dt.datetime:
    return dt.datetime(2025, 7, 1, hour, minute, tzinfo=IST)


def run_steps(strategy: type, **settings):
    return run(strategy, steps(), end=at(9, 55), **settings)


class Levels(bt.Strategy):
    """Buys 10 while the close is above 502 and exits while it is above 505."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        self.entry_when(self.close > 502, qty=10)
        self.exit_when(self.close > 505)


def test_conditions_are_level_triggered_and_fill_at_the_next_open():
    # Closes are 500.5 + k. Bar 2 (close 502.5) enters: fill at the 09:30 open, 503.
    # Bar 5 (close 505.5) exits: 09:45 open, 506. Bar 6 is flat with the entry still
    # true: 09:50 open, 507. Bar 7 is long with the exit true: 09:55 open, 508.
    fills = run_steps(Levels).fills
    assert list(fills["time"]) == [at(9, 30), at(9, 45), at(9, 50), at(9, 55)]
    assert list(fills["side"]) == ["BUY", "SELL", "BUY", "SELL"]
    assert list(fills["price_rupees"]) == [503.0, 506.0, 507.0, 508.0]
    assert list(fills["reason"]) == ["entry", "signal", "entry", "signal"]


def test_pyramiding_adds_while_the_entry_stays_true():
    class Adds(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"
        pyramiding = 2

        def init(self):
            self.entry_when(self.close > 502, qty=10)

    # Entry at bar 2 (fills 09:30), adds at bars 3 and 4 (09:35, 09:40), then no more.
    fills = run_steps(Adds).fills
    assert list(fills["time"]) == [at(9, 30), at(9, 35), at(9, 40)]
    assert list(fills["reason"]) == ["entry", "add", "add"]


def test_a_short_condition_enters_short_and_cover_exits():
    class Shorts(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.short_when(self.close > 502, qty=10)
            self.cover_when(self.close > 504)

    # Short at bar 2 (09:30 open, 503); cover at bar 4 (close 504.5): 09:40 open, 505.
    fills = run_steps(Shorts).fills
    assert list(fills["side"])[:2] == ["SELL", "BUY"]
    assert list(fills["price_rupees"])[:2] == [503.0, 505.0]


def test_on_stop_runs_once_at_the_end():
    stopped = []

    class Stops(Levels):
        def on_stop(self):
            stopped.append(len(self.close))

    run_steps(Stops)
    assert stopped == [0]


def test_no_python_code_runs_while_the_rust_driver_runs_the_bars():
    """The whole grid runs inside the native `run_signals` call: a profiler sees no
    Python function called between its start and its return."""
    seen, inside, calls = [], [], []

    def profiler(frame, event, arg):
        name = getattr(arg, "__name__", "")
        if event == "c_call" and name == "run_signals":
            seen.append(name)
            inside.append(name)
        elif event in ("c_return", "c_exception") and name == "run_signals":
            inside.pop()
        elif event == "call" and inside:
            calls.append(frame.f_code.co_qualname)

    class UpBars(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(self.close > self.open, qty=10)
            self.exit_when(self.close < self.open)

    data = walk_bars([3, -2, 4, -5, 1])
    sys.setprofile(profiler)
    try:
        result = run(UpBars, data)
    finally:
        sys.setprofile(None)
    assert seen == ["run_signals"]
    assert calls == []
    assert len(result.fills) > 0


# ----------------------------------------------------------------- declaration rules


def test_a_strategy_with_on_bar_cannot_use_conditions():
    class Both(Levels):
        def on_bar(self, bar):
            pass

    with pytest.raises(ConfigError, match="on_bar or conditions, not both"):
        run_steps(Both)


def test_a_condition_strategy_cannot_have_event_callbacks():
    class Callbacks(Levels):
        def on_fill(self, fill):
            pass

    with pytest.raises(ConfigError, match="no Python call per event"):
        run_steps(Callbacks)


def test_a_condition_is_declared_once():
    class Twice(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(self.close > 502)
            self.entry_when(self.close > 503)

    with pytest.raises(ConfigError, match="declared once"):
        run_steps(Twice)


@pytest.mark.parametrize("argument", [{"price": 500.0}, {"validity": "gtt"}, {"reverse": True}])
def test_entry_arguments_outside_section_4_1_are_refused(argument):
    class Priced(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(self.close > 502, qty=10, **argument)

    with pytest.raises(ConfigError, match="MARKET DAY order"):
        run_steps(Priced)


@pytest.mark.parametrize(
    "argument",
    [{"stop": bt.atr(2, 3)}, {"trail": bt.trail_atr(1, 3)}, {"size": bt.size.atr_risk(1, 2, 3)}],
    ids=["atr-stop", "atr-trail", "atr-risk-sizer"],
)
def test_atr_order_arguments_are_not_built_for_conditions(argument):
    class Atr(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            self.entry_when(self.close > 502, **{"qty": 10, **argument})

    if "size" in argument:
        Atr.init = lambda self: self.entry_when(self.close > 502, **argument)
    with pytest.raises(NotImplementedError, match="condition order arguments"):
        run_steps(Atr)


def test_a_negative_pyramiding_is_refused():
    class Negative(Levels):
        pyramiding = -1

    with pytest.raises(ConfigError, match="pyramiding"):
        run_steps(Negative)


def test_a_condition_strategy_trades_one_symbol():
    class Pair(bt.Strategy):
        symbols, exchange, interval = ("SBIN", "INFY"), "NSE", "5m"

        def init(self):
            self.entry_when(self.data["SBIN"].close > 502)

    with pytest.raises(ConfigError, match="trades one symbol"):
        run(Pair, {"SBIN": steps(), "INFY": steps()}, end=at(9, 55))


def test_a_condition_strategy_cannot_have_timers():
    class Timed(Levels):
        def init(self):
            super().init()
            self.at("09:30", lambda: None)

    with pytest.raises(ConfigError, match="self.at or self.every"):
        run_steps(Timed)


# ----------------------------------------------------------------- other intervals


def fifteen_minute_trend(strategy):
    """True while the latest closed 15-minute bar closed above its open."""
    bars = strategy.data["SBIN", "15m"]
    return bars.close > bars.open


class TrendConditions(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        trend = fifteen_minute_trend(self)
        self.entry_when(trend, qty=10)
        self.exit_when(~trend)


class TrendOnBar(bt.Strategy):
    """The same rule as `TrendConditions`, written as the on_bar of 14-strategy-api 4.1."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        self.trend = fifteen_minute_trend(self)

    def on_bar(self, bar):
        if self.position.is_flat and self.trend:
            self.buy(qty=10)
        elif self.position.is_long and ~self.trend:
            self.exit()


def test_a_view_of_another_interval_decides_as_on_bar_reads_it():
    data = walk_bars([7, -3, -6, 2, 9, -8, -1, 5, -4])
    with_conditions = run(TrendConditions, data)
    assert len(with_conditions.fills) > 2
    assert_same_result(with_conditions, run(TrendOnBar, data))


# ----------------------------------------------------------------- arrays and functions

FAST, SLOW = 5, 12


class EmaCross(bt.Strategy):
    """The EMA crossing of 10-simple-path 3, written as conditions."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        fast = self.I(ta.ema, self.close, FAST)
        slow = self.I(ta.ema, self.close, SLOW)
        self.entry_when(fast.crossed_above(slow), qty=10, stop=bt.points(3))
        self.exit_when(fast.crossed_below(slow))


def crossings(frame: pd.DataFrame) -> tuple:
    fast, slow = ta.ema(frame.close, FAST), ta.ema(frame.close, SLOW)
    return ta.crossover(fast, slow), ta.crossunder(fast, slow)


DATA = walk_bars([6, 5, -3, 4, -8, -6, 3, -2, 7, -1, -9, 2])


def test_from_signals_gives_the_result_of_the_same_conditions():
    signals = bt.from_signals(crossings, "SBIN", "NSE", "5m", qty=10, stop=bt.points(3))
    assert_same_result(run(signals, DATA), run(EmaCross, DATA))
    assert len(run(signals, DATA).fills) > 2


def test_from_arrays_gives_the_result_of_the_same_conditions():
    entries, exits = crossings(DATA)
    arrays = bt.from_arrays(
        entries, exits, symbol="SBIN", exchange="NSE", interval="5m", qty=10, stop=bt.points(3)
    )
    assert_same_result(run(arrays, DATA), run(EmaCross, DATA))


def test_from_arrays_takes_a_series_indexed_by_bar_open():
    entries, exits = crossings(DATA)
    by_time = bt.from_arrays(
        pd.Series(entries, index=DATA.index),
        pd.Series(exits, index=DATA.index),
        symbol="SBIN",
        exchange="NSE",
        interval="5m",
        qty=10,
        stop=bt.points(3),
    )
    assert_same_result(run(by_time, DATA), run(EmaCross, DATA))


def test_from_arrays_shorts_and_covers_follow_the_short_branch():
    entries, exits = crossings(DATA)

    class ShortCross(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "5m"

        def init(self):
            fast = self.I(ta.ema, self.close, FAST)
            slow = self.I(ta.ema, self.close, SLOW)
            self.short_when(fast.crossed_below(slow), qty=10)
            self.cover_when(fast.crossed_above(slow))

    arrays = bt.from_arrays(
        None,
        None,
        shorts=exits,
        covers=entries,
        symbol="SBIN",
        exchange="NSE",
        interval="5m",
        qty=10,
    )
    assert_same_result(run(arrays, DATA), run(ShortCross, DATA))


def test_from_signals_refuses_a_function_that_reads_later_bars():
    def tomorrow(frame):
        later = frame.close.shift(-1)
        return (later > frame.close).to_numpy(), (later < frame.close).to_numpy()

    with pytest.raises(ConfigError, match="looks ahead"):
        run(bt.from_signals(tomorrow, "SBIN", "NSE", "5m", qty=10), DATA)


def test_from_signals_needs_two_or_four_outputs():
    with pytest.raises(ConfigError, match=r"\(entries, exits\)"):
        run(bt.from_signals(lambda frame: frame.close > 0, "SBIN", "NSE", "5m"), DATA)


def test_from_arrays_refuses_an_argument_a_condition_entry_does_not_take():
    with pytest.raises(ConfigError, match="price"):
        bt.from_arrays([True], [False], symbol="SBIN", exchange="NSE", interval="5m", price=1.0)


def test_from_arrays_needs_entries_or_shorts():
    nothing = bt.from_arrays(None, [True], symbol="SBIN", exchange="NSE", interval="5m")
    with pytest.raises(ConfigError, match="entries or shorts"):
        run(nothing, DATA)


def test_an_array_of_the_wrong_length_names_the_bar_count():
    short_array = bt.from_arrays(
        np.ones(3, dtype=bool), None, symbol="SBIN", exchange="NSE", interval="5m"
    )
    with pytest.raises(ConfigError, match="3 values and the run has 150 strategy bars"):
        run(short_array, DATA)
