"""`self.I`: indicators from openalgo.ta, revealed as their bars close (14-strategy-api
2.2), with look-ahead outputs rejected (03-architecture 5).

The run is `strategy_samples.minute_bars` on a 5-minute strategy: the 5-minute bar j
closes at 500.45 + 0.50 j (the close of minute 5 j + 4), so a 3-bar average of closes
is 500.95 at bar 2 and 501.45 at bar 3.
"""

import math

import numpy as np
import pytest
from strategy_samples import minute_bars, run

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.errors import ConfigError


class Recorder(bt.Strategy):
    """Calls `probe(self)` at each of the first four strategy bars and keeps the result."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    product = "CNC"

    def init(self):
        self.average = self.I(ta.sma, self.close, 3)
        self.seen = []

    def probe(self):
        return None

    def on_bar(self, bar):
        if len(self.seen) < 4:
            self.seen.append(self.probe())


def recorded(probe, strategy=Recorder, **settings):
    seen = []

    class Probing(strategy):
        def on_stop(self):
            seen.extend(self.seen)

    Probing.probe = probe
    run(Probing, **settings)
    return seen


def test_an_indicator_shows_its_value_at_the_latest_closed_bar():
    seen = recorded(lambda s: (len(s.average), s.average[-1]))
    assert [length for length, _ in seen] == [1, 2, 3, 4]
    assert math.isnan(seen[0][1]) and math.isnan(seen[1][1])
    assert seen[2][1] == pytest.approx(500.95)
    assert seen[3][1] == pytest.approx(501.45)


def test_nothing_of_an_indicator_is_visible_in_init():
    found = []

    class InInit(Recorder):
        def init(self):
            super().init()
            found.append(len(self.average))
            with pytest.raises(IndexError, match="0 bars have closed"):
                self.average[-1]

    run(InInit)
    assert found == [0]


def test_a_slice_returns_the_revealed_values():
    seen = recorded(lambda s: s.average[-2:])
    np.testing.assert_allclose(seen[3], [500.95, 501.45])


def test_arithmetic_with_a_series_builds_a_view():
    # Bar 2: close 501.45 minus the average 500.95.
    seen = recorded(lambda s: ((s.close - s.average)[-1], (2 * s.average)[-1]))
    assert seen[2][0] == pytest.approx(0.5)
    assert seen[2][1] == pytest.approx(1001.9)


def test_a_comparison_is_a_view_of_booleans_true_at_its_latest_value():
    seen = recorded(lambda s: ((s.average < s.close)[-1], bool(s.average < s.close)))
    assert seen[2] == (True, True)


def test_a_view_of_numbers_has_no_truth_value():
    class Truthy(Recorder):
        def on_bar(self, bar):
            bool(self.close)

    with pytest.raises(TypeError, match="SBIN.close is a series of numbers"):
        run(Truthy)


def test_crossed_above_is_true_on_the_bar_of_the_cross():
    # Closes 500.45, 500.95, 501.45, 501.95: above 501 from bar 2.
    seen = recorded(lambda s: bool(s.close.crossed_above(501.0)))
    assert seen == [False, False, True, False]


def test_crossed_below_is_true_on_the_bar_of_the_cross():
    seen = recorded(lambda s: bool((-s.close).crossed_below(-501.0)))
    assert seen == [False, False, True, False]


def test_a_function_outside_openalgo_ta_is_refused():
    class Custom(Recorder):
        def init(self):
            self.I(np.mean, self.close)

    with pytest.raises(ConfigError, match="openalgo.ta"):
        run(Custom)


def test_a_non_causal_indicator_is_rejected_by_name():
    class Fractals(Recorder):
        def init(self):
            self.I(ta.fractals, self.high, self.low)

    with pytest.raises(ConfigError, match=r"ta\.fractals .*fractal_up"):
        run(Fractals)


class Ichimoku(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    product = "CNC"

    def init(self):
        self.lines = self.I(ta.ichimoku, self.high, self.low, self.close)


def test_a_look_ahead_output_is_rejected_when_used():
    class UsesLaggingSpan(Ichimoku):
        def on_bar(self, bar):
            self.lines[4][-1]

    with pytest.raises(ConfigError, match="'lagging_span' looks ahead"):
        run(UsesLaggingSpan)


def test_the_causal_outputs_of_the_same_function_work():
    found = []

    class UsesConversionLine(Ichimoku):
        def on_stop(self):
            found.append((len(self.lines), len(self.lines[0])))

    run(UsesConversionLine)
    assert found == [(5, 375)]


def test_a_function_with_several_outputs_returns_one_view_each():
    found = []

    class Macd(Recorder):
        def init(self):
            super().init()
            self.parts = self.I(ta.macd, self.close)

        def on_stop(self):
            found.append([view.label for view in self.parts])

    run(Macd)
    assert found == [
        ["macd(SBIN.close).macd", "macd(SBIN.close).signal", "macd(SBIN.close).histogram"]
    ]


def test_an_indicator_over_two_symbols_is_not_built_yet():
    class Pair(bt.Strategy):
        symbols, exchange, interval = ("SBIN", "ICICIBANK"), "NSE", "5m"
        product = "CNC"

        def init(self):
            self.I(ta.correlation, self.data["SBIN"].close, self.data["ICICIBANK"].close, 10)

    frames = {"SBIN": minute_bars(), "ICICIBANK": minute_bars()}
    with pytest.raises(NotImplementedError, match="several symbols or intervals"):
        run(Pair, data=frames)


def test_series_of_two_intervals_cannot_be_combined():
    class Mixed(Recorder):
        def init(self):
            self.close - self.data["SBIN", "15m"].close

    with pytest.raises(ConfigError, match="different series"):
        run(Mixed)


def test_the_assumptions_state_the_tolerance_and_each_check():
    # 75 strategy bars: cuts at bars 44, 52, 59 and 67; a 3-bar average is exact on any
    # window of 16 bars.
    checks = run(Recorder).assumptions["indicator_checks"]
    assert checks == (
        "window tolerance 1e-06 (relative); sma(SBIN.close, 3): truncation passed at 4 cuts, "
        "warm-up 16 bars"
    )


def test_an_indicator_on_another_interval_follows_that_interval():
    seen = []

    class Quarterly(Recorder):
        def init(self):
            super().init()
            self.quarter = self.I(ta.sma, self.data["SBIN", "15m"].close, 1)

        def probe(self):
            return len(self.quarter)

        def on_stop(self):
            seen.extend(self.seen)

    run(Quarterly)
    # 5-minute closes at 09:20, 09:25, 09:30, 09:35: the first 15-minute bar closes at 09:30.
    assert seen == [0, 0, 1, 1]
