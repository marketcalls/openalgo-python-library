"""What an indicator needs before it is ready (decision D25, `_warm_up_measure`): an
indicator of other views needs theirs as well, a condition is compared where its events
are, a very long average needs more than the longest window and says so, and an
indicator the reference bars cannot compute says why.

The runs use the seeded daily walk of `warm_up_bars` from 2000 (drift 0.03 percent a
session, moves of 1.5 percent), one session at a time from 30 starts spread over the
last two decades, so each first decision is made on history the run loaded. Expected
first values are those of the same `ta` functions over all the bars given; expected
needs are worked out from the definitions in the tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from warm_up_bars import SETTINGS, walk

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._causality_checks import IndicatorChecks
from openalgo.backtest._warm_up_measure import FAILED, NEVER_SETTLES, ReferenceCall, ReferenceNeeds

DATA = walk("2000-01-01", "2026-06-30", seed=21, drift=0.0003, vol=0.015)
CLOSES = DATA.close.to_numpy()
STARTS = DATA.index[np.linspace(3000, len(DATA) - 5, 30).astype(int)]


def before(values: np.ndarray) -> np.ndarray:
    """`values` one bar later: NaN on the first bar."""
    return np.concatenate(([np.nan], values[:-1]))


class Reading(bt.Strategy):
    """Records the first decision of the view `build` makes: (bar open, its value)."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    build = None
    seen: list = []

    def init(self):
        self.view = type(self).build(self)

    def on_bar(self, bar):
        if not self.seen:
            self.seen.append((bar.time, self.view[-1]))


def first_values(build) -> tuple[list[tuple[int, object]], int]:
    """((row of the first decision, the view's value there), one per start; the need
    the runs recorded)."""
    values, needs = [], set()
    for start in STARTS:
        strategy = type("Reading", (Reading,), {"build": staticmethod(build), "seen": []})
        result = bt.backtest(strategy, data=DATA, start=start.date(), end=start.date(), **SETTINGS)
        ((at, value),) = strategy.seen
        values.append((DATA.index.get_loc(pd.Timestamp(at)), value))
        needs.update(need["bars"] for need in result.manifest["arguments"]["warm_up"]["needs"])
    (need,) = needs
    return values, need


def breakout(s):
    return s.I(ta.crossover, s.close, s.close.rolling_max(100).shift(1))


def test_a_breakout_over_a_shifted_rolling_high_needs_the_window_the_shift_and_the_bar_before():
    # A 100-bar high is known from its 100th bar, a bar later once shifted, and a crossing
    # reads the bar before as well: 102 bars at least.
    values, need = first_values(breakout)
    assert need >= 102
    expected = ta.crossover(CLOSES, before(np.asarray(ta.highest(CLOSES, 100))))
    assert [bool(value) for _, value in values] == [bool(expected[row]) for row, _ in values]


def test_a_state_condition_is_ready_where_its_whole_history_agrees():
    def flip(s):
        up = s.close > s.close.rolling_max(20).shift(1)
        down = s.close < s.close.rolling_min(20).shift(1)
        return s.I(ta.flip, up, down)

    values, need = first_values(flip)
    up = before(np.asarray(ta.highest(CLOSES, 20))) < CLOSES
    down = before(np.asarray(ta.lowest(CLOSES, 20))) > CLOSES
    expected = ta.flip(up, down)
    assert need >= 22
    assert [bool(value) for _, value in values] == [bool(expected[row]) for row, _ in values]


def test_an_indicator_of_an_indicator_needs_at_least_its_inputs_warm_up():
    def smoothed(s):
        return s.I(ta.sma, s.I(ta.ema, s.close, 50), 10)

    def alone(s):
        return s.I(ta.ema, s.close, 50)

    start = STARTS[-1].date()
    inner = bt.backtest(single(alone), data=DATA, start=start, **SETTINGS)
    outer = bt.backtest(single(smoothed), data=DATA, start=start, **SETTINGS)
    need = outer.manifest["arguments"]["warm_up"]["needs"]
    inner_need = inner.manifest["arguments"]["warm_up"]["needs"][0]["bars"]
    assert max(entry["bars"] for entry in need) >= inner_need


def single(build) -> type[bt.Strategy]:
    class Single(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

        def init(self):
            self.view = build(self)

    return Single


# ----- needs the reference bars measure ------------------------------------------------


def on_closes(function, *arguments):
    """The outputs of `function` on the reference bars' closes."""

    def outputs(bars):
        values = function(pd.Series(bars["close"], index=bars["times"]), *arguments)
        return [np.asarray(values, dtype=float)]

    return ReferenceCall(outputs)


def test_an_average_longer_than_the_longest_window_needs_more_than_that_window():
    # An EMA of 10,000 bars keeps (1 - 2/10,001) of its seed's gap a bar: more than the
    # 32,768-bar longest window leaves, and less at each doubling, unlike a running sum.
    found = ReferenceNeeds().need("ema 10000", on_closes(ta.ema, 10_000))
    assert (found.bars, found.more) == (32_768, True)


def test_an_average_the_longest_window_holds_is_measured():
    # An EMA of 2,000 bars needs about 8.5 x 2,000 = 17,000 bars: the 32,768-bar window.
    found = ReferenceNeeds().need("ema 2000", on_closes(ta.ema, 2_000))
    assert (found.bars, found.more) in ((16_384, False), (32_768, False))


def test_a_running_sum_is_still_told_apart_from_a_long_average():
    def balance(bars):
        close, volume = (pd.Series(bars[name], index=bars["times"]) for name in ("close", "volume"))
        return [np.asarray(ta.obv(close, volume), dtype=float)]

    found = ReferenceNeeds().need("obv", ReferenceCall(balance))
    assert (found.bars, found.note) == (None, NEVER_SETTLES)


def test_an_indicator_the_reference_bars_cannot_compute_says_why():
    def refuses(bars):
        raise ValueError("Period (70000) cannot be greater than data length (65536)")

    found = ReferenceNeeds().need("sma 70000", ReferenceCall(refuses))
    assert found.bars is None
    assert found.note == FAILED.format(
        error="Period (70000) cannot be greater than data length (65536)"
    )


def test_the_failure_is_the_note_of_the_indicators_check():
    def refuses(bars):
        raise ValueError("no")

    checks = IndicatorChecks("sha256:strategy").for_run()
    whole = [CLOSES[:200]]
    checks.check("k", "sma(SBIN.close, 70000)", lambda a, b: [CLOSES[a:b]], whole, None,
                 ("SBIN", "D"), ReferenceCall(refuses))  # fmt: skip
    assert checks.assumption().endswith(
        "sma(SBIN.close, 70000): truncation passed at 4 cuts, could not be computed on the "
        "engine's reference bars (no), so its warm-up cannot be measured"
    )


@pytest.mark.parametrize("length", [20, 50, 100])
def test_a_condition_on_a_rolling_high_is_measured_from_its_events(length):
    # close > its own rolling high of `length` bars shifted by one: true on a few bars
    # only, each needing the `length` bars before it, so a shorter window shows only there.
    def outputs(bars):
        close = np.asarray(bars["close"])
        return [close > before(np.asarray(ta.highest(close, length)))]

    found = ReferenceNeeds().need(f"breakout {length}", ReferenceCall(outputs))
    assert found.bars >= length + 1
