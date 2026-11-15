"""How far back the history before `start` is asked for, and what a sweep keeps of it
(decision D25, `_warm_up_load`, `_warm_up_days`): sparse bars are asked for further back
until the need is met, a failed load of a larger combination's history does not change
a smaller one's, and a need `init` did not register on the reference bars says so.

The bars are the seeded walks of `warm_up_bars`. The warm-ups used are those of the
engine's reference bars: 256 bars for a 20-bar EMA, 2,048 for a 200-bar one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from warm_up_bars import IST, SETTINGS, intraday, walk

import openalgo.backtest as bt
import openalgo.backtest._warm_up_load as wl
from openalgo import ta

# The share of minutes with a trade in a thinly traded share's one-minute bars.
TRADED_SHARE = 0.35


def warm_up(result: bt.Result) -> dict:
    return result.manifest["arguments"]["warm_up"]


# ----- sparse bars: asked for further back until the need is met ------------------------


class SlowMinutes(bt.Strategy):
    """A 200-bar EMA of one-minute closes: 2,048 bars."""

    symbol, exchange, interval, product = "SMALLCAP", "NSE", "1m", "MIS"

    def init(self):
        self.line = self.I(ta.ema, self.close, 200)


def thinly_traded(first: str) -> pd.DataFrame:
    """One-minute bars from `first` to 2026-03-31 with a trade in about one minute of
    three, at minutes drawn by a seeded generator: about a third of the bars a full
    session holds, at no regular spacing."""
    bars = intraday(first, "2026-03-31", minutes=1)
    return bars[np.random.default_rng(1).random(len(bars)) < TRADED_SHARE]


def test_sparse_bars_are_asked_for_further_back_until_the_need_is_met():
    data = thinly_traded("2025-10-01")
    result = bt.backtest(SlowMinutes, data=data, start="2026-03-02", **SETTINGS)
    assert result.assumptions["warm_up"].endswith("ready at start")
    assert warm_up(result)["first_decision"] is None


def test_sparse_bars_that_begin_too_late_say_where_they_begin():
    data = thinly_traded("2026-02-23")
    result = bt.backtest(SlowMinutes, data=data, start="2026-03-02", **SETTINGS)
    (warning,) = [note for note in result.warnings if "waited" in note or "no decision" in note]
    assert warning.endswith(
        "the bars given for SMALLCAP begin on 2026-02-23; give earlier ones to decide from "
        "2026-03-02."
    )


# ----- a need init did not register on the reference bars --------------------------------


class Branching(bt.Strategy):
    """Registers a daily EMA only from its second init, the run's: the reference bars
    never see it."""

    symbol, exchange, interval, product = "SBIN", "NSE", "5m", "MIS"
    runs = 0

    def init(self):
        type(self).runs += 1
        self.line = self.I(ta.ema, self.close, 20)
        if type(self).runs > 1:
            self.daily = self.I(ta.ema, self.data["SBIN", "D"].close, 20)


def test_a_need_init_did_not_register_on_the_reference_bars_says_so():
    data = intraday("2025-06-02", "2026-03-31", minutes=5)
    result = bt.backtest(Branching, data=data, start="2026-03-02", **SETTINGS)
    (warning,) = [note for note in result.warnings if "waited" in note or "no decision" in note]
    assert "ema(SBIN[D].close, 20) needs 256 D bars of SBIN" in warning
    assert warning.endswith(
        "no history was loaded for it: init did not register it when it ran on the "
        "engine's reference bars to learn what history to load."
    )


# ----- a failed load of a larger combination's history ----------------------------------


class Crossing(bt.Strategy):
    """Long while the 10-bar EMA of five-minute closes is above the slower one."""

    symbol, exchange, interval, product = "SBIN", "NSE", "5m", "MIS"
    fast, slow = 10, 20

    def init(self):
        self.fast_line = self.I(ta.ema, self.close, self.fast)
        self.slow_line = self.I(ta.ema, self.close, self.slow)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast_line[-1] > self.slow_line[-1]:
            self.buy(qty=1)
        elif self.position.is_long and self.fast_line[-1] < self.slow_line[-1]:
            self.exit()


def off_lattice(day: dt.date) -> pd.DataFrame:
    """Five-minute bars with the eleventh bar of `day` moved two minutes off the
    five-minute lattice: a load that reaches `day` fails."""
    bars = intraday("2025-11-03", "2026-03-31", minutes=5, seed=4)
    stamps = bars.index.tolist()
    moved = stamps.index(dt.datetime.combine(day, dt.time(10, 5), tzinfo=IST))
    stamps[moved] += dt.timedelta(minutes=2)
    return bars.set_axis(pd.DatetimeIndex(stamps, name="timestamp"))


# About 20 sessions before start: inside the 2,048 bars slow=200 asks for, outside the
# 256 bars slow=20 asks for.
BROKEN = off_lattice(dt.date(2026, 2, 3))
RUN = {"data": BROKEN, "start": "2026-03-02", "end": "2026-03-06", **SETTINGS}


def same_run(one: bt.Result, other: bt.Result) -> bool:
    return (
        one.assumptions["warm_up"] == other.assumptions["warm_up"]
        and one.manifest["data"] == other.manifest["data"]
        and one.trades.equals(other.trades)
        and one.metrics["net_profit"] == other.metrics["net_profit"]
    )


def test_a_failed_load_of_a_larger_combinations_history_leaves_a_smaller_one_as_alone():
    alone = bt.backtest(Crossing, params={"slow": 20}, **RUN)
    swept = bt.optimize(
        Crossing, workers=1, progress=lambda line: None, slow=(20, 200), **RUN
    )  # fmt: skip
    chosen = swept.select(slow=20)
    row = swept.table[swept.table.slow == 20].iloc[0]
    assert alone.assumptions["warm_up"].endswith("ready at start")
    assert same_run(chosen, alone)
    assert same_run(chosen.rerun(), alone)
    assert (row.trades, row.objective) == (len(alone.trades), alone.metrics["net_profit"])


def test_the_larger_combination_whose_load_failed_is_its_single_run():
    alone = bt.backtest(Crossing, params={"slow": 200}, **RUN)
    swept = bt.optimize(
        Crossing, workers=1, progress=lambda line: None, slow=(20, 200), **RUN
    )  # fmt: skip
    assert "could not be loaded: TimestampError" in alone.assumptions["warm_up"]
    assert same_run(swept.select(slow=200), alone)


# ----- stepping back over a thinly traded share -------------------------------------------


class SlowFiveMinutes(bt.Strategy):
    """A 300-bar EMA of five-minute closes: 4,096 bars, about 55 full sessions, so about
    180 sessions of a share with a trade in three five-minute bars of ten."""

    symbol, exchange, interval, product = "BBB", "NSE", "5m", "MIS"

    def init(self):
        self.line = self.I(ta.ema, self.close, 300)


THIN = intraday("2025-03-03", "2026-03-31", minutes=5, seed=5)
THIN = THIN[np.random.default_rng(3).random(len(THIN)) < 0.3]
THIN_RUN = {"start": "2026-03-02", "end": "2026-03-06", **SETTINGS}


def moved_off_the_lattice(bars: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """`bars` with the fourth bar of `day` moved two minutes off the five-minute lattice."""
    stamps = bars.index.tolist()
    moved = [k for k, stamp in enumerate(stamps) if stamp.date() == day][3]
    stamps[moved] += dt.timedelta(minutes=2)
    return bars.set_axis(pd.DatetimeIndex(stamps, name="timestamp"))


def test_a_step_that_fails_keeps_the_steps_before_it_and_says_why():
    # The steps back from 2026-03-02 load fine down to the one that reaches 2025-07-15,
    # where a bar is off the lattice: the bars of the steps before it are kept.
    data = moved_off_the_lattice(THIN, dt.date(2025, 7, 15))
    result = bt.backtest(SlowFiveMinutes, data=data, **THIN_RUN)
    history = result.assumptions["warm_up"].split("history before start: ")[1]
    assert history.startswith("BBB ") and not history.startswith("BBB none")
    (stopped,) = [note for note in result.warnings if note.startswith("BBB history before")]
    assert "failed (TimestampError" in stopped and "2025-07-15" in stopped
    (wait,) = [note for note in result.warnings if "waited" in note or "no decision" in note]
    assert "failed (TimestampError" in wait


def test_a_gap_at_a_step_day_does_not_stop_the_steps_back():
    # A three-week suspension covers 2025-10-29, the first day of a step: that step
    # still returns older bars (from before the gap), so stepping goes on to bars that
    # meet the need.
    gap = THIN[
        (THIN.index.date >= dt.date(2025, 10, 22)) & (THIN.index.date <= dt.date(2025, 11, 12))
    ]
    result = bt.backtest(SlowFiveMinutes, data=THIN.drop(gap.index), **THIN_RUN)
    assert result.assumptions["warm_up"].endswith("ready at start")


def daily_holes(first: str, last: str, share: float) -> pd.DataFrame:
    """Daily bars from `first` to `last` with a bar on about `share` of the sessions."""
    bars = walk(first, last, seed=6)
    return bars[np.random.default_rng(4).random(len(bars)) < share]


class LongDaily(bt.Strategy):
    """A 200-bar EMA of daily closes: 2,048 bars."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        self.line = self.I(ta.ema, self.close, 200)


def test_each_step_back_asks_only_for_the_days_before_the_last_one(monkeypatch):
    # Bars on two sessions in five since 2008 never hold 2,048 bars, so every step of the
    # ladder that returns older bars is taken: each asks for the days before the one
    # the step before it began on, and none is asked for twice.
    asked = []
    loading = wl.load_bars

    def recording(symbol, exchange, interval, start, end, **options):
        asked.append((start, end))
        return loading(symbol, exchange, interval, start, end, **options)

    monkeypatch.setattr(wl, "load_bars", recording)
    bt.backtest(LongDaily, data=daily_holes("2008-01-01", "2026-03-31", 0.4), start="2026-01-01",
                end="2026-03-31", **SETTINGS)  # fmt: skip
    assert len(asked) > 3
    for (first, _), (_, before) in zip(asked, asked[1:], strict=False):
        assert before == first - dt.timedelta(days=1)
    assert asked[0][1] == dt.date(2025, 12, 31)


class Average(bt.Strategy):
    """A 20-bar simple average of daily closes: 32 bars."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        self.line = self.I(ta.sma, self.close, 20)


def test_own_data_whose_history_stops_at_a_gap_says_where_its_bars_begin():
    # No bar from 2024-10-01 to 2025-11-30. The first step, 63 days before 2026-01-01,
    # holds the 22 sessions of December; the next, 79 days before, adds none, and its
    # first day is more than a week before the bars held: the history stops at the gap.
    # The bars given begin in 2020, and the message says so.
    bars = walk("2020-01-01", "2026-03-31", seed=6)
    hole = (bars.index.date >= dt.date(2024, 10, 1)) & (bars.index.date < dt.date(2025, 12, 1))
    data = bars[~hole]
    result = bt.backtest(Average, data=data, start="2026-01-01", **SETTINGS)
    (wait,) = [note for note in result.warnings if "waited" in note]
    asked = dt.date(2026, 1, 1) - dt.timedelta(days=79)
    assert wait.endswith(
        f"the bars given for SBIN begin on {data.index[0].date()} but hold none from {asked} "
        "to 2025-12-01, where the history stopped; give bars without the gap to decide from "
        "2026-01-01."
    )
