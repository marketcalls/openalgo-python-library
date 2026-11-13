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
from warm_up_bars import IST, SETTINGS, intraday

import openalgo.backtest as bt
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
