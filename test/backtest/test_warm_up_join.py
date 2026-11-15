"""The history before `start` is in the strategy's series before `init` runs (decision
D25, `_warm_up_history`, `_warm_up_join`): an indicator longer than the run is computed
over it, a view of another interval holds whole bars, a strategy bar cut by a datetime
`start` holds the bars before it, a split between the history and the run is caught, a
declared one is adjusted for, and the check reads no bar after the first decision.

The bars are the seeded walks of `warm_up_bars`. Expected values are the full-history
ones: the indicator computed by `openalgo.ta` over all the bars given, or the bars of a
day or an hour aggregated here by hand (first open, highest high, total volume).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from warm_up_bars import IST, SETTINGS, intraday, walk

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.data import CorporateAction

# A 200-bar EMA's warm-up is within the window tolerance of the full-history value.
WINDOW_TOLERANCE = 1e-6
DAILY = walk("2015-01-01", "2026-06-30")


def first_decision(strategy: type, **arguments) -> tuple:
    """(bar, the strategy's record) at the strategy's first `on_bar`."""
    seen = []

    class Recording(strategy):
        def on_bar(self, bar):
            if not seen:
                seen.append((bar, self.record()))

    bt.backtest(Recording, **arguments, **SETTINGS)
    return seen[0]


class SlowAverage(bt.Strategy):
    """Reads a 200-bar EMA of the close."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        self.line = self.I(ta.ema, self.close, 200)

    def record(self):
        return len(self.close), self.line[-1]


@pytest.mark.parametrize("start", ["2026-01-01", "2026-04-01"])
def test_a_run_shorter_than_its_indicators_period_is_computed_over_the_history(start):
    # 120 and 60 sessions, both fewer than 200: the 200-bar EMA is computed over the
    # history its warm-up asks for, and its first value is the full-history one.
    bar, (closed, value) = first_decision(SlowAverage, data=DAILY, start=start, end="2026-06-30")
    row = DAILY.index.get_loc(pd.Timestamp(bar.time))
    expected = float(ta.ema(DAILY.close.to_numpy(), 200)[row])
    assert value == pytest.approx(expected, rel=WINDOW_TOLERANCE)
    assert closed > 200


def test_a_from_signals_function_with_a_long_average_runs_on_a_short_run():
    seen = []

    def signals(frame):
        seen.append(frame.index[0])
        fast, slow = ta.ema(frame.close, 20), ta.ema(frame.close, 200)
        return ta.crossover(fast, slow), ta.crossunder(fast, slow)

    strategy = bt.from_signals(signals, "SBIN", "NSE", "D", qty=1, product="CNC")
    bt.backtest(strategy, data=DAILY, start="2026-04-01", end="2026-06-30", **SETTINGS)
    assert seen[0].date() < dt.date(2026, 4, 1)


# ----- a datetime start: the morning before it is history ------------------------------

FIVE_MINUTES = intraday("2026-03-02", "2026-03-13", minutes=5)
ELEVEN = dt.datetime(2026, 3, 10, 11, 0, tzinfo=IST)


def day_of(bars: pd.DataFrame, day: dt.date) -> tuple[float, float, float]:
    """(first open, highest high, total volume) of the bars of `day`."""
    rows = bars[bars.index.date == day]
    return rows.open.iloc[0], rows.high.max(), rows.volume.sum()


@pytest.mark.parametrize("read_in_init", [False, True])
def test_a_daily_view_holds_the_whole_start_day_after_a_datetime_start(read_in_init):
    seen = []

    class Daily(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "5m", "MIS"

        def init(self):
            self.line = self.I(ta.sma, self.close, 3)
            if read_in_init:
                self.days = self.data["SBIN", "D"]

        def on_bar(self, bar):
            if bar.time.date() == dt.date(2026, 3, 11) and not seen:
                days = self.data["SBIN", "D"]
                seen.append((days.open[-1], days.high[-1], days.volume[-1]))

    bt.backtest(Daily, data=FIVE_MINUTES, start=ELEVEN, end="2026-03-13", base="5m", **SETTINGS)
    assert seen == [day_of(FIVE_MINUTES, dt.date(2026, 3, 10))]


def test_a_strategy_bar_cut_by_a_datetime_start_holds_the_base_bars_before_it():
    # Hourly bars from 09:15; the run starts at 10:00, inside the 09:15 to 10:15 bar,
    # which is its first: built from its 12 five-minute bars, nine of them before 10:00.
    seen = []

    class Hourly(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1h", "MIS"

        def on_bar(self, bar):
            if not seen:
                seen.append((bar.time, bar.open, bar.high, bar.volume, self.open[-1]))

    ten = dt.datetime(2026, 3, 10, 10, 0, tzinfo=IST)
    bt.backtest(Hourly, data=FIVE_MINUTES, start=ten, end="2026-03-13", base="5m", **SETTINGS)
    hour = FIVE_MINUTES[(FIVE_MINUTES.index >= ten.replace(hour=9, minute=15))]
    hour = hour[hour.index < ten.replace(minute=15)]
    assert len(hour) == 12
    opened = hour.open.iloc[0]
    assert seen == [
        (ten.replace(hour=9, minute=15), opened, hour.high.max(), hour.volume.sum(), opened)
    ]


# ----- a split between the history and the run ------------------------------------------

SPLIT_DAY = dt.date(2025, 1, 1)


def split_at(day: dt.date) -> pd.DataFrame:
    """Daily bars whose prices halve from `day` on, unadjusted (a 1:2 split)."""
    data = walk("2018-01-01", "2026-06-30", seed=1)
    after = data.index.date >= day
    data.loc[after, ["open", "high", "low", "close"]] *= 0.5
    return data


class Average(bt.Strategy):
    """Reads a 20-bar EMA."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        self.line = self.I(ta.ema, self.close, 20)

    def record(self):
        return self.history(None, len(self.close)).index[0]


def test_a_split_between_the_history_and_the_run_leaves_the_run_without_history():
    data = split_at(SPLIT_DAY)
    bar, first_bar_seen = first_decision(Average, data=data, start=SPLIT_DAY)
    # No bar before the split is seen: the strategy waited inside the run.
    assert first_bar_seen.date() == SPLIT_DAY
    assert bar.time.date() > SPLIT_DAY


def test_a_split_between_the_history_and_the_run_is_named_in_the_results():
    result = bt.backtest(Average, data=split_at(SPLIT_DAY), start=SPLIT_DAY, **SETTINGS)
    text = result.assumptions["warm_up"]
    assert (
        f"history before start: SBIN none (the bars before {SPLIT_DAY} could not be used: 1 "
        f"suspected unadjusted split or bonus ({SPLIT_DAY}: the previous close"
    ) in text
    (warning,) = [note for note in result.warnings if "waited" in note]
    assert f"its bars before {SPLIT_DAY} could not be used (1 suspected unadjusted split" in warning


def test_a_split_inside_the_history_but_before_what_the_run_keeps_costs_it_nothing():
    # The split is two years before start; the 20-bar EMA keeps much less than that.
    data = split_at(dt.date(2023, 1, 2))
    result = bt.backtest(Average, data=data, start=SPLIT_DAY, **SETTINGS)
    assert result.assumptions["warm_up"].endswith("ready at start")
    assert np.isfinite(result.metrics["net_profit"])


# ----- a declared split inside the history: the history is adjusted for it --------------

PRICES = ["open", "high", "low", "close"]
ADJUSTED = walk("2018-01-01", "2026-06-30", seed=1)
# Ten sessions before start, the bars before EX_DATE are left at twice their adjusted
# prices with half their volume: a 1:2 split (face value 10 to 5) the data are not
# adjusted for.
EX_DATE = ADJUSTED.index[ADJUSTED.index.searchsorted(pd.Timestamp(SPLIT_DAY, tz=IST)) - 10]
UNADJUSTED = ADJUSTED.copy()
UNADJUSTED.loc[UNADJUSTED.index < EX_DATE, PRICES] *= 2.0
UNADJUSTED.loc[UNADJUSTED.index < EX_DATE, "volume"] /= 2.0
DECLARED = [CorporateAction.split(EX_DATE.date(), 10, 5)]


class Ema50(bt.Strategy):
    """Reads a 50-bar EMA and the bars it is computed over."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        self.line = self.I(ta.ema, self.close, 50)

    def record(self):
        return self.line[-1], self.history(None, len(self.close))


def test_a_declared_split_inside_the_history_gives_the_adjusted_series_indicators():
    bar, (value, _) = first_decision(
        Ema50, data=UNADJUSTED, start=SPLIT_DAY, corporate_actions=DECLARED
    )
    row = ADJUSTED.index.get_loc(pd.Timestamp(bar.time))
    expected = float(ta.ema(ADJUSTED.close.to_numpy(), 50)[row])
    assert bar.time.date() == SPLIT_DAY
    assert value == pytest.approx(expected, rel=WINDOW_TOLERANCE)


def test_the_history_before_a_declared_ex_date_is_back_adjusted_by_its_factor():
    # Prices before the ex-date divided by 2 and volume multiplied by 2: the adjusted bars.
    _, (_, seen) = first_decision(
        Ema50, data=UNADJUSTED, start=SPLIT_DAY, corporate_actions=DECLARED
    )
    history = seen[seen.index < pd.Timestamp(SPLIT_DAY, tz=IST)]
    # By day: a special session's bar is stamped at that session's open, not at 09:15.
    expected = ADJUSTED[np.isin(ADJUSTED.index.date, history.index.date)]
    assert len(history) == len(expected) > 50
    assert np.array_equal(history[PRICES].to_numpy(), expected[PRICES].to_numpy())
    assert np.array_equal(history["volume"].to_numpy(), expected["volume"].to_numpy())


def test_the_assumptions_say_the_history_was_adjusted_for_the_declared_split():
    run = {"data": UNADJUSTED, "start": SPLIT_DAY, "corporate_actions": DECLARED}
    result = bt.backtest(Ema50, **run, **SETTINGS)
    assert (
        f"adjusted for a split (2 shares for every 1) on {EX_DATE.date()}: prices before it "
        "divided by 2, volume multiplied by 2"
    ) in result.assumptions["warm_up"]
    assert result.assumptions["warm_up"].endswith("ready at start")


# ----- the join is checked with the bars up to the first decision only ------------------

JOIN_DAY = dt.date(2026, 3, 10)
QUARTER_HOURS = intraday("2026-02-02", "2026-03-13", minutes=15)
RUN_END = {"end": "2026-03-13", **SETTINGS}


def fallen_at_open(recovers_at_noon: bool) -> pd.DataFrame:
    """QUARTER_HOURS with prices halved from the open of JOIN_DAY on, and, when
    `recovers_at_noon`, restored from 12:00 that day: the two are the same bars up to
    12:00 on JOIN_DAY."""
    bars = QUARTER_HOURS.copy()
    bars.loc[bars.index.date >= JOIN_DAY, PRICES] *= 0.5
    if recovers_at_noon:
        later = bars.index >= pd.Timestamp(dt.datetime.combine(JOIN_DAY, dt.time(12, 0)), tz=IST)
        bars.loc[later, PRICES] *= 2.0
    return bars


class QuarterHourAverage(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"

    def init(self):
        self.line = self.I(ta.ema, self.close, 10)


def test_whether_the_history_is_kept_does_not_depend_on_bars_after_the_first_decision():
    holds, recovers = (
        bt.backtest(QuarterHourAverage, data=fallen_at_open(recovered), start=JOIN_DAY, **RUN_END)
        for recovered in (False, True)
    )
    assert holds.assumptions["warm_up"] == recovers.assumptions["warm_up"]
    assert holds.manifest["arguments"]["warm_up"] == recovers.manifest["arguments"]["warm_up"]
    assert "could not be used: 1 suspected unadjusted split" in holds.assumptions["warm_up"]
