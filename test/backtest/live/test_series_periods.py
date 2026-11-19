"""Period stores of a live strategy series equal a backtest's (16-session-design unit 6):
a week or a month that today continues is still forming, so neither the history's part
of it nor today's is shown as a bar until the period closes, as in a backtest, which
builds the period from all its bars at once.

The strategy decides on 15-minute bars and reads its daily, weekly and monthly bars. Its
80-bar SMA makes the backtest keep six sessions before the live day; the live run gets
them as its history. The live days: Thursday 2025-06-26, in the middle of a week and of
a month, and Tuesday 2025-07-01, the day after a week began and the first of a month.
The bars are those of `series_samples`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from series_samples import LiveRun, backtest_views, before, live_bar, minute_bars, on_day

import openalgo.backtest as bt
from openalgo import ta

BARS = minute_bars()
PERIODS = ("D", "W", "M")
MID_WEEK_AND_MONTH = dt.date(2025, 6, 26)
FIRST_OF_A_MONTH = dt.date(2025, 7, 1)


class Periods(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def init(self):
        self.trend = self.I(ta.sma, self.close, 80)


def reading(strategy: bt.Strategy) -> dict:
    """The closes and the close instants of the daily, weekly and monthly bars a
    decision of `Periods` sees, and the open of its first strategy bar."""
    seen = {}
    for interval in PERIODS:
        store = strategy._ctx.store("SBIN", interval)
        shown = store.visible()
        seen[interval] = store.column("ts_close")[:shown].copy()
        seen[f"{interval} close"] = strategy.data["SBIN", interval].close.values.copy()
    seen["first open"] = int(strategy._ctx.stores["SBIN"].column("ts_open")[0])
    return seen


def live_views(day: dt.date, first_open: int) -> list[tuple]:
    """(decision instant, reading) at every `on_bar` of a live run on `day` fed that
    day's bars as tick bars, its history the bars from `first_open` on."""
    seen: list[tuple] = []

    class Recording(Periods):
        def on_bar(self, bar):
            seen.append((self._ctx.decided_at, reading(self)))

    history = before(BARS, day)
    run = LiveRun(Recording, history[history["ts_open"] >= first_open], 15, day=day)
    for row in on_day(BARS, day).itertuples():
        run.close_slice(live_bar(row))
    return seen


@pytest.fixture(scope="module", params=[MID_WEEK_AND_MONTH, FIRST_OF_A_MONTH], ids=str)
def both_seen(request) -> tuple[list[tuple], list[tuple]]:
    day = request.param
    backtest = [(when, seen) for when, _, seen in backtest_views(Periods, BARS, reading, day)]
    return backtest, live_views(day, backtest[0][1]["first open"])


def test_the_live_run_decides_at_the_backtests_instants(both_seen):
    backtest, live = both_seen
    assert [when for when, _ in live] == [when for when, _ in backtest]


@pytest.mark.parametrize("name", [*PERIODS, *(f"{interval} close" for interval in PERIODS)])
def test_period_bars_equal_the_backtests_at_every_decision(both_seen, name):
    backtest, live = both_seen
    for (when, expected), (_, got) in zip(backtest, live, strict=True):
        np.testing.assert_array_equal(got[name], expected[name], err_msg=f"at {when}")
