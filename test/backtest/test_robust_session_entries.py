"""Random entries of an all-intraday run stay within one session each (design 17, section
2.4; lead, 2026-11-18): from the entry window to the square-off, a session's trades
together, their holding times kept where they fit, so the random runs hold the run's
trade count.

Expected values are worked by hand from 5-minute closes of NSE sessions (09:15 to 15:30,
75 bars), with the entry window to 15:00 and the square-off at 15:15.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from robust_runs import FREE, nse_sessions

import openalgo.backtest as bt
from openalgo.backtest._robust._random_entries import EntryPlan, Trade, schedule
from openalgo.backtest._robust._session_entries import (
    SessionTimes,
    Window,
    fits,
    place,
    windows,
)
from openalgo.backtest.data._sessions import day_start_ns

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MINUTE = 60 * 10**9
DAY = nse_sessions("2025-03-03", 1)[0]


def at(day: dt.date, hour: int, minute: int) -> int:
    """UTC nanoseconds of an IST clock time on `day`."""
    return int(day_start_ns(day)) + (60 * hour + minute) * MINUTE


def session(day: dt.date) -> SessionTimes:
    """`day`'s NSE session with the entry window to 15:00 and the square-off at 15:15."""
    number = (day - dt.date(1970, 1, 1)).days
    return SessionTimes(
        number, at(day, 9, 15), at(day, 15, 30), at(day, 9, 15), at(day, 15, 0), at(day, 15, 15)
    )


def closes(day: dt.date) -> np.ndarray:
    """The 75 closes of `day`'s 5-minute bars: 09:20 to 15:30."""
    return np.array([at(day, 9, 20) + 5 * MINUTE * k for k in range(75)], dtype=np.int64)


def test_a_sessions_stretch_runs_from_the_entry_window_to_the_square_off():
    # Entries at closes 09:20 (position 0) to 14:55 (67, the last before 15:00); exits up
    # to the close at 15:15 (71), when the square-off would act on a later one.
    assert windows(closes(DAY), [session(DAY)]) == (Window(0, 67, 71),)


def test_a_group_fits_when_its_trades_held_one_bar_each_fit():
    # The largest need takes the largest room, and so on down.
    assert fits([3, 1], [2, 5])
    assert not fits([3, 3], [5, 2])
    assert not fits([1, 1], [5])  # two groups need two sessions


def test_a_hold_longer_than_the_stretch_is_cut_to_fit():
    # One trade held 8 bars in a stretch with its last exit at 5: cut to 5, entered at 0.
    (spot,) = place([8], [], [1], [Window(0, 4, 5)], np.zeros(2, dtype=np.uint64))
    assert (spot.entry, spot.exit) == (0, 5)


@pytest.mark.parametrize("run", range(8))
def test_a_sessions_trades_stay_together_in_one_session_without_overlap(run):
    # Two sessions' groups (days 1 and 2) of two trades each, a reversal between the two
    # of a group; two stretches of positions 0 to 9 and 20 to 29.
    stretches = [Window(0, 7, 9), Window(20, 27, 29)]
    holds, waits, days = [2, 3, 1, 2], [0, 1, 0], [1, 1, 2, 2]
    draws = np.random.default_rng(run).integers(0, 2**63, 8, dtype=np.uint64)
    spots = place(holds, waits, days, stretches, draws)
    assert [spot.place for spot in spots] == [0, 1, 2, 3]
    for first, second in ((spots[0], spots[1]), (spots[2], spots[3])):
        stretch = next(s for s in stretches if s.first <= first.entry <= s.last_exit)
        assert stretch.first <= first.entry and second.entry <= stretch.last_entry
        assert second.exit <= stretch.last_exit
        assert second.entry >= first.exit  # a reversal may go back to back
    assert {spots[0].entry < 20, spots[2].entry < 20} == {True, False}
    assert [spot.exit - spot.entry for spot in spots] == holds


def test_a_schedule_of_an_all_intraday_plan_decides_within_each_session():
    days = nse_sessions("2025-03-03", 3)
    all_closes = np.concatenate([closes(day) for day in days])
    stretches = tuple(windows(all_closes, [session(day) for day in days]))
    number = (days[0] - dt.date(1970, 1, 1)).days
    trades = tuple(Trade("SBIN", "BUY", 1, "MIS", 70, number + k) for k in range(3))
    plan = EntryPlan(trades, {"SBIN": all_closes}, {"SBIN": stretches})
    decided = schedule(plan, seed=4, run=0)
    stamps = sorted(decided)
    entries = [when for when in stamps if decided[when][-1].side == "BUY"]
    exits = [when for when in stamps if decided[when][0].side is None]
    assert len(entries) == len(exits) == 3
    for entry, exit_ in zip(entries, exits, strict=True):
        day = pd.Timestamp(entry, tz="UTC").tz_convert(IST).date()
        assert at(day, 9, 15) <= entry < at(day, 15, 0)
        assert (
            exit_ <= at(day, 15, 15) and pd.Timestamp(exit_, tz="UTC").tz_convert(IST).date() == day
        )


class SessionTrades(bt.Strategy):
    """MIS longs of 10 shares entered at the close of a session's bars 2, 20, 40, 56 and
    64 and exited 5 bars later: five trades a session, each within it."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    ENTRIES = (2, 20, 40, 56, 64)

    def init(self):
        self.day, self.count, self.entered = None, -1, None

    def on_bar(self, bar):
        day = pd.Timestamp(bar.ts_close - 1, tz="UTC").tz_convert(IST).date()
        if day != self.day:
            self.day, self.count = day, -1
        self.count += 1
        if self.position.is_flat and self.count in self.ENTRIES:
            self.buy(qty=10)
            self.entered = self.count
        elif self.position.is_long and self.count - self.entered >= 5:
            self.exit()


def session_bars(days: list[dt.date]) -> pd.DataFrame:
    """5-minute bars that rise 0.10 a bar within each session from 500, back to 500 at
    each open: every long held within a session makes money."""
    stamps, opens = [], []
    for day in days:
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST))
        stamps += [start + pd.Timedelta(minutes=5 * k) for k in range(75)]
        opens += [500.0 + 0.10 * k for k in range(75)]
    opens = np.array(opens)
    index = pd.DatetimeIndex(stamps, name="timestamp")
    columns = {"open": opens, "high": opens + 0.1, "low": opens - 0.05, "close": opens + 0.05}
    return pd.DataFrame({**columns, "volume": 1_000.0}, index=index)


def test_the_random_entries_of_an_all_intraday_run_keep_its_trade_count():
    result = bt.backtest(SessionTrades, data=session_bars(nse_sessions("2025-03-03", 8)), **FREE)
    trades = result.trade_list()
    assert len(trades) == 40 and (trades["entry_time"].dt.date == trades["exit_time"].dt.date).all()
    report = result.robustness(
        checks=["vs_random"],
        workers=1,
        runs=6,
        vs_random={"families": ["random_entries"]},
        progress=lambda line: None,
    )
    table = report.check("vs_random").table
    assert len(table) == 6 and (table["trades"] == 40).all()
