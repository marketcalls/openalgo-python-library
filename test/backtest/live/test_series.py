"""The live series and its stores (`_live/series.py`; 16-session-design 2.5 A.10, B and
C): today's base bars added slice by slice, each slice's bar taken once (the strategy
keeps the bars it decided on: a broker candle fills only a slice with no bar, lead
decision 2026-11-21, 2.4.0 item P2-1), missing-data marks set and lifted, and bars
visible from the decision at their close.

The bars are `series_samples.minute_bars`: the live day 2025-07-01 on NSE, its 375
one-minute slices from 09:15 IST.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from series_samples import LIVE_DAY, MINUTE_NS, before, live_bar, minute_bars, on_day

from openalgo.backtest._live.series import LiveSeries
from openalgo.backtest.errors import CalendarError, ConfigError, LiveError

BARS = minute_bars()
TODAY = on_day(BARS, LIVE_DAY)
HISTORY = before(BARS, LIVE_DAY)
FIVE_MINUTES_NS = 5 * MINUTE_NS


class Clock:
    """A decision instant a test sets (UTC nanoseconds), None before the first."""

    def __init__(self) -> None:
        self.at: int | None = None

    def __call__(self) -> int | None:
        return self.at


def series_with(slices: int, *, history=None) -> LiveSeries:
    """The live day's series with the tick bars of its first `slices` slices."""
    series = LiveSeries("SBIN", "NSE", LIVE_DAY, history=history)
    for row in TODAY.iloc[:slices].itertuples():
        series.take(live_bar(row))
    return series


def five_minute_store(series: LiveSeries, clock: Clock):
    return series.store("5m", clock)


def test_init_sees_no_bar_and_the_history_is_counted():
    clock = Clock()
    store = five_minute_store(series_with(0, history=HISTORY), clock)
    assert store.visible() == 0
    assert store.history_bars == len(HISTORY) // 5


def test_the_history_is_visible_from_the_first_decision():
    clock = Clock()
    store = five_minute_store(series_with(0, history=HISTORY), clock)
    clock.at = int(TODAY.ts_open.iloc[0])
    assert store.visible() == store.history_bars


def test_a_strategy_bar_is_visible_from_the_decision_at_its_close():
    clock = Clock()
    store = five_minute_store(series_with(5), clock)
    first_close = int(TODAY.ts_open.iloc[0]) + FIVE_MINUTES_NS
    clock.at = first_close - 1
    assert store.visible() == 0
    clock.at = first_close
    assert store.visible() == 1


def test_a_forming_strategy_bar_is_not_shown():
    clock = Clock()
    store = five_minute_store(series_with(7), clock)
    clock.at = int(TODAY.ts_close.iloc[6])
    assert list(store.column("ts_open")) == [int(TODAY.ts_open.iloc[0])]


def test_a_strategy_bar_is_built_from_its_base_bars():
    clock = Clock()
    store = five_minute_store(series_with(5), clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    first = TODAY.iloc[:5]
    assert (store.column("open")[0], store.column("close")[0]) == (
        first.open.iloc[0],
        first.close.iloc[4],
    )
    assert (store.column("high")[0], store.column("low")[0]) == (first.high.max(), first.low.min())
    assert store.column("volume")[0] == first.volume.sum()


def test_a_candle_never_replaces_the_tick_bar_of_its_slice():
    # The bar a decision read stays the slice's bar for every later close.
    clock = Clock()
    series = series_with(5)
    store = five_minute_store(series, clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    candle = dataclasses.replace(
        live_bar(TODAY.iloc[4], "candle"),
        high=TODAY.high.iloc[4] + 1.0,
        close=TODAY.high.iloc[4] + 1.0,
    )
    with pytest.raises(ConfigError, match="keeps the bars it decided on"):
        series.take(candle)
    assert store.column("close")[0] == TODAY.close.iloc[4]


def test_a_candle_filling_a_slice_changes_the_stores_version():
    clock = Clock()
    series = series_with(4)
    store = five_minute_store(series, clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    before_candle = store.version
    series.take(live_bar(TODAY.iloc[4], "candle"))
    assert store.version != before_candle


def test_the_store_keeps_its_version_while_nothing_changes():
    clock = Clock()
    store = five_minute_store(series_with(5), clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    assert store.version == store.version


def test_the_candle_of_a_slice_with_no_tick_bar_fills_the_store():
    clock = Clock()
    series = series_with(4)
    series.take(live_bar(TODAY.iloc[4], "candle"))
    store = five_minute_store(series, clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    assert store.column("close")[0] == TODAY.close.iloc[4]


def test_another_tick_bar_for_a_slice_with_one_is_refused():
    series = series_with(1)
    other = dataclasses.replace(live_bar(TODAY.iloc[0]), volume=TODAY.volume.iloc[0] + 1)
    with pytest.raises(ConfigError, match="has a bar already"):
        series.take(other)


def test_a_tick_bar_after_the_candle_of_its_slice_is_refused():
    series = series_with(0)
    series.take(live_bar(TODAY.iloc[0], "candle"))
    with pytest.raises(ConfigError, match="has a bar already"):
        series.take(live_bar(TODAY.iloc[0]))


def test_another_candle_for_a_slice_a_candle_filled_is_refused():
    series = series_with(0)
    series.take(live_bar(TODAY.iloc[0], "candle"))
    later = dataclasses.replace(live_bar(TODAY.iloc[0], "candle"), volume=TODAY.volume.iloc[0] + 7)
    with pytest.raises(ConfigError, match="has a bar already"):
        series.take(later)


def test_a_slice_marked_missing_is_reported_for_its_strategy_bar():
    series = series_with(2)
    gap = int(TODAY.ts_open.iloc[2])
    series.mark_missing(gap)
    first_open = int(TODAY.ts_open.iloc[0])
    assert series.missing_slices(first_open, first_open + FIVE_MINUTES_NS) == (gap,)


def test_a_mark_is_not_reported_for_another_strategy_bar():
    series = series_with(2)
    series.mark_missing(int(TODAY.ts_open.iloc[2]))
    later = int(TODAY.ts_open.iloc[5])
    assert series.missing_slices(later, later + FIVE_MINUTES_NS) == ()


def test_a_bar_of_a_missing_slice_lifts_its_mark():
    series = series_with(2)
    gap = int(TODAY.ts_open.iloc[2])
    series.mark_missing(gap)
    series.take(live_bar(TODAY.iloc[2], "candle"))
    assert series.missing_slices(gap, gap + MINUTE_NS) == ()


def test_a_slice_with_a_bar_cannot_be_marked_missing():
    series = series_with(1)
    with pytest.raises(ConfigError, match="only a slice without one"):
        series.mark_missing(int(TODAY.ts_open.iloc[0]))


def test_the_bar_a_slice_has_now_is_given():
    series = series_with(1)
    series.take(live_bar(TODAY.iloc[1], "candle"))
    assert series.bar(int(TODAY.ts_open.iloc[0])).source == "ticks"
    assert series.bar(int(TODAY.ts_open.iloc[1])).source == "candle"
    assert series.bar(int(TODAY.ts_open.iloc[2])) is None


def test_the_tick_bar_of_a_slice_is_at_hand():
    series = series_with(1)
    assert series.tick_bar(int(TODAY.ts_open.iloc[0])) == live_bar(TODAY.iloc[0])


def test_a_slice_with_only_a_candle_has_no_tick_bar():
    series = series_with(0)
    series.take(live_bar(TODAY.iloc[0], "candle"))
    assert series.tick_bar(int(TODAY.ts_open.iloc[0])) is None


def test_a_candle_equal_to_the_bar_its_slice_has_changes_nothing():
    series = series_with(0)
    candle = live_bar(TODAY.iloc[0], "candle")
    series.take(candle)
    before_again = series.version
    series.take(candle)
    assert series.version == before_again


def test_the_tick_bar_its_slice_has_taken_again_changes_nothing():
    series = series_with(1)
    opened = int(TODAY.ts_open.iloc[0])
    before_again = series.version
    series.take(series.tick_bar(opened))
    assert (series.version, series.bar(opened)) == (before_again, live_bar(TODAY.iloc[0]))


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"symbol": "INFY"}, "reached the series"),
        ({"exchange": "BSE"}, "reached the series"),
        ({"source": "quote"}, "ticks or is a candle"),
        ({"ts_open_ns": int(TODAY.ts_open.iloc[0]) + 1}, "opens no 1m slice"),
        ({"ts_close_ns": int(TODAY.ts_close.iloc[0]) + MINUTE_NS}, "not at its slice's close"),
    ],
)
def test_a_bar_that_is_not_one_of_todays_slices_is_refused(change, words):
    series = LiveSeries("SBIN", "NSE", LIVE_DAY)
    with pytest.raises(ConfigError, match=words):
        series.take(dataclasses.replace(live_bar(TODAY.iloc[0]), **change))


def test_history_after_today_begins_is_refused():
    with pytest.raises(ConfigError, match="before today"):
        LiveSeries("SBIN", "NSE", LIVE_DAY, history=BARS)


def test_history_without_the_bars_layout_is_refused():
    with pytest.raises(ConfigError, match="lacks the columns"):
        LiveSeries("SBIN", "NSE", LIVE_DAY, history=HISTORY.drop(columns="volume"))


def test_a_day_the_exchange_has_no_session_on_is_refused():
    with pytest.raises(CalendarError, match="no session"):
        LiveSeries("SBIN", "NSE", "2025-07-05")


def test_a_base_interval_that_is_not_intraday_is_refused():
    with pytest.raises(ConfigError, match="intraday"):
        LiveSeries("SBIN", "NSE", LIVE_DAY, base="D")


def test_an_interval_the_base_cannot_give_is_refused():
    with pytest.raises(ConfigError, match="whole multiple"):
        LiveSeries("SBIN", "NSE", LIVE_DAY, base="5m").store("7m", Clock())


def test_a_kernel_bar_the_series_does_not_hold_is_a_live_error():
    clock = Clock()
    store = five_minute_store(series_with(5), clock)
    clock.at = int(TODAY.ts_close.iloc[9])
    with pytest.raises(LiveError, match="does not hold"):
        store.reveal((0, int(TODAY.ts_open.iloc[5])))


def test_the_bar_revealed_is_the_series_bar_of_the_kernels_row():
    clock = Clock()
    store = five_minute_store(series_with(5), clock)
    clock.at = int(TODAY.ts_close.iloc[4])
    bar = store.reveal((0, int(TODAY.ts_open.iloc[0])))
    assert (bar.symbol, bar.ts_open, bar.ts_close) == (
        "SBIN",
        int(TODAY.ts_open.iloc[0]),
        int(TODAY.ts_close.iloc[4]),
    )


def test_a_daily_store_shows_today_only_after_the_session_close():
    clock = Clock()
    series = series_with(len(TODAY), history=HISTORY)
    store = series.store("D", clock)
    clock.at = int(TODAY.ts_close.iloc[-2])
    assert store.visible() == store.history_bars
    clock.at = int(TODAY.ts_close.iloc[-1])
    assert store.visible() == store.history_bars + 1
    assert np.isclose(store.column("close")[-1], TODAY.close.iloc[-1])
