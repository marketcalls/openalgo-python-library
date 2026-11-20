"""One strategy's broker candles in a live session (`_live/_session_candles.py`;
16-session-design 2.5 A.10, B and C): the strategy keeps the bars it decided on (lead
decision 2026-11-21, 2.4.0 item P2-1). A candle of a slice with a tick bar is checked
against it and kept as drift, never taken by the series; a candle of a slice with no bar
fills it, and a later candle of a slice never replaces the bar it has.

The bars are `series_samples.minute_bars` of the live day; the keeper reads a strategy's
plan through `Part`, a stand-in holding a real live series whose first slices are
released, with no catch-up left (every slice is released live).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from series_samples import LIVE_DAY, live_bar, minute_bars, on_day

from openalgo.backtest._live._session_candles import CandleKeeper
from openalgo.backtest._live.series import LiveSeries
from openalgo.backtest.calendar import default_calendar

TODAY = on_day(minute_bars(), LIVE_DAY)
#: A read of the broker's candles begun this long after the day's first slice opened,
#: UTC nanoseconds: after every slice the tests release.
READ_NS = int(TODAY.ts_open.iloc[0]) + 10 * 60 * 10**9


class Part:
    """What the candle keeper reads of a strategy's plan: its live series, the slices
    released (the first `released`), its spec, and a catch-up that owns no slice."""

    def __init__(self, series: LiveSeries, released: int) -> None:
        self.series, self.released = series, released
        self.slices = series.slices
        self.spec = SimpleNamespace(name="S", symbol="SBIN", exchange="NSE")
        self.calendar = default_calendar()

    @property
    def missing(self) -> tuple[int, ...]:
        return self.series.missing_slices(self.slices[0][0], self.slices[self.released - 1][1])

    def catch_up(self, candles: Any, read_ns: int, *, before: Any, refuse_orders: bool) -> tuple:
        return ()


class Events:
    """The event stream's lines the keeper writes, as (kind, fields)."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, dict]] = []

    def __call__(self, kind: str, **fields: Any) -> None:
        self.lines.append((kind, fields))


def opened(slot: int) -> int:
    return int(TODAY.ts_open.iloc[slot])


def candle_row(slot: int, **changed: float) -> dict:
    """The broker's candle of slice `slot` as a history row, its fields `changed`."""
    row = TODAY.iloc[slot]
    fields = {name: float(row[name]) for name in ("open", "high", "low", "close", "volume")}
    return {"ts_open": opened(slot), **fields, **changed}


def keeper_of(tick_bars: int, released: int) -> tuple[CandleKeeper, LiveSeries, Events]:
    """A keeper whose series has the tick bars of its first `tick_bars` slices and whose
    first `released` slices are released (those without a tick bar made none)."""
    series = LiveSeries("SBIN", "NSE", LIVE_DAY)
    for slot in range(tick_bars):
        series.take(live_bar(TODAY.iloc[slot]))
    events = Events()
    keeper = CandleKeeper(Part(series, released), event=events)
    keeper.watch(0)
    for slot in range(released):
        keeper.closed(opened(slot))
    return keeper, series, events


def higher_close(slot: int) -> dict:
    """Slice `slot`'s candle closing 0.50 above its tick bar, its high raised with it."""
    row = TODAY.iloc[slot]
    return candle_row(slot, close=float(row.high) + 0.5, high=float(row.high) + 0.5)


def test_the_series_keeps_the_tick_bar_of_a_slice_whose_candle_differs():
    keeper, series, _ = keeper_of(tick_bars=2, released=2)
    keeper.take([higher_close(0)], READ_NS)
    assert series.bar(opened(0)) == live_bar(TODAY.iloc[0])


def test_a_candle_of_a_slice_with_a_tick_bar_is_kept_as_drift():
    keeper, _, _ = keeper_of(tick_bars=2, released=2)
    keeper.take([higher_close(0)], READ_NS)
    (drift,) = keeper.drifts
    expected = float(TODAY.high.iloc[0]) + 0.5 - float(TODAY.close.iloc[0])
    assert (drift.ts_open_ns, drift.close) == (opened(0), pytest.approx(expected))


def test_a_slice_whose_candle_was_checked_is_asked_for_no_more():
    keeper, _, _ = keeper_of(tick_bars=2, released=2)
    keeper.take([candle_row(0)], READ_NS)
    assert keeper.wanted(READ_NS) == (opened(1),)


def test_a_candle_read_again_records_no_second_drift():
    keeper, _, _ = keeper_of(tick_bars=2, released=2)
    keeper.take([higher_close(0)], READ_NS)
    keeper.take([higher_close(0)], READ_NS)
    assert len(keeper.drifts) == 1


def test_a_later_different_candle_of_a_checked_slice_is_kept_as_drift_too():
    keeper, series, _ = keeper_of(tick_bars=2, released=2)
    keeper.take([candle_row(0)], READ_NS)
    keeper.take([higher_close(0)], READ_NS)
    assert [drift.ts_open_ns for drift in keeper.drifts] == [opened(0), opened(0)]
    assert series.bar(opened(0)) == live_bar(TODAY.iloc[0])


def test_a_candle_fills_a_released_slice_with_no_bar():
    keeper, series, _ = keeper_of(tick_bars=2, released=3)
    keeper.take([candle_row(2)], READ_NS)
    assert series.bar(opened(2)) == live_bar(TODAY.iloc[2], "candle")


def test_a_later_different_candle_never_replaces_the_candle_that_filled_a_slice():
    keeper, series, _ = keeper_of(tick_bars=2, released=3)
    keeper.take([candle_row(2)], READ_NS)
    keeper.take([higher_close(2)], READ_NS)
    assert series.bar(opened(2)) == live_bar(TODAY.iloc[2], "candle")
