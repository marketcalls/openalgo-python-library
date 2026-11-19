"""The session plan's catch-up when the broker's candles of today lag or are absent
(`_live/plan.py`, `_plan_release.py`; 16-session-design 2.5 B and C, 3.1 steps 11 to
14, unit 7): a minute whose candle is not published yet is never released without a
bar before the candle checks' last read (close plus 7 s), a slice still without a
candle then is marked missing and named in the warnings, a candle read later fills it,
and each catch-up reads a strategy at most once.

The market is `plan_samples`': SBIN's candles of the two sessions before Monday
2026-11-16 and of that day from 09:15 to the minute each test names; a candle the
broker publishes late is put in when the test says.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import (
    EmaTrend,
    backtest_arguments,
    build,
    candles,
    ist,
    market,
)

import openalgo.backtest as bt
from openalgo.backtest._live._bars_candle import candle_bar

LAGGING = ist(10, 14)  # the minute whose candle the broker publishes late
MINUTE_NS = 60 * 10**9


def candle_of(opened: int, today_until=(10, 15)) -> tuple:
    """The seeded candle (start, open, high, low, close, volume) opening at `opened`."""
    return next(row for row in candles(today_until=today_until) if row[0] == opened)


@pytest.fixture
def lagging(make_fake_openalgo, monkeypatch, tmp_path):
    """Build EmaTrend's plan at an instant while the broker has published today's
    candles up to 10:13 only (10:14's lags); returns (the fake, the session plan)."""

    def make(at):
        fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=at, today_until=(10, 14))
        return fake, build(fake)

    return make


def publish(fake, opened: int) -> None:
    """The broker publishes the late candle opening at `opened`."""
    fake.seed_candles("SBIN", "NSE", [candle_of(opened)])


def test_a_last_candle_not_published_yet_is_not_released_without_a_bar(lagging):
    _, plan = lagging(ist(10, 15, 3))
    part = plan.strategies[0]
    assert part.released == 59
    assert part.uncaught(ist(10, 15, 3)) == (LAGGING,)


def test_a_last_candle_not_published_yet_is_not_marked_missing_before_its_last_read(lagging):
    _, plan = lagging(ist(10, 15, 3))
    assert plan.strategies[0].series.missing_slices(ist(9, 15), ist(10, 15)) == ()


def test_a_candle_published_late_is_caught_up_by_the_next_read(lagging):
    fake, plan = lagging(ist(10, 15, 3))
    publish(fake, LAGGING)
    fake.clock.advance(25)
    plan.catch_up(fake.client(), fake.clock.now)
    part = plan.strategies[0]
    assert part.released == 60
    assert part.series.bar(LAGGING).close == candle_of(LAGGING)[4]
    # The kernel's 5-minute bar closes with the close of its fifth minute, 10:14.
    assert part.engine.strategy_bars(0)["close"][-1] == candle_of(LAGGING)[4]


def test_after_a_late_candle_the_strategy_reads_what_the_backtest_reads(lagging):
    fake, plan = lagging(ist(10, 15, 3))
    publish(fake, LAGGING)
    fake.clock.advance(25)
    plan.catch_up(fake.client(), fake.clock.now)
    seen = []

    class Recording(EmaTrend):
        def on_bar(self, bar):
            seen.append((self._ctx.decided_at, self.close[-1], self.fast[-1]))

    bt.backtest(**backtest_arguments(Recording, fake.client(), fake.clock.now()))
    strategy = plan.strategies[0].strategy
    now = (plan.strategies[0].context.decided_at, strategy.close[-1], strategy.fast[-1])
    assert now == seen[-1]


def test_a_slice_still_without_a_candle_at_its_last_read_is_marked_missing(lagging):
    _, plan = lagging(ist(10, 15, 8))
    part = plan.strategies[0]
    assert (part.released, part.series.bar(LAGGING)) == (60, None)
    assert part.series.missing_slices(ist(9, 15), ist(10, 15)) == (LAGGING,)


def test_the_slices_marked_missing_are_the_ones_the_session_checks_candles_of(lagging):
    _, plan = lagging(ist(10, 15, 8))
    assert plan.strategies[0].missing == (LAGGING,)


def test_a_slice_marked_missing_is_named_in_the_warnings(lagging):
    _, plan = lagging(ist(10, 15, 8))
    assert (
        "EmaTrend: 1 minute of NSE:SBIN (2026-11-16 10:14 IST) marked missing, since no broker "
        "candle of them was read by 7 s after their close: no decision is made on a bar "
        "holding one until its candle is read"
    ) in plan.warnings


def test_a_candle_read_later_for_a_slice_marked_missing_fills_the_series(lagging):
    fake, plan = lagging(ist(10, 15, 8))
    publish(fake, LAGGING)
    fake.clock.advance(20)
    plan.catch_up(fake.client(), fake.clock.now)
    part = plan.strategies[0]
    assert part.series.bar(LAGGING).close == candle_of(LAGGING)[4]
    assert part.missing == ()


def test_the_data_workers_candles_without_the_last_slice_leave_it_for_a_later_read(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(
        make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), today_until=(10, 16)
    )
    part = build(fake).strategies[0]
    # At 10:17:03 the worker's first read holds 10:15's candle, and 10:16's lags.
    row = candle_of(ist(10, 15), today_until=(10, 16))
    fields = dict(zip(("open", "high", "low", "close", "volume"), row[1:], strict=True))
    first = candle_bar("SBIN", "NSE", {"ts_open": row[0], **fields}, MINUTE_NS)
    part.catch_up([first], ist(10, 17, 3))
    assert part.released == 61
    assert part.uncaught(ist(10, 17, 3)) == (ist(10, 16),)


def test_with_no_candle_of_today_the_slices_given_up_on_are_marked_missing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), today_until=None)
    part = build(fake).strategies[0]
    # 09:15 to 10:13 are past their last read; 10:14 (10:15:07) is waited for.
    given_up = tuple(range(ist(9, 15), ist(10, 14), MINUTE_NS))
    assert (part.released, part.missing) == (59, given_up)


def test_with_no_candle_of_today_a_warning_says_what_to_check(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), today_until=None)
    [note] = [note for note in build(fake).warnings if "marked missing" in note]
    assert note.startswith(
        "EmaTrend: 59 minutes of NSE:SBIN (2026-11-16 09:15 IST to 2026-11-16 10:13 IST) "
        "marked missing, since OpenAlgo returned no 1-minute candle of today (check the "
        "broker's login in OpenAlgo"
    )


def test_a_later_read_without_the_new_candles_marks_them_missing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5))
    plan = build(fake)
    fake.clock.advance(125)  # 10:17:10: 10:15 and 10:16 are past their last read
    plan.catch_up(fake.client(), fake.clock.now)
    part = plan.strategies[0]
    assert (part.released, part.missing) == (62, (ist(10, 15), ist(10, 16)))
    assert any("2 minutes of NSE:SBIN" in note for note in plan.warnings)


def test_in_the_first_minute_after_the_open_a_read_with_no_candle_releases_nothing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(9, 16, 3), today_until=None)
    plan = build(fake)
    assert (plan.strategies[0].released, plan.strategies[0].missing) == (0, ())
    assert not [note for note in plan.warnings if "marked missing" in note]


def test_before_the_open_a_read_with_no_candle_warns_of_nothing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(9, 0), today_until=None)
    assert not [note for note in build(fake).warnings if "marked missing" in note]


def history_reads(fake) -> int:
    """How many history calls the fake has answered."""
    return sum(1 for entry in fake.log.calls() if entry.what == "history")


def test_a_catch_up_reads_each_strategy_once_however_slow_the_read(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(
        make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), today_until=(10, 17)
    )
    plan = build(fake)
    fake.clock.advance(60)  # 10:16:05: 10:15 has settled
    fake.faults.slow("history", 70)  # the read ends at 10:17:15, after 10:16 has settled
    before = history_reads(fake)
    plan.catch_up(fake.client(), fake.clock.now)
    part = plan.strategies[0]
    assert history_reads(fake) - before == 1
    assert (part.released, part.uncaught(fake.clock.now())) == (61, (ist(10, 16),))
