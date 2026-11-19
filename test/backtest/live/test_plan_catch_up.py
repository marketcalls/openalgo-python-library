"""The session plan's catch-up on the fake OpenAlgo (`_live/plan.py`; 16-session-design
3.1 steps 11 to 14, 08-live-trading section 5 rule 7, unit 7): from the open to the
present, today's broker candles feed the live series and the kernel slice by slice, and
the pauses decide nothing and send nothing; a slice that closes while the plan is built
is caught up, never skipped.

The market is `plan_samples`': SBIN's candles of the two sessions before Monday
2026-11-16 and of that day from 09:15 to the time each test names. A slice's candle is
trusted from its close plus 2 s (`plan.CANDLE_SETTLE_S`).
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import (
    DAY,
    HISTORY_DAYS,
    MINUTE_NS,
    EmaTrend,
    build,
    candles,
    ist,
    market,
)

from openalgo import ta
from openalgo.backtest._live._bars_candle import candle_bar
from openalgo.backtest._live._plan_engine import check_grid
from openalgo.backtest._live._types import LiveBar
from openalgo.backtest._live.data_worker import DataWorker
from openalgo.backtest._live.inbox import Candles, Inbox
from openalgo.backtest._live.series import LiveSeries
from openalgo.backtest.errors import ConfigError, LiveError

FIVE_MINUTES = 5


def today(rows: list) -> list:
    """The candles of the live day among `rows`."""
    return [row for row in rows if row[0] >= ist(9, 15)]


def candle(row: tuple) -> LiveBar:
    """A seeded candle (start, open, high, low, close, volume) as a broker candle."""
    start, *prices = row
    fields = dict(zip(("open", "high", "low", "close", "volume"), prices, strict=True))
    return candle_bar("SBIN", "NSE", {"ts_open": start, **fields}, MINUTE_NS)


@pytest.fixture
def plan_at(make_fake_openalgo, monkeypatch, tmp_path):
    """Build EmaTrend's plan (or `strategy`'s) on the market at an instant, with today's
    candles to (hour, minute) `until`; returns (the fake, the session plan)."""

    def make(at, until=(10, 15), strategy=EmaTrend, **options):
        fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=at, today_until=until)
        return fake, build(fake, strategy, **options)

    return make


def test_every_slice_from_the_open_whose_candle_was_read_is_caught_up(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    assert plan.strategies[0].released == 60  # 09:15 to 10:14


def test_a_slice_closed_less_than_two_seconds_before_the_read_waits_for_a_later_one(plan_at):
    _, plan = plan_at(ist(10, 15, 1))
    part = plan.strategies[0]
    assert part.released == 59
    assert part.uncaught(ist(10, 15, 2)) == (ist(10, 14),)


def test_the_live_series_takes_todays_candles(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    series = plan.strategies[0].series
    expected = today(candles())
    assert [series.bar(row[0]).close for row in expected] == [row[4] for row in expected]
    assert {series.bar(row[0]).source for row in expected} == {"candle"}


def test_the_kernel_builds_its_strategy_bars_from_todays_candles(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    bars = plan.strategies[0].engine.strategy_bars(0)
    expected = today(candles())
    # A 5-minute bar closes with the close of its fifth minute.
    closes = [row[4] for row in expected[FIVE_MINUTES - 1 :: FIVE_MINUTES]]
    assert bars["close"].tolist() == closes
    assert bars["ts_close"][-1] == ist(10, 15)


class Watching(EmaTrend):
    """Records every decision it is asked for: on_bar and a timer at 09:30."""

    def init(self):
        super().init()
        self.decisions = []
        self.at("09:30", lambda: self.decisions.append(("timer", self._ctx.decided_at)))

    def on_bar(self, bar):
        self.decisions.append(("on_bar", self._ctx.decided_at))
        self.buy(quantity=1)


def test_the_catch_up_calls_neither_on_bar_nor_a_timer(plan_at):
    _, plan = plan_at(ist(10, 15, 5), strategy=Watching)
    assert plan.strategies[0].strategy.decisions == []


def test_the_catch_up_sends_nothing_to_openalgo(plan_at):
    fake, plan = plan_at(ist(10, 15, 5), strategy=Watching)
    assert plan.strategies[0].engine.take_commands() == []
    assert fake.orders() == []
    assert [entry.what for entry in fake.log.calls() if entry.what == "placeorder"] == []


def test_a_timer_missed_before_the_start_is_logged_as_skipped(plan_at, caplog):
    with caplog.at_level(logging.INFO, logger="openalgo.backtest.live"):
        plan_at(ist(10, 15, 5), strategy=Watching)
    assert "timers due at 2026-11-16 09:30" in caplog.text
    assert "missed before the session started and are skipped" in caplog.text


def test_after_the_catch_up_the_strategy_has_seen_the_last_closed_bar(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    part = plan.strategies[0]
    assert part.context.decided_at == ist(10, 15)
    assert part.strategy.close[-1] == today(candles())[-1][4]


def test_a_minute_without_a_trade_is_released_without_a_bar(
    make_fake_openalgo, monkeypatch, tmp_path
):
    quiet = ist(9, 40)
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), skip=(quiet,))
    part = build(fake).strategies[0]
    assert part.released == 60
    assert part.series.bar(quiet) is None
    bars = part.engine.strategy_bars(0)
    nine_forty = bars["ts_open"].tolist().index(quiet)
    assert bars["open"][nine_forty] == next(
        row[1] for row in candles() if row[0] == quiet + MINUTE_NS
    )


def test_a_minute_without_a_trade_is_not_marked_missing(make_fake_openalgo, monkeypatch, tmp_path):
    quiet = ist(9, 40)
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), skip=(quiet,))
    part = build(fake).strategies[0]
    assert part.missing == ()


def test_before_the_open_nothing_is_caught_up(plan_at):
    _, plan = plan_at(ist(9, 0), until=None)
    part = plan.strategies[0]
    assert part.released == 0
    assert part.setup.today.empty
    assert part.engine.advance()[0] == "waiting"


def test_a_slice_that_closes_while_the_plan_is_built_is_caught_up(
    make_fake_openalgo, monkeypatch, tmp_path
):
    fake = market(
        make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), today_until=(10, 16)
    )
    # Today's first read takes 70 s: the 10:15 slice closes and settles meanwhile.
    fake.faults.slow("history", 70)
    part = build(fake).strategies[0]
    assert part.released == 61
    assert part.series.bar(ist(10, 15)).close == today(candles(today_until=(10, 16)))[-1][4]


def test_slices_that_close_after_the_build_are_caught_up_from_the_data_workers_candles(plan_at):
    fake, plan = plan_at(ist(10, 15, 5), until=(10, 17))
    part = plan.strategies[0]
    fake.clock.advance(120)  # 10:17:05: the 10:14, 10:15 and 10:16 slices have settled
    asked_at = fake.clock.now()
    inbox = Inbox(fake.clock.now)
    worker = DataWorker(fake.client(), inbox, day=DAY, now=fake.clock.now)
    worker.check_candles("SBIN", "NSE", part.uncaught(asked_at), closed_ns=ist(10, 17))
    worker.step()
    message = inbox.take().message
    assert isinstance(message, Candles)
    bars = [candle_bar("SBIN", "NSE", row, MINUTE_NS) for row in message.rows]
    part.catch_up(bars, asked_at)
    assert part.released == 62
    assert part.series.bar(ist(10, 16)).close == today(candles(today_until=(10, 17)))[-1][4]


def test_a_candle_read_again_for_a_slice_caught_up_changes_nothing(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    part = plan.strategies[0]
    version = part.series.version
    part.catch_up([candle(row) for row in today(candles())], ist(10, 15, 5))
    assert (part.released, part.series.version) == (60, version)


def test_the_catch_up_takes_broker_candles_only(plan_at):
    fake, plan = plan_at(ist(10, 15, 5), until=(10, 16))
    fake.clock.advance(60)
    row = today(candles(today_until=(10, 16)))[-1]
    ticks = LiveBar("SBIN", "NSE", row[0], row[0] + MINUTE_NS, *row[1:], ticks=3, source="ticks")
    with pytest.raises(ConfigError, match="the catch-up takes broker candles, not 'ticks' bars"):
        plan.strategies[0].catch_up([ticks], fake.clock.now())


def test_the_catch_up_takes_candles_of_the_strategys_instrument_only(plan_at):
    fake, plan = plan_at(ist(10, 15, 5), until=(10, 16))
    fake.clock.advance(60)
    start, *prices = today(candles(today_until=(10, 16)))[-1]
    fields = dict(zip(("open", "high", "low", "close", "volume"), prices, strict=True))
    other = candle_bar("INFY", "NSE", {"ts_open": start, **fields}, MINUTE_NS)
    with pytest.raises(ConfigError, match="a candle of NSE:INFY reached the catch-up of EmaTrend"):
        plan.strategies[0].catch_up([other], fake.clock.now())


class Ordering:
    """A kernel session whose pauses made an order: every call is the engine's, but
    `take_commands` hands over one Place."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def take_commands(self):
        return [{"command": "place", "order_id": 1}]

    def __getattr__(self, name):
        return getattr(self._engine, name)


def test_an_order_made_during_the_catch_up_is_refused_not_sent(plan_at):
    fake, plan = plan_at(ist(10, 15, 5), until=(10, 16))
    part = plan.strategies[0]
    part.engine = Ordering(part.engine)
    fake.clock.advance(60)
    with pytest.raises(
        LiveError, match="orders were made while the session caught up .* sends nothing"
    ):
        part.catch_up([candle(today(candles(today_until=(10, 16)))[-1])], fake.clock.now())


def test_a_session_whose_exchange_closed_before_the_plan_is_refused(plan_at):
    with pytest.raises(LiveError, match="NSE's session of 2026-11-16 ended before the plan"):
        plan_at(ist(15, 30, 5), until=(15, 30))


class LongAverage(EmaTrend):
    interval = "1m"

    def init(self):
        self.line = self.I(ta.sma, self.close, 100)


def test_a_history_too_short_for_an_indicator_refuses_the_plan(
    make_fake_openalgo, monkeypatch, tmp_path
):
    rows = candles()
    last_half_hour = [
        row for row in rows if ist(15, 0, day=HISTORY_DAYS[-1]) <= row[0] < ist(9, 15)
    ]
    fake = market(make_fake_openalgo, monkeypatch, tmp_path, at=ist(10, 15, 5), rows=last_half_hour)
    with pytest.raises(ConfigError, match="too short to compute sma"):
        build(fake, LongAverage)


class Counting(EmaTrend):
    def init(self):
        super().init()
        self.state.update(trades=0, set_in_init=True)


def test_the_state_saved_by_an_earlier_start_wins_over_what_init_set(plan_at):
    _, plan = plan_at(ist(10, 15, 5), strategy=Counting, user_states={"Counting": {"trades": 2}})
    assert plan.strategies[0].strategy.state == {"trades": 2, "set_in_init": True}


def test_the_indicators_are_ready_once_their_warm_up_has_closed(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # One session before the day: 75 of the 128 bars the 10-bar EMA needs. Its 128th bar
    # is the day's 53rd, which closes at 09:15 plus 53 times 5 minutes.
    fake = market(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        at=ist(13, 45, 5),
        days=HISTORY_DAYS[-1:],
        today_until=(13, 45),
    )
    part = build(fake).strategies[0]
    assert not part.indicators_ready(ist(13, 35))
    assert part.indicators_ready(ist(13, 40))


def test_with_the_history_of_two_sessions_the_indicators_are_ready_at_the_open(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    assert plan.strategies[0].indicators_ready(ist(9, 20))


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"first_day": 20_000}, "kernel session's NSE grid"),
        ({"base_nanos": 5 * MINUTE_NS}, "kernel session's NSE grid"),
    ],
)
def test_a_kernel_grid_other_than_the_bar_builders_is_refused(plan_at, change, words):
    _, plan = plan_at(ist(10, 15, 5))
    part = plan.strategies[0]
    config = {**part.setup.config, **change}
    with pytest.raises(LiveError, match=words):
        check_grid(config, "NSE", DAY, part.calendar, part.series)


def test_a_live_series_of_another_day_is_refused_by_the_grid_check(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    part = plan.strategies[0]
    other = LiveSeries("SBIN", "NSE", HISTORY_DAYS[-1])
    with pytest.raises(LiveError, match="holds other slices than the bars are built on"):
        check_grid(part.setup.config, "NSE", DAY, part.calendar, other)


def test_the_kernels_grid_is_the_bar_builders(plan_at):
    _, plan = plan_at(ist(10, 15, 5))
    part = plan.strategies[0]
    check_grid(part.setup.config, "NSE", DAY, part.calendar, part.series)
    assert len(part.series.slices) == 375 and part.series.slices[0] == (ist(9, 15), ist(9, 16))
