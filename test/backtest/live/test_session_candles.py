"""A live session's bars and the broker's candles, on the fake OpenAlgo
(`_live/session.py`, `_live/_session_trader.py`; 16-session-design 2.5 A.10, B and C,
3.1 steps 11 to 15 and 7.2 row F42, unit 8): a tick gap whose candle is kept as drift
while the strategy and the kernel keep the tick bar, a start in the middle of a minute
whose earlier minutes come from candles, and a decision skipped for missing data.

The strategy keeps the bars it decided on (lead decision 2026-11-21, 2.4.0 item P2-1):
a broker candle read after a decision never replaces the tick bar of its minute, so a
crossing between the tick bar and a later, different candle of the same minute is seen
by exactly one decision; a candle only fills a minute that has no tick bar.

SBIN trades every 10 s from the start; in the tick gap test the price climbs by 0.50
with each trade of a minute, so a bar missing its first trades has another open than the
broker's candle. That test also trades INFY, whose ticks go on through SBIN's gap: with
no tick on any instrument for 10 s the feed would be dead, and the gap's slice would make
no tick bar at all (16-session-design 6.5).
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, PRICE, ist, rig

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live._series_store import LiveStore
from openalgo.backtest._live._types import LOGGER_NAME


class Opens(bt.Strategy):
    """A 1-minute strategy that keeps, at each decision, the open of the bar before the
    one that just closed (a bar whose broker candle has been read by then)."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    seen: list[tuple[int, float]] = []

    def on_bar(self, bar):
        type(self).seen.append((bar.ts_close, float(self.open[-2])))


class InfyTicks(bt.Strategy):
    """Trades nothing: its instrument, INFY, keeps the feed ticking."""

    symbol, exchange, interval = "INFY", "NSE", "1m"

    def on_bar(self, bar):
        return None


class Every5(bt.Strategy):
    """A 5-minute strategy that keeps the closes of its decisions."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    seen: list[int] = []

    def on_bar(self, bar):
        type(self).seen.append(bar.ts_close)


class LastCloses(bt.Strategy):
    """A 1-minute strategy that keeps, at each decision, the closes of its last four
    bars, by the decision's close."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    seen: dict[int, list[float]] = {}

    def on_bar(self, bar):
        type(self).seen[bar.ts_close] = list(self.close.values[-4:])


#: The level the crossing tests' two-bar average crosses above, rupees.
LEVEL = 500.8


class AverageCross(bt.Strategy):
    """A 1-minute strategy that keeps the closes of the decisions at which the two-bar
    average of the close (`self.I`) crosses above `LEVEL`."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    crossed: list[int] = []

    def init(self):
        self.average = self.I(ta.sma, self.close, 2)

    def on_bar(self, bar):
        if self.average.crossed_above(LEVEL)[-1]:
            type(self).crossed.append(bar.ts_close)


#: SBIN's price in the 14:41 minute, the one watching begins in after a start at
#: 14:40:58 and the countdown; every other minute trades at `PRICE`, so the series holds
#: this close only when it holds that minute's bar.
SEED_PRICE = 503.0


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


@pytest.fixture
def late_seed_candle(make_fake_openalgo, monkeypatch, tmp_path):
    """A session started at 14:40:58 whose watching begins in the 14:41 minute, which
    trades at `SEED_PRICE`. The reads of that minute's candle at 14:42:02, 14:42:04 and
    14:42:07 fail, so its candle is read at 14:43:02, after the bar of 14:42 has closed
    at 14:43:00. Stopped at 14:45:30."""
    LastCloses.seen = {}
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 58), trades_until=None)
    world.script.trade()
    world.script.price = lambda now_ns: SEED_PRICE if ist(14, 41) <= now_ns < ist(14, 42) else PRICE

    def candle_reads_fail():
        for _ in range(3):
            world.fake.faults.gateway_error("history", status=503)

    world.script.trades(ist(14, 41), ist(15, 2))
    world.script.at(ist(14, 42), candle_reads_fail)
    world.stop_at(ist(14, 45, 30))
    return world


def crossing(make_fake_openalgo, monkeypatch, tmp_path, *, traded: float, candle: tuple):
    """A session started at 14:40:05 in which SBIN trades at `PRICE` up to 14:43, at
    `traded` in the 14:43 minute (its tick bar's close) and at 503.00 from 14:44. The
    broker's candle of the 14:43 minute, read at 14:44:02, is `candle` (open, high, low,
    close, volume): it differs from the tick bar, as a candle read two seconds after the
    close often does. Stopped at 14:46:30."""
    AverageCross.crossed = []
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)

    def price(now_ns: int) -> float:
        if now_ns < ist(14, 43):
            return PRICE
        return traded if now_ns < ist(14, 44) else 503.0

    world.script.price = price
    candle_row = (ist(14, 43), *candle)
    world.script.at(
        ist(14, 43, 59) + NS // 2,
        lambda: world.fake.seed_candles("SBIN", "NSE", [candle_row]),
    )
    world.stop_at(ist(14, 46, 30))
    return world


def test_a_crossing_between_a_tick_bar_and_a_later_candle_is_seen_once_on_the_tick_bars(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Two-bar averages on the tick bars: 14:43 (500.00 + 500.50) / 2 = 500.25, 14:44
    # (500.50 + 503.00) / 2 = 501.75, so the average crosses 500.80 at the 14:45 decision.
    # Had the candle (close 502.00) replaced the 14:43 bar after its decision, that
    # decision would read 501.00 before 501.75 and no decision would see the crossing.
    world = crossing(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        traded=500.5,
        candle=(500.5, 502.0, 500.5, 502.0, 60.0),
    )
    world.run(AverageCross)
    assert AverageCross.crossed == [ist(14, 45)]


def test_a_candle_that_differs_the_other_way_makes_no_second_crossing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Two-bar averages on the tick bars: 14:43 (500.00 + 502.00) / 2 = 501.00, crossing
    # 500.80 at the 14:44 decision; 14:44 (502.00 + 503.00) / 2 = 502.50, no crossing.
    # Had the candle (close 500.50) replaced the 14:43 bar, the 14:45 decision would read
    # 500.25 before 501.75 and see the same crossing a second time.
    world = crossing(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        traded=502.0,
        candle=(502.0, 502.0, 500.5, 500.5, 60.0),
    )
    world.run(AverageCross)
    assert AverageCross.crossed == [ist(14, 44)]


def test_a_minute_the_dead_feed_left_without_a_bar_is_filled_by_its_candle_for_the_next_decision(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The feed dies at 14:43:30 and lives again at 14:43:52: the 14:43 minute has no tick
    # bar and is marked missing. Its trades at 14:43:05 to :35 are at 500.00 and at :45
    # and :55 at 506.00, so its broker candle, read at 14:44:02, closes at 506.00.
    LastCloses.seen = {}
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    world.script.price = lambda now_ns: 506.0 if now_ns >= ist(14, 43, 40) else PRICE
    world.script.at(ist(14, 43, 30), world.fake.faults.feed_dead)
    world.script.at(ist(14, 43, 52), world.fake.faults.feed_alive)
    world.stop_at(ist(14, 45, 30))
    outcome = world.run(LastCloses)
    assert LastCloses.seen[ist(14, 45)] == [PRICE, PRICE, 506.0, 506.0]
    assert outcome.skipped_decisions == ()


def climbing(now_ns: int) -> float:
    """500.00 at each minute's first trade, 0.50 more at each later one."""
    return PRICE + 0.5 * ((now_ns // NS) % 60 // 10)


def kernel_opens(monkeypatch) -> dict[int, float]:
    """The open of each SBIN strategy bar the kernel closes, by its open (UTC
    nanoseconds), as its rows hand them to the series: (instrument, ts_open, ts_close,
    open, ...)."""
    opens: dict[int, float] = {}
    reveal = LiveStore.reveal

    def keep(self, row):
        if self.symbol == "SBIN":
            opens[row[1]] = row[3]
        return reveal(self, row)

    monkeypatch.setattr(LiveStore, "reveal", keep)
    return opens


def test_F42_a_tick_gap_is_drift_and_the_strategy_keeps_the_tick_bar(
    make_fake_openalgo, monkeypatch, tmp_path
):
    Opens.seen = []
    kernel = kernel_opens(monkeypatch)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.script.price = climbing
    # The first three trades of SBIN in the 14:43 minute send no tick.
    world.script.at(ist(14, 42, 59), lambda: world.fake.faults.tick_gap("SBIN", "NSE", trades=3))
    world.stop_at(ist(14, 46))
    outcome = world.run([Opens, InfyTicks])
    drift = [d for d in outcome.drift if d.symbol == "SBIN"]
    gap = [d for d in drift if d.ts_open_ns == ist(14, 43)]
    assert len(gap) == 1
    assert gap[0].open == pytest.approx(-1.5)
    others = [d for d in drift if d.ts_open_ns != ist(14, 43)]
    assert others and all(d.open == 0 and d.close == 0 for d in others)
    # Decided at 14:45:01, the 14:43 bar is still the tick bar the 14:44 decision read:
    # its first trade was the minute's fourth (the broker's candle opened at 500.00).
    assert (ist(14, 45), PRICE + 1.5) in Opens.seen
    # The kernel keeps the same tick bar.
    assert kernel[ist(14, 43)] == pytest.approx(PRICE + 1.5)


def test_a_start_mid_minute_catches_the_minutes_before_it_up_from_candles(
    make_fake_openalgo, monkeypatch, tmp_path
):
    Every5.seen = []
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 58))
    world.stop_at(ist(14, 51))
    outcome = world.run(Every5)
    # Watching began in the 14:41 minute: its bar seeds only, the 14:40 minute closed
    # after the plan was built and came from its candle; neither keeps the 14:45
    # decision from being made.
    assert Every5.seen == [ist(14, 45), ist(14, 50)]
    assert outcome.skipped_decisions == ()


def test_a_decision_on_a_bar_with_a_minute_of_no_data_is_skipped(
    make_fake_openalgo, monkeypatch, tmp_path
):
    Every5.seen = []
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 58))

    def candles_fail():
        for _ in range(60):
            world.fake.faults.gateway_error("history", status=503)

    world.script.at(ist(14, 40, 58), candles_fail)
    world.stop_at(ist(14, 46))
    outcome = world.run(Every5)
    assert Every5.seen == []
    assert [(s.decided_at_ns, s.reason) for s in outcome.skipped_decisions] == [
        (ist(14, 45), "missing data")
    ]
    assert any("marked missing" in warning for warning in outcome.warnings)


def test_a_seed_minute_with_no_trade_is_lifted_once_a_later_candle_is_read(
    make_fake_openalgo, monkeypatch, tmp_path
):
    Every5.seen = []
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 58), trades_until=None)
    # SBIN trades at the start, then not in the 14:41 minute, then every 10 s.
    world.script.trade()
    world.script.trades(ist(14, 42), ist(15, 2))
    world.stop_at(ist(14, 51))
    outcome = world.run(Every5)
    assert Every5.seen == [ist(14, 45), ist(14, 50)]
    assert outcome.skipped_decisions == ()


def test_a_decision_waits_while_the_minute_watching_began_in_has_no_candle(late_seed_candle):
    # The 14:42 bar does not hold that minute, but a crossing reads the bar before it:
    # deciding at 14:43 would read a series with a hole the candle fills two seconds later.
    outcome = late_seed_candle.run(LastCloses)
    assert [(s.decided_at_ns, s.reason) for s in outcome.skipped_decisions] == [
        (ist(14, 43), "missing data")
    ]


def test_every_decision_after_a_start_mid_minute_reads_that_minutes_bar(late_seed_candle):
    late_seed_candle.run(LastCloses)
    assert {close: SEED_PRICE in closes for close, closes in LastCloses.seen.items()} == {
        ist(14, 44): True,
        ist(14, 45): True,
    }
