"""A live session's bars and the broker's candles, on the fake OpenAlgo
(`_live/session.py`, `_live/_session_trader.py`; 16-session-design 2.5 A.10, B and C,
3.1 steps 11 to 15 and 7.2 row F42, unit 8): a tick gap whose candle the series takes
while the kernel keeps its tick bar, a start in the middle of a minute whose earlier
minutes come from candles, and a decision skipped for missing data.

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


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


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


def test_F42_a_tick_gap_is_drift_and_the_series_continues_from_the_candle(
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
    # Decided at 14:45:01, the 14:43 bar reads as the broker's candle (it opened at 500).
    assert (ist(14, 45), PRICE) in Opens.seen
    # The kernel keeps the tick bar it ran: its first trade was the minute's fourth.
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
