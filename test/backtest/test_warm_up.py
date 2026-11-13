"""Automatic indicator warm-up (decision D25, `_warm_up`): no decision before the
strategy's indicators have closed their warm-up of bars, history before `start` loaded
and counted, a wait inside the run when the history is short.

The bars: 40 NSE sessions from 2026-01-01, each opening and closing at the price of an
eight-bar zigzag, 100.00, 99.50, 99.00, 99.50, 100.00, 100.50, 101.00, 100.50 (high and
low 1.00 either side). The strategy buys 10 shares when the 2-bar EMA crosses above the
3-bar EMA and exits when it crosses below, delivery (CNC), no charges, no slippage, so
an order decided at bar k's close fills at bar k + 1's open, which is bar k + 1's price.

Worked by hand (the textbook recursion e[k] = e[k-1] + a (c[k] - e[k-1]), seeded with
the first close, a = 2/3 and 1/2): the fast EMA crosses above the slow one at bars 4,
12, 20, 28 and 36 and below it at bars 8, 16, 24 and 32. Both EMAs settle in the first
window the check tries, 16 bars: over 15 steps a seed gap of at most 2.00 (the price
range) shrinks to at most 2.00 x (1/2)^15 = 0.00006, inside the tolerance 1e-6 x 99.00.
So the strategy is ready at the close of bar 15 (16 bars closed, bars before `start`
counted).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo import ta

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ZIGZAG = (100.0, 99.5, 99.0, 99.5, 100.0, 100.5, 101.0, 100.5)
BARS = 40
DAYS = list(bt.calendar.default_calendar().sessions("NSE", "2026-01-01", "2026-03-31")["date"])[
    :BARS
]
CLOSES = [ZIGZAG[k % len(ZIGZAG)] for k in range(BARS)]
# Bars at whose close the fast EMA crosses above, and below, the slow one.
CROSSES_ABOVE = (4, 12, 20, 28, 36)
CROSSES_BELOW = (8, 16, 24, 32)
WARM_UP_BARS = 16
SETTINGS = {"capital": 100_000, "costs": None, "slippage": 0}


def zigzag(first: int = 0) -> pd.DataFrame:
    """Bars `first` to 39 of the module docstring."""
    closes = np.array(CLOSES[first:])
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in DAYS[first:]],
        name="timestamp",
    )
    columns = {"open": closes, "high": closes + 1.0, "low": closes - 1.0, "close": closes}
    return pd.DataFrame({**columns, "volume": np.full(len(closes), 1_000.0)}, index=index)


def textbook_ema(closes: list[float], length: int) -> list[float]:
    """The recursion of the module docstring, in plain Python."""
    weight, values = 2 / (length + 1), [closes[0]]
    for close in closes[1:]:
        values.append(values[-1] + weight * (close - values[-1]))
    return values


class EmaCross(bt.Strategy):
    """Buys 10 shares when the fast EMA crosses above the slow one, exits below it."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    fast, slow = 2, 3

    def init(self):
        self.fast_line = self.I(ta.ema, self.close, self.fast)
        self.slow_line = self.I(ta.ema, self.close, self.slow)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast_line.crossed_above(self.slow_line):
            self.buy(qty=10)
        elif self.position.is_long and self.fast_line.crossed_below(self.slow_line):
            self.exit()


class EmaCrossConditions(bt.Strategy):
    """EmaCross as conditions."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def init(self):
        fast = self.I(ta.ema, self.close, 2)
        slow = self.I(ta.ema, self.close, 3)
        self.entry_when(fast.crossed_above(slow), qty=10)
        self.exit_when(fast.crossed_below(slow))


class GuardedEmaCross(EmaCross):
    """EmaCross with a warm-up guard of its own."""

    def on_bar(self, bar):
        if len(self.close) < self.slow:
            return
        super().on_bar(bar)


def fills(result: bt.Result) -> list[tuple[dt.date, str]]:
    """(session date, side) of every fill."""
    frame = result.fills
    return list(zip(frame["time"].dt.date, frame["side"], strict=True))


def at_bar(*bars_and_sides: tuple[int, str]) -> list[tuple[dt.date, str]]:
    return [(DAYS[bar], side) for bar, side in bars_and_sides]


# A strategy ready at bar 15 decides first at the crossunder of bar 16 (flat: nothing),
# then buys at bar 20 (fill 21), exits at 24 (fill 25), buys at 28 (fill 29), exits at
# 32 (fill 33) and buys at 36 (fill 37).
AFTER_THE_WARM_UP = at_bar(
    (21, "BUY"), (25, "SELL"), (29, "BUY"), (33, "SELL"), (37, "BUY")
)  # fmt: skip


def test_an_ema_crossover_before_the_warm_up_no_longer_trades():
    # Without the warm-up the crossover at bar 4 bought at bar 5's open.
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_own_data_that_begins_at_start_waits_inside_the_run_and_says_so():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    (warning,) = [text for text in result.warnings if "waited" in text]
    assert warning.startswith(
        "The strategy waited 15 strategy bars inside the run for its indicators, to "
        f"{DAYS[15]} 15:30 IST: SBIN D needs 16 bars (ema(SBIN.close, 2)) and 0 bars "
        f"closed before {DAYS[0]}."
    )
    assert result.manifest["arguments"]["warm_up"]["waited_bars"] == 15


def test_the_first_decision_is_at_the_close_of_the_warm_up_bar():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    assert f"first decision {DAYS[15]} 15:30 IST" in result.assumptions["warm_up"]


def test_history_before_start_lets_the_strategy_trade_from_start():
    # Start at bar 20: the 20 bars before it are history, more than the 16 needed, so
    # the crossover at bar 20, the first bar of the run, buys at bar 21's open.
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP
    assert not [text for text in result.warnings if "waited" in text]
    assert "ready at start" in result.assumptions["warm_up"]


def test_the_indicators_are_computed_over_the_history():
    # At the first decision, bar 20, the fast EMA is the one seeded at bar 0, and the
    # 20 bars of history count in len(self.close).
    seen = []

    class Recording(EmaCross):
        def on_bar(self, bar):
            if not seen:
                seen.append((bar.time.date(), self.fast_line[-1], len(self.close)))
            super().on_bar(bar)

    bt.backtest(Recording, data=zigzag(), start=DAYS[20], **SETTINGS)
    ((day, fast, closed),) = seen
    assert (day, closed) == (DAYS[20], 21)
    assert fast == pytest.approx(textbook_ema(CLOSES, 2)[20], abs=1e-12)


def test_without_history_the_same_start_waits_again():
    # The same run with no bars before bar 20: ready at bar 35, so the crossover at bar
    # 36 is the first trade.
    result = bt.backtest(EmaCross, data=zigzag(20), **SETTINGS)
    assert fills(result) == at_bar((37, "BUY"))


def test_equity_starts_at_start_with_the_history_left_out():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], **SETTINGS)
    assert result.equity.index[0].date() == DAYS[20]
    assert len(result.equity) == BARS - 20


def test_a_condition_strategy_waits_as_on_bar_does():
    result = bt.backtest(EmaCrossConditions, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_a_strategy_that_keeps_its_own_guard_still_works():
    result = bt.backtest(GuardedEmaCross, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_the_assumptions_state_the_rule_the_need_and_the_history():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], **SETTINGS)
    text = result.assumptions["warm_up"]
    assert text.startswith("automatic: no decision before start, nor before every self.I")
    assert "SBIN D needs 16 bars (ema(SBIN.close, 2))" in text
    assert f"history before start: SBIN 20 strategy bars from {DAYS[0]} to {DAYS[19]}" in text


def test_each_indicator_has_its_own_warm_up_in_the_checks():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    checks = result.assumptions["indicator_checks"]
    assert "ema(SBIN.close, 2): truncation passed" in checks
    assert "ema(SBIN.close, 3): truncation passed" in checks


def test_the_manifest_records_the_warm_up_used():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], **SETTINGS)
    warm_up = result.manifest["arguments"]["warm_up"]
    assert warm_up["needs"] == [
        {"symbol": "SBIN", "interval": "D", "bars": WARM_UP_BARS, "indicator": "ema(SBIN.close, 2)"}
    ]
    assert (warm_up["first_decision"], warm_up["waited_bars"]) == (None, 0)
    assert "SBIN history" in result.manifest["data"]


def test_a_rerun_waits_the_same_way():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], **SETTINGS)
    again = result.rerun()
    assert fills(again) == fills(result)
    assert again.manifest["arguments"]["warm_up"] == result.manifest["arguments"]["warm_up"]


def test_every_combination_of_a_sweep_is_its_single_run():
    swept = bt.optimize(
        EmaCross, data=zigzag(), start=DAYS[20], workers=1, fast=(2, 3), slow=(3, 5), **SETTINGS
    )
    for row in swept.table.itertuples():
        params = {"fast": int(row.fast), "slow": int(row.slow)}
        alone = bt.backtest(EmaCross, data=zigzag(), start=DAYS[20], params=params, **SETTINGS)
        assert row.trades == len(alone.trades)
        assert row.objective == alone.metrics["net_profit"]


class WithAverage(EmaCross):
    """EmaCross with a 30-bar simple average registered too."""

    def init(self):
        super().init()
        self.mean = self.I(ta.sma, self.close, 30)


def test_the_longest_warm_up_of_the_strategy_sets_the_wait():
    # A 30-bar average is unknown (NaN) in a 16-bar window and settled exactly by a
    # 32-bar one: ready at bar 31, so the first trade is the crossover at bar 36.
    result = bt.backtest(WithAverage, data=zigzag(), **SETTINGS)
    assert fills(result) == at_bar((37, "BUY"))
    assert "SBIN D needs 32 bars (sma(SBIN.close, 30))" in result.assumptions["warm_up"]


class Cumulative(EmaCross):
    """EmaCross with on-balance volume registered too: a running sum no window settles."""

    def init(self):
        super().init()
        self.balance = self.I(ta.obv, self.close, self.volume)


def test_an_indicator_no_window_settles_is_not_waited_for_and_the_result_says_so():
    # Volume 1,000 + 10 k^2 at bar k: each eight-bar zigzag moves the balance by
    # 10 x (-1 - 4 + 9 + 16 + 25 + 36 - 49 - 64) = -320, so a window that starts later
    # never reaches the running sum.
    data = zigzag()
    data["volume"] = 1_000.0 + 10.0 * np.arange(BARS) ** 2
    result = bt.backtest(Cumulative, data=data, **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP
    (warning,) = [text for text in result.warnings if "does not settle" in text]
    assert warning.startswith("obv(SBIN.close, SBIN.volume) does not settle within the bars")


def test_a_strategy_without_indicators_decides_from_start():
    result = bt.backtest(
        bt.from_arrays(
            [k == 4 for k in range(BARS)],
            [k == 8 for k in range(BARS)],
            symbol="SBIN",
            exchange="NSE",
            interval="D",
            product="CNC",
            qty=10,
        ),
        data=zigzag(),
        **SETTINGS,
    )
    assert fills(result) == at_bar((5, "BUY"), (9, "SELL"))
    assert "no indicator needs a warm-up" in result.assumptions["warm_up"]
