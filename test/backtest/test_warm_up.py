"""Automatic indicator warm-up (decision D25, `_warm_up`): no decision before the
strategy's indicators have closed their warm-up of bars, history before `start` loaded
and counted, a wait inside the run when the history is short.

The bars: 72 NSE sessions from 2026-01-01, each opening and closing at the price of an
eight-bar zigzag, 100.00, 99.50, 99.00, 99.50, 100.00, 100.50, 101.00, 100.50 (high and
low 1.00 either side). The strategy buys 10 shares when the 2-bar EMA crosses above the
3-bar EMA and exits when it crosses below, delivery (CNC), no charges, no slippage, so
an order decided at bar k's close fills at bar k + 1's open, which is bar k + 1's price.

Worked by hand (the textbook recursion e[k] = e[k-1] + a (c[k] - e[k-1]), seeded with
the first close, a = 2/3 and 1/2): the fast EMA crosses above the slow one at bars 4,
12, 20, ..., 68 and below it at bars 8, 16, ..., 64, whichever bar the EMAs are seeded
at before them. On the engine's reference bars the 2-bar EMA settles in the first window
the check tries, 16 bars, and the 3-bar EMA in the second, 32: over 15 steps its seed
gap shrinks by (1/2)^15 = 0.00003, inside the tolerance 1e-6 only for a gap below 3
percent of the price, and the reference bars open with gaps of up to 8 percent. So the
strategy is ready at the close of the 32nd bar of its series, bars before `start`
counted, and a run keeps those 32 bars of history before `start`.
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
BARS = 72
DAYS = list(bt.calendar.default_calendar().sessions("NSE", "2026-01-01", "2026-05-31")["date"])[
    :BARS
]
CLOSES = [ZIGZAG[k % len(ZIGZAG)] for k in range(BARS)]
WARM_UP_BARS = 32
# The first bar of the runs that start with history: a crossing above, after more bars
# than the warm-up needs.
START = 36
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


# A strategy ready at bar 31 decides first at the crossunder of bar 32 (flat: nothing),
# then buys at bar 36 (fill 37), exits at 40 (fill 41), and so on every 4 bars to the buy
# at bar 68 (fill 69).
AFTER_THE_WARM_UP = at_bar(
    *((bar, "BUY" if (bar - 37) % 8 == 0 else "SELL") for bar in range(37, BARS, 4))
)


def test_an_ema_crossover_before_the_warm_up_no_longer_trades():
    # Without the warm-up the crossover at bar 4 would buy at bar 5's open.
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_own_data_that_begins_at_start_waits_inside_the_run_and_says_so():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    (warning,) = [text for text in result.warnings if "waited" in text]
    assert warning == (
        "The strategy waited inside the run for its indicators: SBIN waited 31 strategy "
        f"bars, to {DAYS[31]} 15:30 IST. ema(SBIN.close, 3) needs 32 D bars of SBIN and 0 "
        f"bars closed before {DAYS[0]}: the bars given for SBIN begin on {DAYS[0]}; give "
        f"earlier ones to decide from {DAYS[0]}."
    )
    assert result.manifest["arguments"]["warm_up"]["waited"] == {"SBIN": 31}


def test_the_first_decision_is_at_the_close_of_the_warm_up_bar():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    assert f"first decision {DAYS[31]} 15:30 IST" in result.assumptions["warm_up"]


def test_history_before_start_lets_the_strategy_trade_from_start():
    # Start at bar 36: the 36 bars before it are history, more than the 32 needed, so
    # the crossover at bar 36, the first bar of the run, buys at bar 37's open.
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP
    assert not [text for text in result.warnings if "waited" in text]
    assert "ready at start" in result.assumptions["warm_up"]


def test_the_indicators_are_computed_over_the_history_they_need():
    # The run keeps the 32 bars before bar 36 its EMAs need, bars 4 to 35: at the first
    # decision, bar 36, the fast EMA is the one seeded at bar 4, and the 32 bars of
    # history count in len(self.close).
    seen = []

    class Recording(EmaCross):
        def on_bar(self, bar):
            if not seen:
                seen.append((bar.time.date(), self.fast_line[-1], len(self.close)))
            super().on_bar(bar)

    bt.backtest(Recording, data=zigzag(), start=DAYS[START], **SETTINGS)
    ((day, fast, closed),) = seen
    assert (day, closed) == (DAYS[START], 33)
    assert fast == pytest.approx(textbook_ema(CLOSES[4:], 2)[32], abs=1e-12)


def test_without_history_the_same_start_waits_again():
    # The same run with no bars before bar 36: ready at its 32nd bar, bar 67, so the
    # crossover at bar 68 is the first trade.
    result = bt.backtest(EmaCross, data=zigzag(START), **SETTINGS)
    assert fills(result) == at_bar((69, "BUY"))


def test_equity_starts_at_start_with_the_history_left_out():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], **SETTINGS)
    assert result.equity.index[0].date() == DAYS[START]
    assert len(result.equity) == BARS - START


def test_a_condition_strategy_waits_as_on_bar_does():
    result = bt.backtest(EmaCrossConditions, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_a_strategy_that_keeps_its_own_guard_still_works():
    result = bt.backtest(GuardedEmaCross, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP


def test_the_assumptions_state_the_rule_the_need_and_the_history():
    # 32 sessions x 7/5 x 1.1 = 49.28, so 50 days and a week more, 57, which the ladder
    # of spans (16 days, each step a quarter longer, rounded up: 16, 20, 25, 32, 40, 50,
    # 63) rounds up to 63 days before start. The bars begin on 2026-01-01, more than a
    # week after that day, which the line says.
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], **SETTINGS)
    text = result.assumptions["warm_up"]
    asked = DAYS[START] - dt.timedelta(days=63)
    assert text.startswith("automatic: no decision before start, nor before every self.I")
    assert "ema(SBIN.close, 3) needs 32 D bars of SBIN" in text
    assert (
        f"history before start: SBIN 32 strategy bars from {DAYS[4]} to {DAYS[35]} (asked "
        f"from {asked}; the source returned bars from {DAYS[0]})"
    ) in text


def test_each_indicator_has_its_own_warm_up_in_the_checks():
    result = bt.backtest(EmaCross, data=zigzag(), **SETTINGS)
    checks = result.assumptions["indicator_checks"]
    assert "ema(SBIN.close, 2): truncation passed" in checks
    assert "ema(SBIN.close, 3): truncation passed" in checks


def test_the_manifest_records_the_warm_up_used():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], **SETTINGS)
    warm_up = result.manifest["arguments"]["warm_up"]
    assert warm_up["needs"] == [
        {"symbol": "SBIN", "interval": "D", "bars": WARM_UP_BARS, "indicator": "ema(SBIN.close, 3)"}
    ]
    assert (warm_up["first_decision"], warm_up["waited"], warm_up["not_waited_for"]) == (
        None,
        {},
        [],
    )
    assert "SBIN history" in result.manifest["data"]


def test_a_rerun_waits_the_same_way():
    result = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], **SETTINGS)
    again = result.rerun()
    assert fills(again) == fills(result)
    assert again.manifest["arguments"]["warm_up"] == result.manifest["arguments"]["warm_up"]


def test_every_combination_of_a_sweep_is_its_single_run():
    swept = bt.optimize(
        EmaCross, data=zigzag(), start=DAYS[START], workers=1, fast=(2, 3), slow=(3, 5), **SETTINGS
    )
    for row in swept.table.itertuples():
        params = {"fast": int(row.fast), "slow": int(row.slow)}
        alone = bt.backtest(EmaCross, data=zigzag(), start=DAYS[START], params=params, **SETTINGS)
        assert row.trades == len(alone.trades)
        assert row.objective == alone.metrics["net_profit"]


class WithAverage(EmaCross):
    """EmaCross with a 40-bar simple average registered too."""

    def init(self):
        super().init()
        self.mean = self.I(ta.sma, self.close, 40)


def test_the_longest_warm_up_of_the_strategy_sets_the_wait():
    # A 40-bar average is unknown (NaN) in a 32-bar window and settled exactly by a
    # 64-bar one: ready at bar 63, so the first trade is the crossover at bar 68.
    result = bt.backtest(WithAverage, data=zigzag(), **SETTINGS)
    assert fills(result) == at_bar((69, "BUY"))
    assert "sma(SBIN.close, 40) needs 64 D bars of SBIN" in result.assumptions["warm_up"]


class Cumulative(EmaCross):
    """EmaCross with on-balance volume registered too: a running sum no window settles."""

    def init(self):
        super().init()
        self.balance = self.I(ta.obv, self.close, self.volume)


def test_an_indicator_no_window_settles_is_not_waited_for_and_the_result_says_so():
    # On-balance volume is a running sum of signed volumes: a window that starts later
    # never reaches it, on the reference bars as on any.
    result = bt.backtest(Cumulative, data=zigzag(), **SETTINGS)
    assert fills(result) == AFTER_THE_WARM_UP
    (warning,) = [text for text in result.warnings if "does not settle" in text]
    assert warning == (
        "obv(SBIN.close, SBIN.volume) does not settle on the engine's reference bars (it "
        "depends on its whole history, as a running sum does): the strategy did not wait "
        "for it."
    )
    assert result.manifest["arguments"]["warm_up"]["not_waited_for"] == [
        "obv(SBIN.close, SBIN.volume)"
    ]


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
