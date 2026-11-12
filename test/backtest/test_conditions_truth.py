"""A truth test of a view while `init` declares the strategy is refused (14-strategy-api
4.2): `and`, `or`, `not`, `if` and a chained comparison such as `a < b < c` read one true
or false, which a condition over the whole run does not have, so they would quietly keep
only part of a condition. Conditions combine with `&`, `|` and `~`. Once the run has
started (timers, callbacks, `on_bar`, `on_stop`) a view of booleans reads its latest
value as before.

Condition runs use `condition_runs.walk_bars` (5-minute SBIN bars over two sessions);
the timer run uses `strategy_samples.minute_bars` on a 15-minute strategy, whose first
bar closes at 09:30 IST.
"""

import pytest
from condition_runs import run, walk_bars
from strategy_samples import minute_bars

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.errors import ConfigError

DATA = walk_bars([6, 5, -3, 4, -8, -6, 3, -2, 7, -1, -9, 2])
ADVICE = r"combine conditions with & \(and\), \| \(or\) and ~ \(not\)"


class Lines(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def lines(self):
        """A fast and a slow EMA of the close."""
        return self.I(ta.ema, self.close, 5), self.I(ta.ema, self.close, 12)


def declaring(entry):
    """A strategy whose init declares `entry(self, fast, slow)` as its entry condition."""

    class Declares(Lines):
        def init(self):
            fast, slow = self.lines()
            self.entry_when(entry(self, fast, slow), qty=10)

    return Declares


@pytest.mark.parametrize(
    "entry",
    [
        lambda s, fast, slow: fast.crossed_above(slow) and s.close > slow,
        lambda s, fast, slow: fast.crossed_above(slow) or s.close > slow,
        lambda s, fast, slow: not fast.crossed_above(slow),
        lambda s, fast, slow: slow < fast < s.close,
    ],
    ids=["and", "or", "not", "chained-comparison"],
)
def test_a_truth_test_inside_a_declaration_is_refused(entry):
    with pytest.raises(ConfigError, match=ADVICE):
        run(declaring(entry), DATA)


def test_the_refusal_says_how_to_write_a_chained_comparison():
    chained = declaring(lambda s, fast, slow: slow < fast < s.close)
    with pytest.raises(ConfigError, match=r"write a < b < c as \(a < b\) & \(b < c\)"):
        run(chained, DATA)


def test_an_if_on_a_condition_in_init_is_refused():
    class Branches(Lines):
        def init(self):
            fast, slow = self.lines()
            if fast > slow:
                self.entry_when(fast.crossed_above(slow), qty=10)

    with pytest.raises(ConfigError, match=ADVICE):
        run(Branches, DATA)


def test_a_truth_test_of_numbers_in_init_names_the_series():
    class Numbers(Lines):
        def init(self):
            if self.close:
                self.entry_when(self.close > 500, qty=10)

    with pytest.raises(ConfigError, match=r"SBIN\.close was tested for truth in init"):
        run(Numbers, DATA)


def test_a_truth_test_of_another_interval_in_init_is_refused():
    class Trend(Lines):
        def init(self):
            quarter = self.data["SBIN", "15m"]
            self.entry_when(quarter.close > quarter.open and self.close > 500, qty=10)

    rising = r"\(SBIN\[15m\]\.close > SBIN\[15m\]\.open\) was tested for truth in init"
    with pytest.raises(ConfigError, match=rising):
        run(Trend, DATA)


# ----------------------------------------------------------------- once the run starts


def test_a_timer_reads_a_condition_before_the_first_bar_has_closed():
    seen = []

    class Early(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"
        product = "CNC"

        def init(self):
            self.at("09:20", lambda: seen.append(bool(self.close > 0)))

    bt.backtest(Early, data=minute_bars(), capital=1_000_000, costs=None, slippage=0)
    # No 15-minute bar has closed at 09:20, so the condition reads false.
    assert seen == [False]


def test_a_callback_reads_a_condition():
    seen = []

    class Fills(Lines):
        def on_bar(self, bar):
            if self.position.is_flat and not self.orders:
                self.buy(qty=10)

        def on_fill(self, fill):
            seen.append(bool(self.close > 0))

    run(Fills, DATA)
    assert seen[0] is True


def test_on_stop_of_a_condition_strategy_reads_a_condition():
    seen = []

    class Stops(Lines):
        def init(self):
            fast, slow = self.lines()
            self.entry_when(fast.crossed_above(slow), qty=10)

        def on_stop(self):
            # The Rust driver reveals no bar to Python, so nothing is visible: false.
            seen.append(bool(self.close > 0))

    run(Stops, DATA)
    assert seen == [False]
