"""`self.data[symbol, interval]`, `self.history` and the single-instrument shortcuts
(14-strategy-api 2.2): another interval is visible only once its bar has closed
(03-architecture 4.2, "Higher timeframes").

Bars: `strategy_samples.minute_bars`, the bar opening k minutes after 09:15 opening at
500.00 + 0.10 k (high + 0.50, low - 0.50, close + 0.05, volume 1,000).
"""

import pandas as pd
import pytest
from strategy_samples import at, minute_bars, run

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError


class OneMinute(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    product = "CNC"


def views_at(interval, decisions, strategy=OneMinute, **settings):
    """(decision instant, bars visible, latest bar's values) at the given decisions."""
    seen = {}

    class Watching(strategy):
        def on_bar(self, bar):
            if self.now in decisions:
                view = self.data["SBIN", interval]
                latest = None
                if len(view):
                    latest = (view.open[-1], view.high[-1], view.low[-1], view.close[-1])
                seen[self.now] = (len(view), latest)

    run(Watching, **settings)
    return seen


def test_a_15_minute_bar_on_a_1_minute_strategy_appears_only_after_its_close():
    seen = views_at("15m", (at(9, 29), at(9, 30)))
    # The 09:15 to 09:30 bar: open of k = 0, high and low over k = 0 to 14, close of k = 14.
    assert seen == {
        at(9, 29): (0, None),
        at(9, 30): (1, (500.0, 501.9, 499.5, 501.45)),
    }


def test_a_daily_view_appears_at_the_session_close():
    seen = views_at("D", (at(15, 29), at(15, 30)))
    assert seen[at(15, 29)] == (0, None)
    assert seen[at(15, 30)][0] == 1
    assert seen[at(15, 30)][1][0] == 500.0


def test_a_finer_interval_than_the_strategy_works_from_the_base_bars():
    class Quarterly(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"
        product = "CNC"

    seen = views_at("1m", (at(9, 30),), strategy=Quarterly)
    # Fifteen 1-minute bars have closed by 09:30; the last is k = 14.
    assert seen[at(9, 30)] == (15, (501.4, 501.9, 500.9, 501.45))


def test_an_interval_the_base_cannot_give_is_refused():
    class FromQuarters(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"
        product = "CNC"

        def init(self):
            self.data["SBIN", "5m"]

    with pytest.raises(ConfigError, match="cannot be built from the run's 15m base bars"):
        run(FromQuarters, base="15m", data=minute_bars().resample("15min").first())


def test_history_returns_the_last_n_strategy_bars():
    frames = []

    class Remembering(OneMinute):
        def on_bar(self, bar):
            if self.now == at(9, 18):
                frames.append(self.history("SBIN", 2))

    run(Remembering)
    expected = pd.DataFrame(
        {
            "open": [500.1, 500.2],
            "high": [500.6, 500.7],
            "low": [499.6, 499.7],
            "close": [500.15, 500.25],
            "volume": [1000.0, 1000.0],
        },
        index=pd.DatetimeIndex([at(9, 16), at(9, 17)], name="time"),
    )
    pd.testing.assert_frame_equal(frames[0], expected, check_index_type=False, rtol=1e-12)


def test_the_shortcuts_are_the_single_instrument_series():
    found = []

    class Shortcut(OneMinute):
        def on_bar(self, bar):
            if self.now == at(9, 16):
                found.append((self.close[-1], self.data["SBIN"].close[-1], len(self.volume)))

    run(Shortcut)
    assert found == [(500.05, 500.05, 1)]


def test_a_view_after_the_run_shows_every_bar():
    found = []

    class AtStop(OneMinute):
        def on_stop(self):
            found.append((len(self.close), len(self.data["SBIN", "15m"])))

    run(AtStop)
    assert found == [(375, 25)]
