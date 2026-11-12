"""`self.at` and `self.every` (14-strategy-api 2.3): timers fire in step 2 of the slice
and the orders they place fill at that same open (03-architecture 4.2 step 2).

Intraday bars: `strategy_samples.minute_bars`, the 1-minute bar opening k minutes after
09:15 opening at 500.00 + 0.10 k, on a 15-minute strategy with a 1-minute base. Daily
bars: one per session, the n-th opening at 100 + n.
"""

import datetime as dt

import pytest
from strategy_samples import at, daily_bars, run

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError


class Quarterly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"
    product = "CNC"


def utc_nanos(moment: dt.datetime) -> int:
    return int(moment.timestamp()) * 1_000_000_000


def test_an_order_placed_by_a_timer_fills_at_that_same_open():
    class BuyAtTwenty(Quarterly):
        def init(self):
            self.at("09:20", self.enter)

        def enter(self):
            self.buy(qty=10)

    result = run(BuyAtTwenty)
    fills = result.fills
    # 09:20 is k = 5: it opens at 500.50.
    assert list(fills["time"]) == [at(9, 20)]
    assert list(fills["price_rupees"]) == [500.5]
    assert list(result.orders["decided_at"]) == [utc_nanos(at(9, 20))]


def test_a_timer_runs_at_the_open_it_is_due_at():
    seen = []

    class Clock(Quarterly):
        def init(self):
            self.at("10:00", lambda: seen.append((self.now, len(self.close))))

    run(Clock)
    # The 09:45 strategy bar closed at 10:00, in step 7 of the slice before: three bars.
    assert seen == [(at(10, 0), 3)]


def test_a_timer_sees_the_fills_due_at_its_open():
    seen = []

    class SeesFill(Quarterly):
        def init(self):
            self.at("09:30", lambda: seen.append(self.position.qty))

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(qty=10)  # decided at 09:30, fills at the 09:30 open

    run(SeesFill)
    assert seen == [10.0]


def test_a_session_without_a_bar_at_or_after_the_time_is_reported():
    class TooLate(Quarterly):
        def init(self):
            self.at("15:45", lambda: None)

    warnings = run(TooLate).warnings
    assert "self.at('15:45') did not run in 1 of 1 sessions" in " ".join(warnings)


@pytest.mark.parametrize("time_ist", ["9:20", "24:00", "09:60", 920])
def test_a_timer_time_is_hh_mm(time_ist):
    class BadTime(Quarterly):
        def init(self):
            self.at(time_ist, lambda: None)

    with pytest.raises(ConfigError, match="HH:MM"):
        run(BadTime)


def test_a_timer_is_registered_in_init():
    class Late(Quarterly):
        def on_bar(self, bar):
            self.at("10:00", lambda: None)

    with pytest.raises(ConfigError, match="registered in init"):
        run(Late)


def test_every_is_for_daily_or_longer_strategies():
    class IntradayEvery(Quarterly):
        def init(self):
            self.every("week", lambda: None)

    with pytest.raises(ConfigError, match="daily or longer"):
        run(IntradayEvery)


def test_every_takes_a_known_period():
    class Fortnight(Quarterly):
        def init(self):
            self.every("fortnight", lambda: None)

    with pytest.raises(ConfigError, match="week"):
        run(Fortnight)


class Weekly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "D"


def test_every_week_orders_fill_at_the_open_after_the_first_session_of_the_week():
    # Tuesday 1 July to Wednesday 9 July 2025: Monday 30 June opened the first week (not
    # in the run); Monday 7 July opens the second, so the timer runs at the open of
    # Tuesday 8 July, the sixth session, which opens at 105.
    days = [dt.date(2025, 7, day) for day in (1, 2, 3, 4, 7, 8, 9)]
    seen = []

    class Rebalancing(Weekly):
        def init(self):
            self.every("week", self.rebalance)

        def rebalance(self):
            seen.append(self.now)
            self.buy(qty=10)

    result = run(Rebalancing, data=daily_bars(days))
    tuesday_open = at(9, 15, dt.date(2025, 7, 8))
    assert seen == [tuesday_open]
    assert list(result.fills["time"]) == [tuesday_open]
    assert list(result.fills["price_rupees"]) == [105.0]
