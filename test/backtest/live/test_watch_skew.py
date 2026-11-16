"""The clock skew estimate from the feed (`_live/_clock_skew.py`; 16-session-design 6.4,
unit 11): which ticks give a sample, the window and its liquid instruments, the least
sample as the estimate, the brokers whose feed cannot be measured, and the words.

Ticks are built by hand on a window that begins at 0: each trade adds 10 units to its
instrument's day volume, received `offset` seconds after its exchange time.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._clock_skew import (
    SERVER_CLOCK_LTT_BROKERS,
    SKEW_MIN_TRADES,
    SKEW_WINDOW_S,
    ClockSkew,
    SkewEstimate,
    skew_words,
)
from openalgo.backtest._live._types import Tick

NS = 10**9
WINDOW_NS = round(SKEW_WINDOW_S * NS)
#: The exchange time of the first tick of each test, UTC nanoseconds (a time of 2026).
T0 = 1_794_800_000 * NS


class Feed:
    """Ticks of one or more instruments for a `ClockSkew`, each instrument's day volume
    rising by 10 units per trade."""

    def __init__(self, skew: ClockSkew) -> None:
        self.skew = skew
        self.volume: dict[str, float] = {}

    def tick(
        self, symbol: str, offset_s: float, *, traded: bool = True, timed: bool = True
    ) -> None:
        """A tick of `symbol` received `offset_s` seconds after its exchange time; a trade
        when `traded`, with that exchange time when `timed`."""
        self.volume[symbol] = self.volume.get(symbol, 0.0) + (10.0 if traded else 0.0)
        event_ns = T0 if timed else None
        tick = Tick(symbol, "NSE", 500.0, T0, event_ns, self.volume[symbol])
        self.skew.sample(tick, T0 + round(offset_s * NS))

    def trades(self, symbol: str, offsets_s: list[float], *, timed: bool = True) -> None:
        """A first tick that only sets the volume, then a trade for each offset."""
        self.tick(symbol, 0.0, traded=False)
        for offset in offsets_s:
            self.tick(symbol, offset, timed=timed)


@pytest.fixture
def skew():
    skew = ClockSkew("angel")
    skew.start(0)
    return skew


def liquid(least_s: float) -> list[float]:
    """Offsets of a liquid instrument's trades whose least is `least_s`."""
    return [least_s + 0.5] * (SKEW_MIN_TRADES - 1) + [least_s]


def test_the_estimate_is_the_least_sample_of_the_liquid_instruments(skew):
    Feed(skew).trades("SBIN", liquid(3.0))
    assert skew.close_due(WINDOW_NS).seconds == pytest.approx(3.0)


def test_the_window_closes_only_after_its_seconds(skew):
    Feed(skew).trades("SBIN", liquid(3.0))
    assert skew.close_due(WINDOW_NS - 1) is None
    assert skew.close_due(WINDOW_NS) is not None


def test_an_instrument_with_fewer_trades_than_the_minimum_gives_no_estimate(skew):
    Feed(skew).trades("SBIN", [3.0] * (SKEW_MIN_TRADES - 1))
    assert skew.close_due(WINDOW_NS) is None


def test_an_illiquid_instruments_samples_are_left_out(skew):
    feed = Feed(skew)
    feed.trades("SBIN", liquid(3.0))
    feed.trades("INFY", [0.1])
    estimate = skew.close_due(WINDOW_NS)
    assert (estimate.seconds, estimate.symbols) == (pytest.approx(3.0), ("SBIN",))


def test_a_tick_whose_volume_did_not_rise_gives_no_sample(skew):
    feed = Feed(skew)
    feed.trades("SBIN", liquid(3.0))
    feed.tick("SBIN", 0.2, traded=False)
    assert skew.close_due(WINDOW_NS).seconds == pytest.approx(3.0)


def test_the_first_tick_of_an_instrument_gives_no_sample(skew):
    feed = Feed(skew)
    feed.tick("SBIN", 0.2, traded=True)
    for offset in liquid(3.0):
        feed.tick("SBIN", offset)
    assert skew.close_due(WINDOW_NS).seconds == pytest.approx(3.0)


def test_the_next_window_starts_empty(skew):
    Feed(skew).trades("SBIN", liquid(3.0))
    skew.close_due(WINDOW_NS)
    assert skew.close_due(2 * WINDOW_NS) is None


def test_a_negative_estimate_is_a_clock_behind_the_exchange(skew):
    Feed(skew).trades("SBIN", liquid(-2.5))
    assert skew.close_due(WINDOW_NS).seconds == pytest.approx(-2.5)


@pytest.mark.parametrize("broker", sorted(SERVER_CLOCK_LTT_BROKERS))
def test_a_broker_whose_last_trade_time_is_the_servers_clock_is_never_measured(broker):
    skew = ClockSkew(broker)
    skew.start(0)
    Feed(skew).trades("SBIN", liquid(3.0))
    assert skew.close_due(WINDOW_NS) is None
    assert broker in skew.unmeasurable_note()


def test_a_measurable_broker_has_no_banner_note(skew):
    assert skew.unmeasurable_note() is None


def test_liquid_trades_without_an_exchange_time_mark_the_feed_untimed(skew):
    Feed(skew).trades("SBIN", liquid(3.0), timed=False)
    assert skew.close_due(WINDOW_NS) is None
    assert skew.untimed


def test_nothing_is_sampled_before_the_first_window_starts():
    skew = ClockSkew("angel")
    Feed(skew).trades("SBIN", liquid(3.0))
    skew.start(0)
    assert skew.close_due(WINDOW_NS) is None


@pytest.mark.parametrize(
    ("seconds", "beyond"), [(2.0, False), (-2.0, False), (2.1, True), (-2.5, True), (0.0, False)]
)
def test_a_skew_is_beyond_the_limit_only_past_2_s_either_way(seconds, beyond):
    assert SkewEstimate(seconds, 20, ("SBIN",), 0).beyond_limit is beyond


@pytest.mark.parametrize(("seconds", "way"), [(3.04, "3.0 s ahead of"), (-2.56, "2.6 s behind")])
def test_the_words_say_how_far_and_which_way(seconds, way):
    assert way in skew_words(SkewEstimate(seconds, 25, ("SBIN",), 0))
