"""The health of the feed (08-live-trading section 8): a dead feed (no tick on any symbol,
which locks entries) told apart from a quiet symbol (an illiquid one while the others
tick), on a manual clock. Ticks count at their receive time.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from openalgo.backtest._live.bars import FeedHealth
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def ist(clock: str) -> int:
    """UTC nanoseconds of an IST clock time on 2026-11-16."""
    instant = dt.datetime.combine(dt.date(2026, 11, 16), dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


SESSION = (ist("09:15"), ist("15:30"))


class ManualClock:
    """The injected `Now`: it moves only when a test sets it."""

    def __init__(self, clock: str) -> None:
        self.ns = ist(clock)

    def __call__(self) -> int:
        return self.ns

    def at(self, clock: str) -> None:
        self.ns = ist(clock)


def health_at(clock: str, **options) -> tuple[FeedHealth, ManualClock]:
    now = ManualClock(clock)
    return FeedHealth(now, session=SESSION, **options), now


# ----- feed dead --------------------------------------------------------------


def test_the_feed_lives_while_some_symbol_ticks_within_dead_after_s():
    health, now = health_at("10:00:00")
    health.on_tick("SBIN", "NSE", now())

    now.at("10:00:09.999")

    assert not health.feed_dead()


def test_the_feed_is_dead_once_no_symbol_has_ticked_for_dead_after_s():
    health, now = health_at("10:00:00")
    health.on_tick("SBIN", "NSE", now())
    now.at("10:00:05")
    health.on_tick("INFY", "NSE", now())

    now.at("10:00:15")

    assert health.feed_dead()


def test_a_feed_with_no_tick_since_the_open_is_dead_after_dead_after_s():
    health, now = health_at("09:00:00")

    now.at("09:15:09.999")
    assert not health.feed_dead()
    now.at("09:15:10")
    assert health.feed_dead()


def test_a_health_created_during_the_session_gives_the_feed_its_full_time():
    health, now = health_at("11:00:00")

    now.at("11:00:09")
    assert not health.feed_dead()
    now.at("11:00:10")
    assert health.feed_dead()


def test_a_tick_brings_a_dead_feed_back():
    health, now = health_at("10:00:00")
    now.at("10:00:30")
    assert health.feed_dead()

    health.on_tick("SBIN", "NSE", now())

    assert not health.feed_dead()


@pytest.mark.parametrize("clock", ["09:14:59", "15:30:00", "16:00:00"])
def test_the_feed_is_never_dead_outside_the_session(clock):
    health, now = health_at("09:00:00")

    now.at(clock)

    assert not health.feed_dead()


def test_dead_after_s_sets_how_long_the_feed_may_be_silent():
    health, now = health_at("10:00:00", dead_after_s=2.5)
    health.on_tick("SBIN", "NSE", now())

    now.at("10:00:02.499")
    assert not health.feed_dead()
    now.at("10:00:02.500")
    assert health.feed_dead()


def test_a_tick_counts_at_its_receive_time_so_a_busy_loop_does_not_kill_a_live_feed():
    health, now = health_at("10:00:00")
    now.at("10:00:08")
    health.on_tick("SBIN", "NSE", ist("10:00:07.500"))

    now.at("10:00:17.499")
    assert not health.feed_dead()
    now.at("10:00:17.500")
    assert health.feed_dead()


def test_a_tick_handled_after_a_later_one_leaves_the_later_receive_time():
    health, _ = health_at("10:00:10")
    health.on_tick("SBIN", "NSE", ist("10:00:09"))
    health.on_tick("SBIN", "NSE", ist("10:00:05"))

    assert health.last_tick_ns("SBIN", "NSE") == ist("10:00:09")


# ----- quiet symbols ----------------------------------------------------------


def test_a_symbol_is_quiet_when_it_has_not_ticked_for_the_seconds_while_others_tick():
    health, now = health_at("10:00:00")
    health.on_tick("ILLIQUID", "NFO", now())
    now.at("10:00:30")
    health.on_tick("SBIN", "NSE", now())

    assert health.quiet("ILLIQUID", "NFO", 30.0)
    assert not health.quiet("ILLIQUID", "NFO", 30.001)
    assert not health.quiet("SBIN", "NSE", 30.0)


def test_a_symbol_that_never_ticked_is_quiet_counted_from_the_open():
    health, now = health_at("09:00:00")
    now.at("09:15:20")
    health.on_tick("SBIN", "NSE", now())

    assert health.quiet("ILLIQUID", "NFO", 20.0)
    assert not health.quiet("ILLIQUID", "NFO", 21.0)


def test_no_symbol_is_quiet_while_the_feed_is_dead():
    health, now = health_at("10:00:00")
    health.on_tick("SBIN", "NSE", now())

    now.at("10:01:00")

    assert health.feed_dead()
    assert not health.quiet("SBIN", "NSE", 30.0)


def test_no_symbol_is_quiet_outside_the_session():
    health, now = health_at("15:00:00")
    health.on_tick("SBIN", "NSE", now())
    now.at("15:30:05")
    health.on_tick("INFY", "NSE", now())

    assert not health.quiet("SBIN", "NSE", 60.0)


def test_symbols_are_told_apart_by_exchange():
    health, now = health_at("10:00:00")
    health.on_tick("SBIN", "BSE", now())
    now.at("10:00:40")
    health.on_tick("SBIN", "NSE", now())

    assert health.quiet("SBIN", "BSE", 30.0)
    assert not health.quiet("SBIN", "NSE", 30.0)


def test_last_tick_ns_is_none_before_a_symbols_first_tick_then_its_last_receive_time():
    health, now = health_at("10:00:00")
    assert health.last_tick_ns("SBIN", "NSE") is None

    now.at("10:00:01.250")
    health.on_tick("SBIN", "NSE", now())

    assert health.last_tick_ns("SBIN", "NSE") == ist("10:00:01.250")


# ----- arguments --------------------------------------------------------------


@pytest.mark.parametrize("seconds", [0, -1.0, math.nan, math.inf, True, "10"])
def test_dead_after_s_must_be_a_positive_number_of_seconds(seconds):
    with pytest.raises(ConfigError, match="dead_after_s"):
        FeedHealth(ManualClock("10:00"), session=SESSION, dead_after_s=seconds)


@pytest.mark.parametrize("seconds", [0, -5.0, math.nan])
def test_quiet_seconds_must_be_a_positive_number(seconds):
    health, _ = health_at("10:00:00")

    with pytest.raises(ConfigError, match="seconds"):
        health.quiet("SBIN", "NSE", seconds)


@pytest.mark.parametrize(
    "session",
    [
        (SESSION[1], SESSION[0]),
        (SESSION[0], SESSION[0]),
        (SESSION[0],),
        (float(SESSION[0]), float(SESSION[1])),
        None,
    ],
)
def test_a_session_must_be_two_instants_with_the_open_first(session):
    with pytest.raises(ConfigError, match="open first"):
        FeedHealth(ManualClock("10:00"), session=session)


def test_a_session_given_as_numpy_instants_is_accepted():
    health = FeedHealth(ManualClock("10:00"), session=tuple(np.int64(ns) for ns in SESSION))

    assert not health.feed_dead()
