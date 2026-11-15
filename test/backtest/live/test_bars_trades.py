"""Which ticks make a live bar, and with what volume (D27): trades only, the day's
cumulative volume, the plausibility of the feed's event time, and a session that starts
watching in the middle of the day.

Expected bars are written out by hand from the ticks.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest

from openalgo.backtest._live._types import LiveBar, Tick
from openalgo.backtest._live.bars import TickBarBuilder

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
MONDAY = dt.date(2026, 11, 16)
SECOND_NS = 10**9


def ist(clock: str) -> int:
    """UTC nanoseconds of an IST clock time on 2026-11-16."""
    instant = dt.datetime.combine(MONDAY, dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


def traded(symbol: str, clock: str, ltp: float, volume: float | None = None) -> Tick:
    """A tick whose last trade time is `clock` IST, received 40 ms later."""
    at = ist(clock)
    return Tick(symbol, "NSE", ltp, at + 40_000_000, ts_event_ns=at, volume=volume)


def quoted(symbol: str, clock: str, ltp: float, volume: float | None = None) -> Tick:
    """A tick that carries no event time, received at `clock` IST."""
    return Tick(symbol, "NSE", ltp, ist(clock), volume=volume)


def watching_from(clock: str) -> TickBarBuilder:
    return TickBarBuilder("NSE", MONDAY, watching_from_ns=ist(clock))


def bars_of(builder: TickBarBuilder, ticks: list[Tick], until: str) -> list[LiveBar]:
    for one in ticks:
        builder.on_tick(one)
    return builder.close_due(ist(until))


def summary(bars: list[LiveBar]) -> list[tuple]:
    """(IST open, open, high, low, close, volume, ticks) of each bar."""
    return [
        (
            dt.datetime.fromtimestamp(b.ts_open_ns / SECOND_NS, IST).strftime("%H:%M"),
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.ticks,
        )
        for b in bars
    ]


# ----- trades only ------------------------------------------------------------


def test_a_minute_of_quote_updates_with_an_unchanged_volume_makes_no_bar():
    ticks = [
        quoted("OPT", "10:00:10", 12.5, 75),
        quoted("OPT", "10:01:20", 12.5, 75),
        quoted("OPT", "10:01:40", 12.6, 75),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "10:02:01")

    assert summary(bars) == [("10:00", 12.5, 12.5, 12.5, 12.5, 75.0, 1)]


def test_without_a_volume_a_tick_is_a_trade_when_its_last_trade_time_advanced():
    ticks = [
        traded("SBIN", "10:00:10", 800.0),
        Tick("SBIN", "NSE", 801.0, ist("10:00:12"), ts_event_ns=ist("10:00:10")),
        traded("SBIN", "10:00:20", 802.0),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "10:01:01")

    assert summary(bars) == [("10:00", 800.0, 802.0, 800.0, 802.0, 0.0, 2)]


def test_a_zero_volume_from_the_sdk_counts_as_no_volume():
    ticks = [traded("SBIN", "10:00:10", 800.0, 0), traded("SBIN", "10:00:20", 801.0, 0.0)]

    bars = bars_of(watching_from("09:00"), ticks, "10:01:01")

    assert summary(bars) == [("10:00", 800.0, 801.0, 800.0, 801.0, 0.0, 2)]


def test_with_neither_volume_nor_trade_time_a_change_of_ltp_is_a_trade():
    ticks = [
        quoted("NIFTY", "10:00:01", 25000.0),
        quoted("NIFTY", "10:00:02", 25000.0),
        quoted("NIFTY", "10:00:03", 25010.0),
        quoted("NIFTY", "10:00:04", 25005.0),
        quoted("NIFTY", "10:01:30", 25005.0),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "10:02:01")

    assert summary(bars) == [("10:00", 25000.0, 25010.0, 25000.0, 25005.0, 0.0, 3)]


@pytest.mark.parametrize(("volume", "event_ns"), [(None, None), (0, 0)], ids=["none", "sdk-zeros"])
def test_an_index_watched_from_the_open_gets_its_first_bar_from_its_first_in_session_ltp(
    volume, event_ns
):
    ticks = [
        Tick("NIFTY", "NSE", ltp, ist(clock), ts_event_ns=event_ns, volume=volume)
        for clock, ltp in [
            ("09:14:58", 24990.0),
            ("09:15:00.500", 25010.0),
            ("09:15:20", 25010.0),
            ("09:15:40", 25020.0),
        ]
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 25010.0, 25020.0, 25010.0, 25020.0, 0.0, 2)]


def test_a_quote_before_a_symbols_first_trade_of_the_day_makes_no_bar():
    yesterday_trade = ist("15:29:50") - 24 * 3600 * SECOND_NS
    stale = [
        Tick("OPT", "NSE", 12.5, ist(clock), ts_event_ns=yesterday_trade, volume=0)
        for clock in ("09:15:02", "09:30:00", "09:46:00")
    ]

    assert bars_of(watching_from("09:00"), stale, "09:47:01") == []


# ----- volume -----------------------------------------------------------------


def test_volume_is_the_rise_of_the_cumulative_day_volume_within_each_bar():
    ticks = [
        traded("SBIN", "09:15:05", 800.0, 1000),
        traded("SBIN", "09:15:40", 800.5, 1500),
        traded("SBIN", "09:16:10", 801.0, 1700),
        traded("SBIN", "09:16:50", 801.5, 2600),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:17:01")

    assert [b.volume for b in bars] == [1500.0, 1100.0]


def test_a_symbol_first_trading_later_counts_its_volume_from_zero_when_watched_from_the_open():
    ticks = [
        traded("SBIN", "09:15:01", 800.0, 100000),
        traded("OPT", "09:47:10", 12.5, 75),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:48:01")

    assert [(b.symbol, b.volume) for b in bars] == [("SBIN", 100000.0), ("OPT", 75.0)]


def test_a_lower_cumulative_volume_is_no_trade_and_adds_nothing():
    ticks = [
        traded("SBIN", "09:15:05", 800.0, 1000),
        traded("SBIN", "09:16:10", 801.0, 1800),
        traded("SBIN", "09:16:20", 790.0, 1700),
        traded("SBIN", "09:17:10", 802.0, 2000),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:18:01")

    assert summary(bars) == [
        ("09:15", 800.0, 800.0, 800.0, 800.0, 1000.0, 1),
        ("09:16", 801.0, 801.0, 801.0, 801.0, 800.0, 1),
        ("09:17", 802.0, 802.0, 802.0, 802.0, 200.0, 1),
    ]


def test_a_zero_volume_never_lowers_the_count():
    ticks = [
        traded("SBIN", "09:15:05", 800.0, 1000),
        traded("SBIN", "09:15:10", 800.5, 0),
        traded("SBIN", "09:16:05", 801.0, 1200),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:17:01")

    assert [(b.volume, b.ticks) for b in bars] == [(1000.0, 2), (200.0, 1)]


def test_volume_is_a_float_when_the_feed_gives_ints():
    ticks = [traded("SBIN", "10:00:10", 800.0, 5000), traded("SBIN", "10:00:20", 800.5, 5100)]

    (only,) = bars_of(watching_from("09:00"), ticks, "10:01:01")

    assert type(only.volume) is float


def test_volume_is_zero_when_the_feed_gives_none():
    ticks = [traded("SBIN", "09:15:05", 800.0), traded("SBIN", "09:15:30", 801.0)]

    assert [b.volume for b in bars_of(watching_from("09:00"), ticks, "09:16:01")] == [0.0]


# ----- a start in the middle of the session -----------------------------------


def test_after_a_mid_session_start_the_partial_slice_makes_no_bar_and_seeds_the_volume():
    builder = watching_from("11:00:30")
    ticks = [
        traded("SBIN", "11:00:35", 800.0, 500000),
        traded("SBIN", "11:00:50", 800.5, 500100),
        traded("SBIN", "11:01:10", 801.0, 500300),
    ]

    bars = bars_of(builder, ticks, "11:02:01")

    assert summary(bars) == [("11:01", 801.0, 801.0, 801.0, 801.0, 200.0, 1)]


def test_after_a_mid_session_start_a_symbols_first_count_only_sets_the_level():
    ticks = [
        traded("SBIN", "11:00:05", 800.0, 800000),
        traded("SBIN", "11:00:20", 800.5, 800050),
    ]

    bars = bars_of(watching_from("11:00:00"), ticks, "11:01:01")

    assert summary(bars) == [("11:00", 800.5, 800.5, 800.5, 800.5, 50.0, 1)]


def test_after_a_mid_session_start_the_partial_slice_seeds_the_last_price():
    ticks = [
        quoted("NIFTY", "11:00:40", 25000.0),
        quoted("NIFTY", "11:01:05", 25000.0),
        quoted("NIFTY", "11:01:10", 25005.0),
    ]

    bars = bars_of(watching_from("11:00:30"), ticks, "11:02:01")

    assert summary(bars) == [("11:01", 25005.0, 25005.0, 25005.0, 25005.0, 0.0, 1)]


def test_watching_from_the_session_open_makes_the_first_slice_whole():
    bars = bars_of(watching_from("09:15"), [traded("SBIN", "09:15:00.300", 800.0, 400)], "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 800.0, 800.0, 800.0, 400.0, 1)]


# ----- event time against receive time ----------------------------------------


def test_a_plausible_event_time_places_the_tick_in_the_slice_it_traded_in():
    builder = watching_from("09:00")
    builder.on_tick(
        Tick("SBIN", "NSE", 800.0, ist("09:16:00.200"), ts_event_ns=ist("09:15:59.900"), volume=10)
    )

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01"))] == [ist("09:15")]
    assert builder.event_time_rejected == 0


def test_a_tick_without_an_event_time_is_placed_by_its_receive_time():
    builder = watching_from("09:00")
    builder.on_tick(quoted("SBIN", "09:16:00.200", 800.0, 10))

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01"))] == [ist("09:16")]
    assert builder.event_time_rejected == 0


def test_an_event_time_exactly_five_seconds_from_receipt_is_still_used():
    builder = watching_from("09:00")
    builder.on_tick(
        Tick("SBIN", "NSE", 800.0, ist("09:16:04.500"), ts_event_ns=ist("09:15:59.500"), volume=1)
    )

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01"))] == [ist("09:15")]


@pytest.mark.parametrize(
    "event_ns",
    [
        0,
        ist("10:00:30") // 1_000_000,
        ist("10:00:30") + (5 * 3600 + 1800) * SECOND_NS,
        ist("10:00:30") - (5 * 3600 + 1800) * SECOND_NS,
        ist("10:05:30"),
        ist("10:00:24.999"),
    ],
    ids=["zero", "milliseconds", "5h30-late", "5h30-early", "5-minutes-ahead", "over-5s-behind"],
)
def test_an_implausible_event_time_is_rejected_and_the_tick_placed_by_its_receive_time(event_ns):
    builder = watching_from("09:00")
    builder.on_tick(Tick("SBIN", "NSE", 800.0, ist("10:00:30"), ts_event_ns=event_ns, volume=10))

    assert [b.ts_open_ns for b in builder.close_due(ist("10:06:01"))] == [ist("10:00")]
    assert builder.event_time_rejected == 1


def test_a_glitched_event_time_does_not_cut_the_minute_short():
    builder = watching_from("09:00")
    ticks = [
        traded("SBIN", f"10:00:{second:02d}", 800.0 + second, 1000 + second)
        for second in (0, 10, 20)
    ]
    ticks.append(
        Tick("SBIN", "NSE", 821.0, ist("10:00:25"), ts_event_ns=ist("10:05:20"), volume=1030)
    )
    ticks.append(traded("SBIN", "10:00:40", 840.0, 1040))

    bars = bars_of(builder, ticks, "10:06:01")

    assert summary(bars) == [("10:00", 800.0, 840.0, 800.0, 840.0, 1040.0, 5)]
    assert builder.late_ticks == 0


def test_an_implausible_event_time_is_warned_once_per_symbol(caplog):
    builder = watching_from("09:00")
    for second in range(3):
        builder.on_tick(Tick("SBIN", "NSE", 800.0, ist(f"10:00:0{second}"), ts_event_ns=0))
    builder.on_tick(Tick("INFY", "NSE", 1500.0, ist("10:00:05"), ts_event_ns=0))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage().split(":")[1] for r in warnings] == ["SBIN", "INFY"]
    assert builder.event_time_rejected == 4


def test_a_tick_out_of_order_inside_its_bar_sets_the_open_and_close_by_its_time():
    ticks = [
        traded("SBIN", "09:15:20", 801.0),
        traded("SBIN", "09:15:05", 800.0),
        traded("SBIN", "09:15:50", 802.0),
        traded("SBIN", "09:15:40", 799.0),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 802.0, 799.0, 802.0, 0.0, 4)]
