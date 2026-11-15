"""Which ticks make a live bar, and with what volume (D27): trades only (a volume that
rose or an LTP that changed; an index by its value), the day's cumulative volume, the
plausibility of the feed's event time, and a session that starts watching in the middle
of the day.

Ticks are shaped as OpenAlgo sends them: the SDK fills a missing volume or last trade
time with 0, and the server may stamp the last trade time with the packet's time.
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
DAY_NS = 24 * 3600 * SECOND_NS


def ist(clock: str) -> int:
    """UTC nanoseconds of an IST clock time on 2026-11-16."""
    instant = dt.datetime.combine(MONDAY, dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


def traded(
    symbol: str, clock: str, ltp: float, volume: float | None = None, exchange: str = "NSE"
) -> Tick:
    """A tick whose last trade time is `clock` IST, received 40 ms later."""
    at = ist(clock)
    return Tick(symbol, exchange, ltp, at + 40_000_000, ts_event_ns=at, volume=volume)


def quoted(
    symbol: str, clock: str, ltp: float, volume: float | None = None, exchange: str = "NSE"
) -> Tick:
    """A tick that carries no event time, received at `clock` IST."""
    return Tick(symbol, exchange, ltp, ist(clock), volume=volume)


def watching_from(clock: str, exchange: str = "NSE") -> TickBarBuilder:
    return TickBarBuilder(exchange, MONDAY, watching_from_ns=ist(clock))


def bars_of(builder: TickBarBuilder, ticks: list[Tick], until: str) -> list[LiveBar]:
    for one in ticks:
        builder.on_tick(one)
    return list(builder.close_due(ist(until)).bars)


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


# ----- trades only: tradable instruments --------------------------------------


def test_a_minute_of_quote_updates_with_an_unchanged_volume_and_ltp_makes_no_bar():
    ticks = [
        quoted("OPT", "10:00:10", 12.5, 75),
        quoted("OPT", "10:01:20", 12.5, 75),
        quoted("OPT", "10:01:40", 12.5, 75),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "10:02:01")

    assert summary(bars) == [("10:00", 12.5, 12.5, 12.5, 12.5, 75.0, 1)]


def option_ticks(shape: str) -> list[Tick]:
    """Quotes of an NFO option not traded today, at the previous close, as the feed
    sends them: the SDK's zeros, LTP mode, or the packet's time as the trade time."""
    clocks = ("09:15:02", "09:15:30", "09:16:10", "09:17:40")
    if shape == "sdk-zeros":
        return [Tick("OPT", "NFO", 12.5, ist(c), ts_event_ns=0, volume=0) for c in clocks]
    if shape == "ltp-mode":
        return [Tick("OPT", "NFO", 12.5, ist(c)) for c in clocks]
    return [
        Tick("OPT", "NFO", 12.5, ist(c), ts_event_ns=ist(c) - 200_000_000, volume=0) for c in clocks
    ]


@pytest.mark.parametrize("shape", ["sdk-zeros", "ltp-mode", "packet-time"])
def test_an_option_not_traded_today_makes_no_bar_however_the_feed_shapes_its_quotes(shape):
    builder = watching_from("09:00", "NFO")

    assert bars_of(builder, option_ticks(shape), "09:18:01") == []


def test_the_trade_time_never_decides_a_trade():
    ticks = [
        traded("SBIN", "10:00:10", 800.0),
        traded("SBIN", "10:00:20", 800.0),
        traded("SBIN", "10:00:30", 800.0),
    ]

    assert bars_of(watching_from("09:00"), ticks, "10:01:01") == []


def test_without_a_volume_a_change_of_ltp_is_a_trade_and_the_first_ltp_only_seeds():
    ticks = [
        quoted("SBIN", "10:00:01", 800.0),
        quoted("SBIN", "10:00:02", 800.0),
        quoted("SBIN", "10:00:03", 801.0),
        quoted("SBIN", "10:00:04", 800.5),
        quoted("SBIN", "10:01:30", 800.5),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "10:02:01")

    assert summary(bars) == [("10:00", 801.0, 801.0, 800.5, 800.5, 0.0, 2)]


def test_two_trades_with_the_same_whole_second_trade_time_close_on_the_later_one():
    ticks = [
        Tick("SBIN", "NSE", 800.0, ist("10:00:30.200"), ts_event_ns=ist("10:00:30")),
        Tick("SBIN", "NSE", 801.0, ist("10:00:59.100"), ts_event_ns=ist("10:00:59")),
        Tick("SBIN", "NSE", 805.0, ist("10:00:59.800"), ts_event_ns=ist("10:00:59")),
    ]

    bars = bars_of(watching_from("09:00"), [ticks[0], ticks[2], ticks[1]], "10:01:01")

    assert summary(bars) == [("10:00", 801.0, 805.0, 801.0, 805.0, 0.0, 2)]


def test_a_zero_volume_from_the_sdk_counts_as_no_volume():
    ticks = [traded("SBIN", "10:00:10", 800.0, 0), traded("SBIN", "10:00:20", 801.0, 0.0)]

    bars = bars_of(watching_from("09:00"), ticks, "10:01:01")

    assert summary(bars) == [("10:00", 801.0, 801.0, 801.0, 801.0, 0.0, 1)]


def test_a_tick_before_the_open_seeds_the_last_known_ltp():
    ticks = [
        quoted("SBIN", "09:08:00", 799.0),
        quoted("SBIN", "09:15:01", 800.0),
        quoted("SBIN", "09:15:20", 802.0),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 802.0, 800.0, 802.0, 0.0, 2)]


def test_a_stale_quote_received_before_a_trade_is_judged_before_it():
    builder = watching_from("09:00")
    ticks = [
        Tick("ILLQ", "NSE", 800.0, ist("10:00:50.340"), ts_event_ns=ist("10:00:50"), volume=1000),
        Tick("ILLQ", "NSE", 800.0, ist("10:01:00.030"), ts_event_ns=ist("10:00:50"), volume=1000),
        Tick("ILLQ", "NSE", 801.0, ist("10:01:00.440"), ts_event_ns=ist("10:01:00"), volume=1100),
    ]

    bars = bars_of(builder, ticks, "10:02:01")

    assert summary(bars) == [
        ("10:00", 800.0, 800.0, 800.0, 800.0, 1000.0, 1),
        ("10:01", 801.0, 801.0, 801.0, 801.0, 100.0, 1),
    ]


def test_a_lagged_burst_is_judged_in_the_order_it_was_received():
    lags = [("09:15:06.100", "09:15:01", 800.0, 5000), ("09:15:06.200", "09:15:02", 801.0, 5200)]
    lags.append(("09:15:06.300", "09:15:03", 802.0, 5300))
    ticks = [
        Tick("SBIN", "NSE", ltp, ist(received), ts_event_ns=ist(traded_at), volume=volume)
        for received, traded_at, ltp, volume in lags
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 802.0, 800.0, 802.0, 5300.0, 3)]


# ----- trades only: indices ---------------------------------------------------


@pytest.mark.parametrize(("volume", "event_ns"), [(None, None), (0, 0)], ids=["none", "sdk-zeros"])
def test_an_index_watched_from_the_open_gets_its_first_bar_from_its_first_in_session_value(
    volume, event_ns
):
    ticks = [
        Tick("NIFTY", "NSE_INDEX", value, ist(clock), ts_event_ns=event_ns, volume=volume)
        for clock, value in [
            ("09:14:58", 25010.0),
            ("09:15:00.500", 25010.0),
            ("09:15:20", 25010.0),
            ("09:15:40", 25020.0),
        ]
    ]

    bars = bars_of(watching_from("09:00", "NSE_INDEX"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 25010.0, 25020.0, 25010.0, 25020.0, 0.0, 3)]


def test_every_index_tick_of_a_watched_slice_is_a_print_changed_or_not():
    ticks = [
        quoted("NIFTY", "10:00:10", 25000.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "10:00:59.500", 25004.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "10:01:00.500", 25004.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "10:01:01.500", 25010.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "10:01:30", 25012.0, exchange="NSE_INDEX"),
    ]

    bars = bars_of(watching_from("09:00", "NSE_INDEX"), ticks, "10:02:01")

    assert summary(bars) == [
        ("10:00", 25000.0, 25004.0, 25000.0, 25004.0, 0.0, 2),
        ("10:01", 25004.0, 25012.0, 25004.0, 25012.0, 0.0, 3),
    ]


def test_an_index_minute_whose_value_never_changes_still_has_its_bar():
    ticks = [quoted("INDIAVIX", "10:00:10", 12.5, exchange="NSE_INDEX")]
    ticks += [
        quoted("INDIAVIX", f"10:01:{s:02d}", 12.5, exchange="NSE_INDEX") for s in range(0, 60, 5)
    ]

    bars = bars_of(watching_from("09:00", "NSE_INDEX"), ticks, "10:02:01")

    assert summary(bars) == [
        ("10:00", 12.5, 12.5, 12.5, 12.5, 0.0, 1),
        ("10:01", 12.5, 12.5, 12.5, 12.5, 0.0, 12),
    ]


def test_after_a_mid_session_start_an_index_value_in_the_partial_slice_only_seeds():
    ticks = [
        quoted("NIFTY", "11:00:40", 25000.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "11:01:05", 25000.0, exchange="NSE_INDEX"),
        quoted("NIFTY", "11:01:10", 25005.0, exchange="NSE_INDEX"),
    ]

    bars = bars_of(watching_from("11:00:30", "NSE_INDEX"), ticks, "11:02:01")

    assert summary(bars) == [("11:01", 25000.0, 25005.0, 25000.0, 25005.0, 0.0, 2)]


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


def test_a_lower_cumulative_volume_at_the_same_ltp_is_no_trade_and_adds_nothing():
    ticks = [
        traded("SBIN", "09:15:05", 800.0, 1000),
        traded("SBIN", "09:16:10", 801.0, 1800),
        traded("SBIN", "09:16:20", 801.0, 1700),
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
    ticks = [
        traded("SBIN", "11:00:35", 800.0, 500000),
        traded("SBIN", "11:00:50", 800.5, 500100),
        traded("SBIN", "11:01:10", 801.0, 500300),
    ]

    bars = bars_of(watching_from("11:00:30"), ticks, "11:02:01")

    assert summary(bars) == [("11:01", 801.0, 801.0, 801.0, 801.0, 200.0, 1)]


def test_after_a_mid_session_start_a_symbols_first_tick_only_seeds_its_count_and_ltp():
    ticks = [
        traded("SBIN", "11:00:05", 800.0, 800000),
        traded("SBIN", "11:00:20", 800.5, 800050),
    ]

    bars = bars_of(watching_from("11:00:00"), ticks, "11:01:01")

    assert summary(bars) == [("11:00", 800.5, 800.5, 800.5, 800.5, 50.0, 1)]


def test_after_a_mid_session_start_a_trade_with_the_first_count_adds_no_volume_to_its_bar():
    ticks = [
        quoted("SBIN", "11:00:40", 800.0),
        traded("SBIN", "11:01:10", 801.0, 900000),
        traded("SBIN", "11:01:30", 801.5, 900040),
    ]

    bars = bars_of(watching_from("11:00:30"), ticks, "11:02:01")

    assert summary(bars) == [("11:01", 801.0, 801.5, 801.0, 801.5, 40.0, 2)]


@pytest.mark.parametrize("price", [12.5, 12.6])
def test_after_a_mid_session_start_a_symbol_that_reported_0_counts_its_first_trade_volume(price):
    quotes = [
        Tick("OPT", "NFO", 12.5, ist(clock), ts_event_ns=0, volume=0)
        for clock in ("11:00:40", "11:05:00", "11:39:50")
    ]
    first_trade = traded("OPT", "11:40:10", price, 75, exchange="NFO")

    bars = bars_of(watching_from("11:00:30", "NFO"), [*quotes, first_trade], "11:41:01")

    assert summary(bars) == [("11:40", price, price, price, price, 75.0, 1)]


def test_after_a_resume_a_symbol_that_reported_0_counts_its_first_trade_volume():
    builder = watching_from("09:00", "NFO")
    builder.close_due(ist("10:00:01"))
    builder.resume(ist("10:00:10"), ist("10:10:00"))
    ticks = [
        Tick("OPT", "NFO", 12.5, ist("10:10:05"), ts_event_ns=0, volume=0),
        traded("OPT", "10:20:10", 12.5, 75, exchange="NFO"),
    ]

    assert summary(bars_of(builder, ticks, "10:21:01")) == [
        ("10:20", 12.5, 12.5, 12.5, 12.5, 75.0, 1)
    ]


def test_after_a_resume_a_0_from_a_symbol_that_traded_today_is_a_missing_count():
    builder = watching_from("09:00")
    builder.on_tick(traded("SBIN", "10:00:05", 800.0, 4000000))
    builder.close_due(ist("10:01:01"))
    builder.resume(ist("10:01:10"), ist("10:30:00"))
    ticks = [
        quoted("SBIN", "10:30:05", 810.0, 0),
        traded("SBIN", "10:31:05", 810.0, 4800000),
        traded("SBIN", "10:31:20", 810.5, 4800300),
    ]

    bars = bars_of(builder, ticks, "10:32:01")

    assert summary(bars) == [("10:31", 810.5, 810.5, 810.5, 810.5, 300.0, 1)]


def test_watching_from_the_session_open_makes_the_first_slice_whole():
    bars = bars_of(watching_from("09:15"), [traded("SBIN", "09:15:00.300", 800.0, 400)], "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 800.0, 800.0, 800.0, 400.0, 1)]


# ----- event time against receive time ----------------------------------------


def test_a_plausible_event_time_places_the_tick_in_the_slice_it_traded_in():
    builder = watching_from("09:00")
    builder.on_tick(
        Tick("SBIN", "NSE", 800.0, ist("09:16:00.200"), ts_event_ns=ist("09:15:59.900"), volume=10)
    )

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01")).bars] == [ist("09:15")]
    assert builder.event_time_rejected == 0


def test_a_tick_without_an_event_time_is_placed_by_its_receive_time():
    builder = watching_from("09:00")
    builder.on_tick(quoted("SBIN", "09:16:00.200", 800.0, 10))

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01")).bars] == [ist("09:16")]
    assert builder.event_time_rejected == 0


@pytest.mark.parametrize(
    ("event_clock", "slice_clock", "rejected"),
    [("09:16:01", "09:16", 0), ("09:16:01.000001", "09:15", 1)],
    ids=["five-seconds-ahead", "just-over"],
)
def test_an_event_time_up_to_five_seconds_ahead_of_receipt_is_used(
    event_clock, slice_clock, rejected
):
    builder = watching_from("09:00")
    builder.on_tick(
        Tick("SBIN", "NSE", 800.0, ist("09:15:56"), ts_event_ns=ist(event_clock), volume=1)
    )

    assert [b.ts_open_ns for b in builder.close_due(ist("09:17:01")).bars] == [ist(slice_clock)]
    assert builder.event_time_rejected == rejected


@pytest.mark.parametrize(
    "event_ns",
    [
        ist("10:00:30") // 1_000_000,
        ist("10:00:30") + (5 * 3600 + 1800) * SECOND_NS,
        ist("10:05:30"),
        ist("10:00:30") + DAY_NS,
    ],
    ids=["milliseconds", "5h30-ahead", "5-minutes-ahead", "next-day"],
)
def test_an_implausible_event_time_is_rejected_and_the_tick_placed_by_its_receive_time(event_ns):
    builder = watching_from("09:00")
    builder.on_tick(Tick("SBIN", "NSE", 800.0, ist("10:00:30"), ts_event_ns=event_ns, volume=10))

    assert [b.ts_open_ns for b in builder.close_due(ist("10:06:01")).bars] == [ist("10:00")]
    assert builder.event_time_rejected == 1


@pytest.mark.parametrize("event_clock", ["10:00:24.999", "09:15:00", "04:30:30"])
def test_an_old_trade_time_of_the_day_is_placed_by_receive_time_without_a_warning(
    event_clock, caplog
):
    builder = watching_from("09:00")
    builder.on_tick(
        Tick("ILLQ", "NSE", 50.0, ist("10:00:30"), ts_event_ns=ist(event_clock), volume=500)
    )

    assert [b.ts_open_ns for b in builder.close_due(ist("10:01:01")).bars] == [ist("10:00")]
    assert builder.event_time_rejected == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_an_event_time_of_zero_is_missing_and_not_warned(caplog):
    builder = watching_from("09:00")
    builder.on_tick(Tick("SBIN", "NSE", 800.0, ist("10:00:30"), ts_event_ns=0, volume=10))

    assert [b.ts_open_ns for b in builder.close_due(ist("10:01:01")).bars] == [ist("10:00")]
    assert builder.event_time_rejected == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_a_snapshot_of_the_previous_session_is_no_trade():
    yesterday = ist("15:29:58") - 3 * DAY_NS
    snapshot = Tick("OPT", "NFO", 12.5, ist("09:15:00.300"), ts_event_ns=yesterday, volume=250000)

    builder = watching_from("09:00", "NFO")

    assert bars_of(builder, [snapshot], "09:16:01") == []
    assert builder.event_time_rejected == 1


def test_a_snapshot_seeds_the_last_known_ltp():
    yesterday = ist("15:29:58") - DAY_NS
    ticks = [
        Tick("OPT", "NFO", 12.5, ist("09:15:00.300"), ts_event_ns=yesterday, volume=250000),
        quoted("OPT", "09:15:20", 12.5, exchange="NFO"),
        quoted("OPT", "09:15:40", 12.8, exchange="NFO"),
    ]

    bars = bars_of(watching_from("09:00", "NFO"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 12.8, 12.8, 12.8, 12.8, 0.0, 1)]


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
    ahead = 60 * SECOND_NS
    for second in range(3):
        received = ist(f"10:00:0{second}")
        builder.on_tick(Tick("SBIN", "NSE", 800.0, received, ts_event_ns=received + ahead))
    builder.on_tick(Tick("INFY", "NSE", 1500.0, ist("10:00:05"), ts_event_ns=ist("10:01:05")))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage().split(":")[1] for r in warnings] == ["SBIN", "INFY"]
    assert builder.event_time_rejected == 4


def test_a_slices_ticks_are_judged_in_receive_order_whatever_order_they_are_handed_over():
    ticks = [
        traded("SBIN", "09:15:20", 801.0, 20),
        traded("SBIN", "09:15:05", 800.0, 10),
        traded("SBIN", "09:15:50", 802.0, 40),
        traded("SBIN", "09:15:40", 799.0, 30),
    ]

    bars = bars_of(watching_from("09:00"), ticks, "09:16:01")

    assert summary(bars) == [("09:15", 800.0, 802.0, 799.0, 802.0, 40.0, 4)]
