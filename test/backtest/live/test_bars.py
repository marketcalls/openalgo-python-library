"""Bars built from WebSocket ticks and stamped at their close (08-live-trading section 5,
D27): the session grid they lie on, a recorded tick sequence and the bars it gives, a
minute without ticks, late ticks, the session edges and the day's cumulative volume.

Expected bars are written out by hand from the ticks; instants are built from IST clock
times with the standard library, not by the module under test.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from openalgo.backtest._live._types import LiveBar, Tick
from openalgo.backtest._live.bars import TickBarBuilder
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.errors import CalendarError, ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
MONDAY = dt.date(2026, 11, 16)
SUNDAY = dt.date(2026, 11, 15)
MINUTE_NS = 60 * 10**9
# How long after the trade the engine receives a tick in these recordings.
RECEIVE_DELAY_NS = 40_000_000


def ist(clock: str, day: dt.date = MONDAY) -> int:
    """UTC nanoseconds of an IST clock time such as "09:15:00.250" on `day`."""
    instant = dt.datetime.combine(day, dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


def tick(symbol: str, clock: str, ltp: float, volume: float | None = None) -> Tick:
    """A tick traded at `clock` IST and received shortly after."""
    at = ist(clock)
    return Tick(symbol, "NSE", ltp, at + RECEIVE_DELAY_NS, ts_event_ns=at, volume=volume)


def bar(
    symbol: str,
    opens: str,
    closes: str,
    ohlc: tuple[float, float, float, float],
    volume: float,
    ticks: int,
) -> LiveBar:
    """The tick-built NSE bar of `symbol` from `opens` to `closes` IST."""
    return LiveBar(symbol, "NSE", ist(opens), ist(closes), *ohlc, volume, ticks, "ticks")


def run(builder: TickBarBuilder, events: list) -> list[LiveBar]:
    """Feed ticks and clock readings (IST clock strings) in order; collect the bars."""
    bars: list[LiveBar] = []
    for event in events:
        if isinstance(event, Tick):
            bars += builder.on_tick(event)
        else:
            bars += builder.close_due(ist(event))
    return bars


@pytest.fixture
def nse() -> TickBarBuilder:
    return TickBarBuilder("NSE", MONDAY)


# ----- the grid ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("exchange", "interval_ns", "count", "first", "last"),
    [
        ("NSE", MINUTE_NS, 375, ("09:15", "09:16"), ("15:29", "15:30")),
        ("NFO", MINUTE_NS, 385, ("09:15", "09:16"), ("15:39", "15:40")),
        ("NSE", 60 * MINUTE_NS, 7, ("09:15", "10:15"), ("15:15", "15:30")),
        ("MCX", 5 * MINUTE_NS, 179, ("09:00", "09:05"), ("23:50", "23:55")),
    ],
)
def test_the_slices_run_from_the_session_open_to_the_close_on_the_interval_lattice(
    exchange, interval_ns, count, first, last
):
    builder = TickBarBuilder(exchange, MONDAY, interval_ns=interval_ns)

    slices = builder.slices

    assert len(slices) == count
    assert slices[0] == (ist(first[0]), ist(first[1]))
    assert slices[-1] == (ist(last[0]), ist(last[1]))
    assert all(
        close == next_open for (_, close), (next_open, _) in zip(slices, slices[1:], strict=False)
    )


def test_the_session_comes_from_the_default_calendar_when_none_is_given(nse):
    assert nse.session == (ist("09:15"), ist("15:30"))


def test_a_special_session_off_the_lattice_starts_with_the_slice_that_holds_its_open():
    calendar = Calendar(builtin=False)
    saturday = dt.date(2026, 11, 14)
    calendar.add_session("NSE", saturday, "18:20", "19:20", name="special session")

    builder = TickBarBuilder("NSE", saturday, interval_ns=15 * MINUTE_NS, calendar=calendar)

    opens = [ist(clock, saturday) for clock in ("18:15", "18:30", "18:45", "19:00", "19:15")]
    closes = [ist(clock, saturday) for clock in ("18:30", "18:45", "19:00", "19:15", "19:20")]
    assert builder.slices == tuple(zip(opens, closes, strict=True))
    assert builder.session == (ist("18:20", saturday), ist("19:20", saturday))


def test_a_day_without_a_session_raises_calendar_error():
    with pytest.raises(CalendarError, match="no session"):
        TickBarBuilder("NSE", SUNDAY)


@pytest.mark.parametrize("interval_ns", [0, -MINUTE_NS, 1_500_000_000, 60e9, True])
def test_an_interval_that_is_not_a_positive_whole_number_of_seconds_raises(interval_ns):
    with pytest.raises(ConfigError, match="interval_ns"):
        TickBarBuilder("NSE", MONDAY, interval_ns=interval_ns)


# ----- a recorded session -----------------------------------------------------


def test_a_recorded_tick_sequence_gives_exactly_the_expected_bars(nse):
    events = [
        tick("SBIN", "09:15:00.200", 800.0, 12000),
        tick("INFY", "09:15:01", 1500.0, 3000),
        tick("SBIN", "09:15:20", 801.5, 12500),
        tick("SBIN", "09:15:45", 799.0, 13100),
        tick("INFY", "09:15:59.900", 1502.0, 3400),
        "09:16:01",
        tick("SBIN", "09:16:02", 799.5, 13300),
        tick("INFY", "09:16:30", 1501.0, 3600),
        tick("SBIN", "09:16:40", 800.5, 13900),
        tick("INFY", "09:17:10", 1503.0, 4000),
        "09:17:11",
        tick("SBIN", "09:18:05", 802.0, 14500),
        "09:19:00",
    ]

    bars = run(nse, events)

    assert bars == [
        bar("SBIN", "09:15", "09:16", (800.0, 801.5, 799.0, 799.0), 13100.0, 3),
        bar("INFY", "09:15", "09:16", (1500.0, 1502.0, 1500.0, 1502.0), 3400.0, 2),
        bar("INFY", "09:16", "09:17", (1501.0, 1501.0, 1501.0, 1501.0), 200.0, 1),
        bar("SBIN", "09:16", "09:17", (799.5, 800.5, 799.5, 800.5), 800.0, 2),
        bar("INFY", "09:17", "09:18", (1503.0, 1503.0, 1503.0, 1503.0), 400.0, 1),
        bar("SBIN", "09:18", "09:19", (802.0, 802.0, 802.0, 802.0), 600.0, 1),
    ]
    assert (nse.late_ticks, nse.outside_ticks, nse.invalid_ticks) == (0, 0, 0)


def test_every_bar_opens_on_a_slice_of_the_grid_and_closes_with_it(nse):
    bars = run(nse, [tick("SBIN", "11:42:30", 805.0), "11:43:00"])

    assert [(b.ts_open_ns, b.ts_close_ns) for b in bars] == [(ist("11:42"), ist("11:43"))]
    assert (bars[0].ts_open_ns, bars[0].ts_close_ns) in nse.slices


def test_a_tick_in_a_later_slice_closes_the_forming_bar_of_its_symbol(nse):
    nse.on_tick(tick("SBIN", "09:15:10", 800.0))

    closed = nse.on_tick(tick("SBIN", "09:16:00", 801.0))

    assert closed == [bar("SBIN", "09:15", "09:16", (800.0,) * 4, 0.0, 1)]


def test_a_tick_closes_no_bar_of_another_symbol(nse):
    nse.on_tick(tick("INFY", "09:15:10", 1500.0))

    assert nse.on_tick(tick("SBIN", "09:16:05", 801.0)) == []


def test_the_clock_closes_a_bar_at_its_close_and_not_before(nse):
    nse.on_tick(tick("SBIN", "09:15:10", 800.0))

    assert nse.close_due(ist("09:15:59.999999")) == []
    assert nse.close_due(ist("09:16")) == [bar("SBIN", "09:15", "09:16", (800.0,) * 4, 0.0, 1)]


def test_a_minute_without_ticks_makes_no_bar(nse):
    events = [tick("SBIN", "09:15:10", 800.0), tick("SBIN", "09:17:05", 803.0), "09:18:00"]

    bars = run(nse, events)

    assert [b.ts_open_ns for b in bars] == [ist("09:15"), ist("09:17")]


def test_bars_closing_together_come_in_time_order_then_symbols_in_registration_order(nse):
    events = [
        tick("SBIN", "09:15:01", 800.0),
        tick("INFY", "09:15:02", 1500.0),
        tick("INFY", "09:16:05", 1501.0),
        tick("TCS", "09:15:30", 4000.0),
    ]
    run(nse, events)

    bars = nse.close_due(ist("09:17"))

    assert [(b.symbol, b.ts_open_ns) for b in bars] == [
        ("SBIN", ist("09:15")),
        ("TCS", ist("09:15")),
        ("INFY", ist("09:16")),
    ]


# ----- the place of a tick ----------------------------------------------------


def test_a_tick_is_placed_by_its_event_time_when_the_feed_gives_it(nse):
    traded_in_first_minute = Tick(
        "SBIN", "NSE", 800.0, ist("09:16:00.200"), ts_event_ns=ist("09:15:59.900")
    )

    nse.on_tick(traded_in_first_minute)

    assert [b.ts_open_ns for b in nse.close_due(ist("09:16:01"))] == [ist("09:15")]


def test_a_tick_without_an_event_time_is_placed_by_its_receive_time(nse):
    nse.on_tick(Tick("SBIN", "NSE", 800.0, ist("09:16:00.200")))

    assert [b.ts_open_ns for b in nse.close_due(ist("09:17"))] == [ist("09:16")]


def test_a_tick_out_of_order_inside_its_bar_sets_the_open_and_close_by_its_time(nse):
    events = [
        tick("SBIN", "09:15:20", 801.0),
        tick("SBIN", "09:15:05", 800.0),
        tick("SBIN", "09:15:50", 802.0),
        tick("SBIN", "09:15:40", 799.0),
        "09:16",
    ]

    assert run(nse, events) == [bar("SBIN", "09:15", "09:16", (800.0, 802.0, 799.0, 802.0), 0.0, 4)]


# ----- late ticks -------------------------------------------------------------


def test_a_tick_for_a_bar_its_symbol_has_closed_is_dropped_and_counted_late(nse):
    run(nse, [tick("SBIN", "09:15:10", 800.0), tick("SBIN", "09:16:01", 801.0)])

    assert nse.on_tick(tick("SBIN", "09:15:59", 790.0)) == []
    assert nse.late_ticks == 1
    assert nse.close_due(ist("09:17")) == [bar("SBIN", "09:16", "09:17", (801.0,) * 4, 0.0, 1)]


def test_a_tick_for_a_slice_the_clock_has_closed_is_late_for_a_symbol_without_a_bar_there(nse):
    nse.close_due(ist("09:16:00.500"))

    assert nse.on_tick(tick("SBIN", "09:15:59.800", 800.0)) == []
    assert nse.late_ticks == 1
    assert nse.close_due(ist("09:17")) == []


def test_a_tick_for_a_slice_before_its_symbols_forming_bar_is_late(nse):
    nse.on_tick(tick("SBIN", "09:16:05", 801.0))

    assert nse.on_tick(tick("SBIN", "09:15:50", 800.0)) == []
    assert nse.late_ticks == 1
    assert nse.close_due(ist("09:17")) == [bar("SBIN", "09:16", "09:17", (801.0,) * 4, 0.0, 1)]


# ----- the session edges ------------------------------------------------------


def test_ticks_before_the_open_and_from_the_close_on_are_ignored_and_counted(nse):
    events = [
        tick("SBIN", "09:14:59.999", 799.0),
        tick("SBIN", "09:15:00", 800.0),
        tick("SBIN", "15:29:59.999", 812.0),
        tick("SBIN", "15:30:00", 813.0),
        "15:31",
    ]

    bars = run(nse, events)

    assert bars == [
        bar("SBIN", "09:15", "09:16", (800.0,) * 4, 0.0, 1),
        bar("SBIN", "15:29", "15:30", (812.0,) * 4, 0.0, 1),
    ]
    assert nse.outside_ticks == 2


def test_the_last_bar_ends_at_the_session_close():
    hourly = TickBarBuilder("NSE", MONDAY, interval_ns=60 * MINUTE_NS)

    bars = run(hourly, [tick("SBIN", "15:20", 810.0), "15:30"])

    assert bars == [bar("SBIN", "15:15", "15:30", (810.0,) * 4, 0.0, 1)]


def test_a_tick_before_a_special_sessions_open_is_outside_though_its_slice_starts_earlier():
    calendar = Calendar(builtin=False)
    saturday = dt.date(2026, 11, 14)
    calendar.add_session("NSE", saturday, "18:20", "19:20")
    builder = TickBarBuilder("NSE", saturday, interval_ns=15 * MINUTE_NS, calendar=calendar)

    early = Tick("SBIN", "NSE", 800.0, ist("18:17", saturday))
    inside = Tick("SBIN", "NSE", 801.0, ist("18:21", saturday))
    builder.on_tick(early)
    builder.on_tick(inside)
    bars = builder.close_due(ist("18:30", saturday))

    assert [(b.ts_open_ns, b.ts_close_ns, b.open) for b in bars] == [
        (ist("18:15", saturday), ist("18:30", saturday), 801.0)
    ]
    assert builder.outside_ticks == 1


# ----- volume -----------------------------------------------------------------


def test_volume_is_the_rise_of_the_cumulative_day_volume_from_one_bar_to_the_next(nse):
    events = [
        tick("SBIN", "09:15:05", 800.0, 1000),
        tick("SBIN", "09:15:40", 800.5, 1500),
        tick("SBIN", "09:16:10", 801.0, 1700),
        tick("SBIN", "09:16:50", 801.5, 2600),
        "09:17",
    ]

    assert [b.volume for b in run(nse, events)] == [1500.0, 1100.0]


def test_a_lower_cumulative_volume_from_a_stale_tick_adds_nothing(nse):
    events = [
        tick("SBIN", "09:15:05", 800.0, 1000),
        tick("SBIN", "09:16:10", 801.0, 1800),
        tick("SBIN", "09:16:20", 801.0, 1700),
        tick("SBIN", "09:17:10", 802.0, 2000),
        "09:18",
    ]

    assert [b.volume for b in run(nse, events)] == [1000.0, 800.0, 200.0]


def test_a_symbol_whose_first_volume_is_after_the_first_slice_counts_from_that_tick(nse):
    events = [
        tick("SBIN", "10:00:05", 800.0, 50000),
        tick("SBIN", "10:00:30", 800.5, 50400),
        "10:01",
    ]

    assert [b.volume for b in run(nse, events)] == [400.0]


def test_volume_is_zero_when_the_feed_gives_none(nse):
    events = [tick("SBIN", "09:15:05", 800.0), tick("SBIN", "09:15:30", 801.0), "09:16"]

    assert [b.volume for b in run(nse, events)] == [0.0]


# ----- unusable input ---------------------------------------------------------


@pytest.mark.parametrize(
    ("ltp", "volume"),
    [
        (0.0, None),
        (-1.0, None),
        (math.nan, None),
        (math.inf, None),
        (800.0, -5.0),
        (800.0, math.nan),
    ],
)
def test_a_tick_with_an_unusable_ltp_or_volume_is_dropped_and_counted(nse, ltp, volume):
    assert nse.on_tick(tick("SBIN", "09:15:05", ltp, volume)) == []
    assert nse.invalid_ticks == 1
    assert nse.close_due(ist("09:16")) == []


def test_a_tick_of_another_exchange_raises_config_error(nse):
    with pytest.raises(ConfigError, match="its own builder"):
        nse.on_tick(Tick("SBIN", "BSE", 800.0, ist("09:15:05")))
