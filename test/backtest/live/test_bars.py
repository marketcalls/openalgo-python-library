"""Bars built from WebSocket ticks and stamped at their close (08-live-trading section 5,
D27): the session grid they lie on, a recorded tick sequence and the bars it gives,
closing by the clock with a grace, and the session edges.

Expected bars are written out by hand from the ticks; instants are built from IST clock
times with the standard library, not by the module under test. Which ticks are trades,
volume, event times and a start in the middle of the session are in
`test_bars_trades.py`.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import FrozenInstanceError

import pytest

from openalgo.backtest._live._types import LiveBar, Tick
from openalgo.backtest._live.bars import TickBarBuilder
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.errors import CalendarError, ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
MONDAY = dt.date(2026, 11, 16)
SUNDAY = dt.date(2026, 11, 15)
SATURDAY = dt.date(2026, 11, 14)
MINUTE_NS = 60 * 10**9
# How long after the trade the engine receives a tick in these recordings.
RECEIVE_DELAY_NS = 40_000_000


def ist(clock: str, day: dt.date = MONDAY) -> int:
    """UTC nanoseconds of an IST clock time such as "09:15:00.250" on `day`."""
    instant = dt.datetime.combine(day, dt.time.fromisoformat(clock), tzinfo=IST)
    return (instant - EPOCH) // dt.timedelta(microseconds=1) * 1000


def tick(
    symbol: str,
    clock: str,
    ltp: float,
    volume: float | None = None,
    *,
    received: str | None = None,
) -> Tick:
    """A tick traded at `clock` IST and received shortly after (or at `received`)."""
    at = ist(clock)
    recv = at + RECEIVE_DELAY_NS if received is None else ist(received)
    return Tick(symbol, "NSE", ltp, recv, ts_event_ns=at, volume=volume)


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
            builder.on_tick(event)
        else:
            bars += builder.close_due(ist(event)).bars
    return bars


def due(builder: TickBarBuilder, clock: str, day: dt.date = MONDAY) -> list[LiveBar]:
    """The bars `close_due` gives out at `clock` IST."""
    return list(builder.close_due(ist(clock, day)).bars)


def builder_at(day: dt.date = MONDAY, **options) -> TickBarBuilder:
    """A builder for NSE watching from 09:00 IST, before the open."""
    return TickBarBuilder("NSE", day, watching_from_ns=ist("09:00", day), **options)


@pytest.fixture
def nse() -> TickBarBuilder:
    return builder_at()


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
    builder = TickBarBuilder(
        exchange, MONDAY, watching_from_ns=ist("08:00"), interval_ns=interval_ns
    )

    slices = builder.slices

    assert len(slices) == count
    assert slices[0] == (ist(first[0]), ist(first[1]))
    assert slices[-1] == (ist(last[0]), ist(last[1]))
    pairs = zip(slices, slices[1:], strict=False)
    assert all(close == next_open for (_, close), (next_open, _) in pairs)


def test_the_session_comes_from_the_default_calendar_when_none_is_given(nse):
    assert nse.session == (ist("09:15"), ist("15:30"))


def test_a_special_session_off_the_lattice_starts_with_the_slice_that_holds_its_open():
    calendar = Calendar(builtin=False)
    calendar.add_session("NSE", SATURDAY, "18:20", "19:20", name="special session")

    builder = builder_at(SATURDAY, interval_ns=15 * MINUTE_NS, calendar=calendar)

    opens = [ist(clock, SATURDAY) for clock in ("18:15", "18:30", "18:45", "19:00", "19:15")]
    closes = [ist(clock, SATURDAY) for clock in ("18:30", "18:45", "19:00", "19:15", "19:20")]
    assert builder.slices == tuple(zip(opens, closes, strict=True))
    assert builder.session == (ist("18:20", SATURDAY), ist("19:20", SATURDAY))


def test_the_muhurat_session_of_the_built_in_calendar_starts_with_the_slice_holding_its_open():
    muhurat = dt.date(2026, 11, 8)

    builder = TickBarBuilder(
        "NSE", muhurat, watching_from_ns=ist("14:00", muhurat), interval_ns=15 * MINUTE_NS
    )

    assert builder.session == (ist("14:20", muhurat), ist("15:30", muhurat))
    opens = [ist(clock, muhurat) for clock in ("14:15", "14:30", "14:45", "15:00", "15:15")]
    closes = [ist(clock, muhurat) for clock in ("14:30", "14:45", "15:00", "15:15", "15:30")]
    assert builder.slices == tuple(zip(opens, closes, strict=True))
    builder.on_tick(Tick("SBIN", "NSE", 800.0, ist("14:21", muhurat), volume=50))
    assert [(b.ts_open_ns, b.ts_close_ns) for b in due(builder, "14:31", muhurat)] == [
        (ist("14:15", muhurat), ist("14:30", muhurat))
    ]


def test_an_mcx_evening_bar_lies_on_the_mcx_session():
    mcx = TickBarBuilder("MCX", MONDAY, watching_from_ns=ist("08:30"))
    crude = [
        Tick("CRUDEOIL", "MCX", 5400.0, ist("23:54:30"), volume=900),
        Tick("CRUDEOIL", "MCX", 5401.0, ist("23:55:00"), volume=950),
    ]
    for one in crude:
        mcx.on_tick(one)

    bars = list(mcx.close_due(ist("23:56")).bars)

    expected = LiveBar(
        "CRUDEOIL", "MCX", ist("23:54"), ist("23:55"), *(5400.0,) * 4, 900.0, 1, "ticks"
    )
    assert bars == [expected]
    assert mcx.outside_ticks == 1


def test_a_day_without_a_session_raises_calendar_error():
    with pytest.raises(CalendarError, match="no session"):
        builder_at(SUNDAY)


@pytest.mark.parametrize(
    "interval_ns",
    [0, -MINUTE_NS, 1_500_000_000, 60e9, True, "1m", 24 * 60 * MINUTE_NS, 48 * 60 * MINUTE_NS],
)
def test_an_interval_that_is_not_whole_seconds_shorter_than_a_day_raises_config_error(
    interval_ns,
):
    with pytest.raises(ConfigError, match="interval_ns"):
        builder_at(interval_ns=interval_ns)


@pytest.mark.parametrize("watching_from_ns", [None, 1.7e18, "09:00", True])
def test_watching_from_must_be_an_instant_in_nanoseconds(watching_from_ns):
    with pytest.raises(ConfigError, match="watching_from_ns"):
        TickBarBuilder("NSE", MONDAY, watching_from_ns=watching_from_ns)


@pytest.mark.parametrize("grace", [-0.5, math.nan, math.inf, "1"])
def test_the_close_grace_must_be_zero_or_more_seconds(grace):
    with pytest.raises(ConfigError, match="close_grace_s"):
        builder_at(close_grace_s=grace)


# ----- a recorded session -----------------------------------------------------


def test_a_recorded_tick_sequence_gives_exactly_the_expected_bars(nse):
    events = [
        tick("SBIN", "09:15:00.200", 800.0, 12000),
        tick("INFY", "09:15:01", 1500.0, 3000),
        tick("SBIN", "09:15:20", 801.5, 12500),
        tick("SBIN", "09:15:30", 801.5, 12500),
        tick("SBIN", "09:15:45", 799.0, 13100),
        tick("INFY", "09:15:59.900", 1502.0, 3400),
        "09:16:00.500",
        tick("SBIN", "09:16:00.300", 799.5, 13300),
        tick("INFY", "09:15:59.950", 1502.5, 3450, received="09:16:00.400"),
        "09:16:01",
        tick("INFY", "09:16:30", 1501.0, 3600),
        tick("SBIN", "09:16:40", 800.5, 13900),
        tick("INFY", "09:17:10", 1503.0, 4000),
        "09:17:11",
        tick("SBIN", "09:17:30", 800.5, 13900),
        tick("SBIN", "09:18:05", 802.0, 14500),
        "09:19:01",
    ]

    bars = run(nse, events)

    assert bars == [
        bar("SBIN", "09:15", "09:16", (800.0, 801.5, 799.0, 799.0), 13100.0, 3),
        bar("INFY", "09:15", "09:16", (1500.0, 1502.5, 1500.0, 1502.5), 3450.0, 3),
        bar("SBIN", "09:16", "09:17", (799.5, 800.5, 799.5, 800.5), 800.0, 2),
        bar("INFY", "09:16", "09:17", (1501.0, 1501.0, 1501.0, 1501.0), 150.0, 1),
        bar("INFY", "09:17", "09:18", (1503.0, 1503.0, 1503.0, 1503.0), 400.0, 1),
        bar("SBIN", "09:18", "09:19", (802.0, 802.0, 802.0, 802.0), 600.0, 1),
    ]
    assert (nse.late_ticks, nse.outside_ticks, nse.invalid_ticks) == (0, 0, 0)


def test_every_bar_opens_on_a_slice_of_the_grid_and_closes_with_it(nse):
    bars = run(nse, [tick("SBIN", "11:42:30", 805.0, 100), "11:43:01"])

    assert [(b.ts_open_ns, b.ts_close_ns) for b in bars] == [(ist("11:42"), ist("11:43"))]
    assert (bars[0].ts_open_ns, bars[0].ts_close_ns) in nse.slices


def test_a_minute_without_ticks_makes_no_bar(nse):
    events = [tick("SBIN", "09:15:10", 800.0, 10), tick("SBIN", "09:17:05", 803.0, 20), "09:18:01"]

    bars = run(nse, events)

    assert [b.ts_open_ns for b in bars] == [ist("09:15"), ist("09:17")]


def test_bars_given_out_together_come_in_time_order_then_symbols_in_registration_order(nse):
    events = [
        tick("SBIN", "09:15:01", 800.0, 10),
        tick("INFY", "09:16:05", 1501.0, 10),
        tick("TCS", "09:15:30", 4000.0, 10),
    ]
    run(nse, events)

    bars = due(nse, "09:17:01")

    assert [(b.symbol, b.ts_open_ns) for b in bars] == [
        ("SBIN", ist("09:15")),
        ("TCS", ist("09:15")),
        ("INFY", ist("09:16")),
    ]


# ----- closing by the clock ---------------------------------------------------


def test_a_bar_is_given_out_once_the_clock_passes_its_close_plus_the_grace(nse):
    nse.on_tick(tick("SBIN", "09:15:10", 800.0, 10))

    assert due(nse, "09:16:00.999999") == []
    assert due(nse, "09:16:01") == [bar("SBIN", "09:15", "09:16", (800.0,) * 4, 10.0, 1)]


@pytest.mark.parametrize(
    ("grace", "early", "closes_at"), [(0, None, "09:16"), (2.5, "09:16:02.4", "09:16:02.5")]
)
def test_the_grace_is_settable(grace, early, closes_at):
    builder = builder_at(close_grace_s=grace)
    builder.on_tick(tick("SBIN", "09:15:10", 800.0, 10))

    if early is not None:
        assert due(builder, early) == []
    assert [b.ts_open_ns for b in due(builder, closes_at)] == [ist("09:15")]


def test_close_due_tells_which_slices_it_closed_with_or_without_bars(nse):
    nse.on_tick(tick("SBIN", "09:16:10", 800.0, 10))

    first = nse.close_due(ist("09:17:01"))
    second = nse.close_due(ist("09:17:30"))
    third = nse.close_due(ist("09:18:01"))

    assert first.slices == ((ist("09:15"), ist("09:16")), (ist("09:16"), ist("09:17")))
    assert [b.ts_open_ns for b in first.bars] == [ist("09:16")]
    assert (second.slices, second.bars) == ((), ())
    assert (third.slices, third.bars) == (((ist("09:17"), ist("09:18")),), ())


def test_closed_slices_are_frozen(nse):
    with pytest.raises(FrozenInstanceError):
        nse.close_due(ist("09:16:01")).bars = ()  # type: ignore[misc]


def test_a_tick_of_a_later_slice_does_not_close_the_earlier_bar(nse):
    nse.on_tick(tick("SBIN", "09:15:10", 800.0, 10))

    assert nse.on_tick(tick("SBIN", "09:16:00.100", 801.0, 20)) is None
    assert due(nse, "09:16:00.500") == []


def test_a_tick_arriving_after_a_later_one_still_counts_in_its_slice_within_the_grace(nse):
    events = [
        tick("SBIN", "10:00:30", 800.0, 100),
        tick("SBIN", "10:01:00.010", 805.0, 130, received="10:01:00.050"),
        tick("SBIN", "10:00:59.990", 803.0, 120, received="10:01:00.060"),
        "10:02:01",
    ]

    assert run(nse, events) == [
        bar("SBIN", "10:00", "10:01", (800.0, 803.0, 800.0, 803.0), 120.0, 2),
        bar("SBIN", "10:01", "10:02", (805.0,) * 4, 10.0, 1),
    ]
    assert nse.late_ticks == 0


def test_the_bars_do_not_depend_on_the_order_the_ticks_arrived_in():
    ticks = [
        tick("SBIN", "09:15:05", 800.0, 100),
        tick("SBIN", "09:15:40", 802.0, 180),
        tick("SBIN", "09:15:59.900", 801.0, 230),
        tick("SBIN", "09:16:00.200", 799.0, 260),
        tick("SBIN", "09:16:20", 799.0, 260),
        tick("SBIN", "09:16:30", 800.5, 300),
    ]
    in_order, shuffled = builder_at(), builder_at()

    bars_in_order = run(in_order, [*ticks, "09:17:01"])
    bars_shuffled = run(shuffled, [ticks[i] for i in (3, 5, 0, 2, 4, 1)] + ["09:17:01"])

    assert bars_shuffled == bars_in_order


# A trade at 09:15:59.990 received 1.06 s later, after its slice's close plus the grace.
IN_FLIGHT = [
    tick("SBIN", "09:15:30", 800.0, 1000),
    tick("SBIN", "09:15:59.990", 803.0, 1200, received="09:16:01.050"),
    tick("SBIN", "09:16:20", 801.0, 1300),
]
IN_FLIGHT_BARS = [
    bar("SBIN", "09:15", "09:16", (800.0,) * 4, 1000.0, 1),
    bar("SBIN", "09:16", "09:17", (801.0,) * 4, 300.0, 1),
]


@pytest.mark.parametrize(
    "events",
    [
        [IN_FLIGHT[0], IN_FLIGHT[1], "09:16:01.100", IN_FLIGHT[2], "09:17:01"],
        [IN_FLIGHT[0], "09:16:01.100", IN_FLIGHT[1], IN_FLIGHT[2], "09:17:01"],
    ],
    ids=["ticks-handed-over-first", "close-due-called-first"],
)
def test_a_tick_is_late_by_its_receive_time_whenever_close_due_runs(events):
    builder = builder_at()

    assert run(builder, events) == IN_FLIGHT_BARS
    assert builder.late_ticks == 1


def test_a_replay_driven_by_receive_times_gives_the_bars_of_the_live_session():
    replay, replayed = builder_at(), []
    for recorded in IN_FLIGHT:
        replayed += replay.close_due(recorded.ts_recv_ns).bars
        replay.on_tick(recorded)
    replayed += due(replay, "09:17:01")

    assert replayed == IN_FLIGHT_BARS
    assert replay.late_ticks == 1


def test_a_late_tick_that_was_no_trade_is_not_counted(nse):
    nse.on_tick(tick("SBIN", "09:15:30", 800.0, 1000))
    due(nse, "09:16:01")

    for received in ("09:16:01.500", "09:16:02", "09:16:03"):
        nse.on_tick(tick("SBIN", "09:15:59.500", 800.0, 1000, received=received))

    assert nse.late_ticks == 0


def test_a_late_trade_is_counted_and_warned_once_per_symbol(nse, caplog):
    for minute in (15, 16, 17):
        clock = f"09:{minute}:58.700"
        received = f"09:{minute + 1}:01.240"
        nse.on_tick(tick("SBIN", f"09:{minute}:10", 800.0 + minute, minute * 100))
        nse.on_tick(tick("SBIN", clock, 810.0 + minute, minute * 100 + 58, received=received))

    bars = due(nse, "09:18:10")

    assert [b.close for b in bars] == [815.0, 816.0, 817.0]
    assert nse.late_ticks == 3
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "clock may be ahead" in warned[0].getMessage()


# ----- a feed outage ----------------------------------------------------------


def test_after_the_feed_resumes_its_slice_only_seeds_so_the_gap_volume_lands_on_no_bar(nse):
    nse.on_tick(tick("SBIN", "09:16:10", 800.0, 200000))
    nse.on_tick(tick("SBIN", "09:16:20", 801.0, 201000))
    due(nse, "09:17:01")

    nse.resume(ist("10:30:00.200"))
    events = [
        tick("SBIN", "10:30:00.100", 812.0, 4200000, received="10:30:00.300"),
        tick("SBIN", "10:30:40", 812.5, 4203000),
        tick("SBIN", "10:31:05", 813.0, 4203500),
        "10:32:01",
    ]

    assert run(nse, events) == [bar("SBIN", "10:31", "10:32", (813.0,) * 4, 500.0, 1)]


def test_after_the_feed_resumes_a_first_count_only_sets_the_level(nse):
    nse.on_tick(tick("SBIN", "09:16:10", 800.0, 200000))
    due(nse, "09:17:01")

    nse.resume(ist("10:30:00"))
    events = [tick("SBIN", "10:31:05", 813.0, 4200000), tick("SBIN", "10:31:20", 813.5, 4200300)]

    assert run(nse, [*events, "10:32:01"]) == [
        bar("SBIN", "10:31", "10:32", (813.5,) * 4, 300.0, 1)
    ]


def test_a_resume_before_the_open_changes_nothing(nse):
    nse.resume(ist("09:10"))

    assert run(nse, [tick("SBIN", "09:15:05", 800.0, 400), "09:16:01"]) == [
        bar("SBIN", "09:15", "09:16", (800.0,) * 4, 400.0, 1)
    ]


def test_a_tick_for_a_slice_the_clock_has_closed_is_dropped_and_counted_late(nse):
    nse.on_tick(tick("SBIN", "09:15:10", 800.0, 10))
    due(nse, "09:16:01")

    nse.on_tick(tick("SBIN", "09:15:59", 790.0, 20, received="09:16:01.200"))

    assert nse.late_ticks == 1
    assert due(nse, "09:17:01") == []


# ----- the session edges ------------------------------------------------------


def test_ticks_before_the_open_and_from_the_close_on_are_ignored_and_counted(nse):
    events = [
        tick("SBIN", "09:14:59.999", 799.0, 5),
        tick("SBIN", "09:15:00", 800.0, 10),
        tick("SBIN", "15:29:59.999", 812.0, 20),
        tick("SBIN", "15:30:00", 813.0, 30),
        "15:31",
    ]

    bars = run(nse, events)

    assert bars == [
        bar("SBIN", "09:15", "09:16", (800.0,) * 4, 10.0, 1),
        bar("SBIN", "15:29", "15:30", (812.0,) * 4, 10.0, 1),
    ]
    assert nse.outside_ticks == 2


def test_ticks_outside_the_session_are_logged_at_debug_once_per_symbol(nse, caplog):
    caplog.set_level(logging.DEBUG, logger="openalgo.backtest.live")

    for clock in ("09:10", "09:11", "09:12"):
        nse.on_tick(tick("SBIN", clock, 800.0, 10))
    nse.on_tick(tick("INFY", "09:10", 1500.0, 10))

    outside = [r for r in caplog.records if "outside the session" in r.getMessage()]
    assert [r.levelno for r in outside] == [logging.DEBUG, logging.DEBUG]
    assert nse.outside_ticks == 4


def test_the_last_bar_ends_at_the_session_close():
    hourly = builder_at(interval_ns=60 * MINUTE_NS)

    bars = run(hourly, [tick("SBIN", "15:20", 810.0, 10), "15:31"])

    assert bars == [bar("SBIN", "15:15", "15:30", (810.0,) * 4, 10.0, 1)]


def test_a_tick_before_a_special_sessions_open_is_outside_though_its_slice_starts_earlier():
    calendar = Calendar(builtin=False)
    calendar.add_session("NSE", SATURDAY, "18:20", "19:20")
    builder = TickBarBuilder(
        "NSE",
        SATURDAY,
        watching_from_ns=ist("18:00", SATURDAY),
        interval_ns=15 * MINUTE_NS,
        calendar=calendar,
    )

    builder.on_tick(Tick("SBIN", "NSE", 800.0, ist("18:17", SATURDAY), volume=10))
    builder.on_tick(Tick("SBIN", "NSE", 801.0, ist("18:21", SATURDAY), volume=20))
    bars = due(builder, "18:31", SATURDAY)

    assert [(b.ts_open_ns, b.ts_close_ns, b.open, b.volume) for b in bars] == [
        (ist("18:15", SATURDAY), ist("18:30", SATURDAY), 801.0, 20.0)
    ]
    assert builder.outside_ticks == 1


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
    nse.on_tick(tick("SBIN", "09:15:05", ltp, volume))

    assert nse.invalid_ticks == 1
    assert due(nse, "09:16:01") == []


def test_a_tick_of_another_exchange_raises_config_error(nse):
    with pytest.raises(ConfigError, match="its own builder"):
        nse.on_tick(Tick("SBIN", "BSE", 800.0, ist("09:15:05")))
