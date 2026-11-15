"""The tick recorder (FR-LIVE-19, recorder half): a recording connects the client's feed,
subscribes, writes every tick to the cache until the stop (the day's close, a given
instant, or Ctrl+C), keeps the feed's outages and its own span, then unsubscribes,
disconnects and finalizes. On manual clocks with a fake feed that a script plays; no
socket is opened.
"""

from __future__ import annotations

import threading

import pytest
from recorder_fakes import (
    DAY,
    INFY,
    NS,
    SBIN,
    Clocks,
    FakeFeed,
    Script,
    interrupt,
    ist_ns,
    quote,
    raw_files,
)

from openalgo.backtest._live._tick_files import read_raw
from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.recorder import (
    Coverage,
    Ending,
    Outage,
    Span,
    Stop,
    finalize,
    read_day,
    read_merged_ticks,
    read_ticks,
    record,
)
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import CalendarError, ConfigError, DataError, LiveError

START = "10:00:00"
STOP = "10:01:00"


def recording(*, start: str = START, **feed_options):
    """The clocks at `start`, a fake feed and the script that plays it."""
    clocks = Clocks(ist_ns(start))
    return clocks, FakeFeed(**feed_options), Script(clocks)


def run(tmp_path, clocks, feed, script, instruments=(SBIN,), **options):
    options.setdefault("stop_at_ns", ist_ns(STOP))
    return record(
        feed,
        list(instruments),
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        cache_dir=tmp_path,
        **options,
    )


def push_at(script: Script, feed: FakeFeed, clock: str, message: dict) -> None:
    script.at(ist_ns(clock), lambda: feed.push(message))


def ltps(tmp_path, symbol: str = "SBIN") -> list[float]:
    return [tick.ltp for tick in read_ticks("NSE", symbol, DAY, cache_dir=tmp_path)]


# ----- recording and reading back ---------------------------------------------------


def test_a_recording_reads_back_every_tick_of_a_symbol_in_receive_order(tmp_path):
    clocks, feed, script = recording()
    trade_s = ist_ns("10:00:00.500") // NS
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4, volume=1_000, ltt=trade_s))
    push_at(script, feed, "10:00:02", quote("INFY", 1_500.0, volume=40))
    push_at(script, feed, "10:00:03", quote("SBIN", 812.5, volume=1_100))

    run(tmp_path, clocks, feed, script, (SBIN, INFY))

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == [
        Tick("SBIN", "NSE", 812.4, ist_ns("10:00:01"), trade_s * NS, 1_000.0),
        Tick("SBIN", "NSE", 812.5, ist_ns("10:00:03"), None, 1_100.0),
    ]


def test_the_summary_counts_the_ticks_written_of_every_instrument_asked_for(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    push_at(script, feed, "10:00:02", quote("SBIN", 812.5))

    summary = run(tmp_path, clocks, feed, script, (SBIN, INFY))

    assert dict(summary.ticks) == {("NSE", "SBIN"): 2, ("NSE", "INFY"): 0}


def test_an_instrument_without_a_tick_reads_back_as_a_recorded_day_with_none(tmp_path):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script, (SBIN, INFY))

    assert read_ticks("NSE", "INFY", DAY, cache_dir=tmp_path) == []


def test_the_summary_lists_the_day_file_of_each_instrument(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    summary = run(tmp_path, clocks, feed, script, (SBIN, INFY))

    written = {(file.exchange, file.symbol, file.ticks) for file in summary.files}
    assert written == {("NSE", "SBIN", 1), ("NSE", "INFY", 0)}


def test_the_receive_time_is_when_the_writer_takes_a_tick_and_the_queued_time_when_it_came(
    tmp_path,
):
    clocks, feed, script = recording()

    def tick_then_busy_writer():
        feed.push(quote("SBIN", 812.4))
        clocks.advance(NS // 4)  # the writer takes the tick a quarter second later

    script.at(ist_ns("10:00:01"), tick_then_busy_writer)

    run(tmp_path, clocks, feed, script)

    records = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records
    assert (int(records["ts_queued_ns"][0]), int(records["ts_recv_ns"][0])) == (
        ist_ns("10:00:01"),
        ist_ns("10:00:01.250"),
    )


def test_a_tick_is_in_its_raw_file_within_a_second_of_its_receipt(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:00.200", quote("SBIN", 812.4))
    seen = []
    script.at(
        ist_ns("10:00:01.200"),
        lambda: seen.append(sum(len(read_raw(path).records) for path in raw_files(tmp_path))),
    )

    run(tmp_path, clocks, feed, script)

    assert seen == [1]


def test_finalizing_leaves_no_raw_file(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    run(tmp_path, clocks, feed, script)

    assert raw_files(tmp_path) == []


def test_messages_that_make_no_tick_are_counted_as_dropped(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 0))
    push_at(script, feed, "10:00:02", quote("SBIN", 812.4))

    summary = run(tmp_path, clocks, feed, script)

    assert (summary.dropped, summary.ticks[("NSE", "SBIN")], summary.lost) == (1, 1, 0)


def test_a_tick_with_an_implausible_trade_time_is_kept_without_it_and_counted(tmp_path):
    clocks, feed, script = recording()
    counted_from_1980 = ist_ns("10:00:00") // NS - 315_532_800
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4, ltt=counted_from_1980))

    summary = run(tmp_path, clocks, feed, script)

    (tick,) = read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)
    assert (tick.ts_event_ns, summary.event_times_rejected) == (None, 1)


# ----- the order across symbols -------------------------------------------------------


def test_every_record_carries_a_sequence_number_increasing_across_symbols(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    push_at(script, feed, "10:00:02", quote("INFY", 1_500.0))
    push_at(script, feed, "10:00:03", quote("SBIN", 812.5))

    run(tmp_path, clocks, feed, script, (SBIN, INFY))

    sbin = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records["seq"].tolist()
    infy = read_day("NSE", "INFY", DAY, cache_dir=tmp_path).records["seq"].tolist()
    assert sbin[0] < infy[0] < sbin[1]


def test_the_merged_read_orders_ticks_of_one_instant_as_they_were_taken(tmp_path):
    clocks, feed, script = recording()

    def two_at_once():
        feed.push(quote("INFY", 1_500.0))
        feed.push(quote("SBIN", 812.4))

    script.at(ist_ns("10:00:01"), two_at_once)

    run(tmp_path, clocks, feed, script, (SBIN, INFY))

    merged = read_merged_ticks([SBIN, INFY], DAY, cache_dir=tmp_path)
    assert [tick.symbol for tick in merged] == ["INFY", "SBIN"]


def test_the_merged_read_keeps_the_receive_order_when_the_wall_clock_steps_back(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    script.at(ist_ns("10:00:01.500"), lambda: clocks.step_wall(-600 * NS))
    push_at(script, feed, "10:00:02", quote("INFY", 1_500.0))  # stamped 09:50:02
    script.at(ist_ns("10:00:03"), interrupt)

    run(tmp_path, clocks, feed, script, (SBIN, INFY))

    merged = read_merged_ticks([INFY, SBIN], DAY, cache_dir=tmp_path)
    assert [(tick.symbol, tick.ts_recv_ns) for tick in merged] == [
        ("SBIN", ist_ns("10:00:01")),
        ("INFY", ist_ns("09:50:02")),
    ]


def test_the_merged_read_puts_the_recordings_of_a_day_in_the_order_of_their_start(tmp_path):
    # The 12:00 recording is written first, the 10:00 one after it.
    for start, first, second in (("12:00:00", "INFY", "SBIN"), ("10:00:00", "SBIN", "INFY")):
        clocks, feed, script = recording(start=start)
        script.at(clocks.mono_ns + NS, lambda f=feed, s=first: f.push(quote(s, 1.0)))
        script.at(clocks.mono_ns + 2 * NS, lambda f=feed, s=second: f.push(quote(s, 2.0)))
        run(tmp_path, clocks, feed, script, (SBIN, INFY), stop_at_ns=clocks.wall_ns + 10 * NS)

    merged = read_merged_ticks([SBIN, INFY], DAY, cache_dir=tmp_path)
    assert [tick.symbol for tick in merged] == ["SBIN", "INFY", "INFY", "SBIN"]


# ----- stopping -----------------------------------------------------------------------


def test_a_recording_stops_at_the_days_close_by_default(tmp_path):
    clocks, feed, script = recording(start="15:29:50")
    push_at(script, feed, "15:29:55", quote("SBIN", 812.4))

    summary = run(tmp_path, clocks, feed, script, stop_at_ns=None)

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.CLOSE, ist_ns("15:30:00"))


def test_the_ticks_before_the_close_are_finalized_when_it_stops_there(tmp_path):
    clocks, feed, script = recording(start="15:29:50")
    push_at(script, feed, "15:29:55", quote("SBIN", 812.4))

    run(tmp_path, clocks, feed, script, stop_at_ns=None)

    assert ltps(tmp_path) == [812.4]


def test_the_default_stop_is_the_latest_close_of_the_instruments_exchanges(tmp_path):
    clocks, feed, script = recording(start="23:54:58")
    crude = {"exchange": "MCX", "symbol": "CRUDEOIL25NOVFUT"}

    summary = run(tmp_path, clocks, feed, script, (SBIN, crude), stop_at_ns=None)

    assert summary.stopped_ns == default_calendar().session("MCX", DAY)[1]


def test_a_recording_stops_at_the_instant_given(tmp_path):
    clocks, feed, script = recording()

    summary = run(tmp_path, clocks, feed, script, stop_at_ns=ist_ns("10:00:30"))

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.STOP_AT, ist_ns("10:00:30"))


def test_ctrl_c_stops_the_recording(tmp_path):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:05"), interrupt)

    summary = run(tmp_path, clocks, feed, script)

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.INTERRUPT, ist_ns("10:00:05"))


def test_ctrl_c_still_finalizes_the_ticks_recorded_before_it(tmp_path):
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    script.at(ist_ns("10:00:01.500"), interrupt)

    run(tmp_path, clocks, feed, script)

    assert ltps(tmp_path) == [812.4]


def test_stopping_unsubscribes_then_disconnects(tmp_path):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script)

    assert feed.calls[:4] == ["connect", "subscribe_quote", "unsubscribe_quote", "disconnect"]


def test_the_recording_subscribes_and_unsubscribes_the_instruments_asked_for(tmp_path):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script, (SBIN, INFY, SBIN))

    assert feed.requests["subscribe_quote"] == feed.requests["unsubscribe_quote"] == [SBIN, INFY]


def test_stopping_removes_the_recordings_hooks_from_the_client(tmp_path):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script)

    assert feed.hooks == {"disconnect": [], "reconnect": []}


@pytest.mark.parametrize("mode", ["ltp", "quote", "depth"])
def test_the_recording_subscribes_in_the_mode_asked_for(tmp_path, mode):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script, mode=mode)

    assert feed.calls[1] == f"subscribe_{mode}"


# ----- outages and spans --------------------------------------------------------------


def test_an_outage_runs_from_the_disconnect_to_the_reconnect(tmp_path):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:10"), feed.drop)
    script.at(ist_ns("10:00:25"), feed.restore)

    summary = run(tmp_path, clocks, feed, script)

    assert summary.outages == (Outage(ist_ns("10:00:10"), ist_ns("10:00:25")),)


def test_the_recordings_span_and_outages_are_kept_with_the_day(tmp_path):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:10"), feed.drop)
    script.at(ist_ns("10:00:25"), feed.restore)

    summary = run(tmp_path, clocks, feed, script)

    outage = Outage(ist_ns("10:00:10"), ist_ns("10:00:25"))
    span = Span(summary.recording, ist_ns(START), ist_ns(STOP), Ending.STOP, (outage,))
    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans == (span,)


def test_an_outage_still_under_way_ends_with_the_recording(tmp_path):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:50"), feed.drop)

    summary = run(tmp_path, clocks, feed, script)

    assert summary.outages == (Outage(ist_ns("10:00:50"), ist_ns(STOP)),)


def test_the_disconnect_the_recording_asks_for_is_no_outage(tmp_path):
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages == ()


@pytest.mark.parametrize(
    "clock, coverage",
    [
        pytest.param("10:00:30", Coverage.RECORDED, id="inside the first recording"),
        pytest.param("10:00:12", Coverage.OUTAGE, id="in its outage"),
        pytest.param("11:00:00", Coverage.NOT_RECORDED, id="between the recordings"),
        pytest.param("12:00:30", Coverage.RECORDED, id="inside the second"),
        pytest.param("09:00:00", Coverage.NOT_RECORDED, id="before both"),
    ],
)
def test_time_outside_every_recording_is_not_recorded_and_no_outage(tmp_path, clock, coverage):
    for start in ("10:00:00", "12:00:00"):
        clocks, feed, script = recording(start=start)
        if start == "10:00:00":
            script.at(ist_ns("10:00:10"), feed.drop)
            script.at(ist_ns("10:00:15"), feed.restore)
        run(tmp_path, clocks, feed, script, stop_at_ns=clocks.wall_ns + 60 * NS)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).coverage(ist_ns(clock)) is coverage


# ----- the feed's thread never waits --------------------------------------------------


def test_the_data_callback_never_blocks_on_a_full_queue_and_counts_what_it_lost(tmp_path):
    clocks, feed, script = recording()
    finished = []

    def burst_while_the_writer_waits():
        # The writer is inside its wait, so it takes nothing off the queue meanwhile.
        pusher = threading.Thread(
            target=lambda: [feed.push(quote("SBIN", 812.0 + n)) for n in range(5)]
        )
        pusher.start()
        pusher.join(timeout=5)
        finished.append(not pusher.is_alive())

    script.at(ist_ns("10:00:01"), burst_while_the_writer_waits)

    summary = run(tmp_path, clocks, feed, script, queue_size=2)

    assert (finished, summary.overflowed, summary.ticks[("NSE", "SBIN")]) == ([True], 3, 2)


# ----- what a recording refuses -------------------------------------------------------


def test_a_feed_that_does_not_connect_raises_live_error(tmp_path):
    clocks, feed, script = recording(connects=False)

    with pytest.raises(LiveError, match="did not connect"):
        run(tmp_path, clocks, feed, script)


def test_a_feed_that_does_not_connect_leaves_no_tick_file(tmp_path):
    clocks, feed, script = recording(connects=False)

    with pytest.raises(LiveError):
        run(tmp_path, clocks, feed, script)

    assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == ["recorder.lock"]


@pytest.mark.parametrize(
    "instruments, options, message",
    [
        pytest.param([], {}, "instruments is empty", id="no instrument"),
        pytest.param([{"symbol": "SBIN"}], {}, "each instrument", id="no exchange"),
        pytest.param(SBIN, {}, "a list of", id="one instrument not in a list"),
        pytest.param([SBIN], {"mode": "full"}, "mode must be one of", id="unknown mode"),
        pytest.param([SBIN], {"queue_size": 0}, "queue_size", id="an empty queue"),
        pytest.param([SBIN], {"stop_at_ns": 1.5e18}, "stop_at_ns is an instant", id="a float"),
        pytest.param(
            [SBIN], {"stop_at_ns": ist_ns(START)}, "not after the start", id="stop at the start"
        ),
    ],
)
def test_an_argument_that_cannot_be_used_raises_config_error(
    tmp_path, instruments, options, message
):
    clocks, feed, script = recording()
    options.setdefault("stop_at_ns", ist_ns(STOP))

    with pytest.raises(ConfigError, match=message):
        record(
            feed,
            instruments,
            now=clocks.wall,
            monotonic=clocks.mono,
            wait=script,
            cache_dir=tmp_path,
            **options,
        )


def test_a_client_without_the_feed_calls_raises_config_error(tmp_path):
    clocks = Clocks(ist_ns(START))

    with pytest.raises(ConfigError, match="subscribe_quote"):
        record(
            object(),
            [SBIN],
            now=clocks.wall,
            monotonic=clocks.mono,
            cache_dir=tmp_path,
            stop_at_ns=ist_ns(STOP),
        )


def test_the_default_stop_on_a_day_without_a_session_raises_calendar_error(tmp_path):
    clocks = Clocks(ist_ns("10:00:00", DAY.replace(day=15)))  # a Sunday

    with pytest.raises(CalendarError, match="no session on 2026-11-15"):
        record(FakeFeed(), [SBIN], now=clocks.wall, monotonic=clocks.mono, cache_dir=tmp_path)


def test_the_default_stop_after_the_close_raises_config_error(tmp_path):
    clocks = Clocks(ist_ns("16:00:00"))

    with pytest.raises(ConfigError, match="closed at 2026-11-16 15:30 IST"):
        record(FakeFeed(), [SBIN], now=clocks.wall, monotonic=clocks.mono, cache_dir=tmp_path)


def test_a_second_recording_on_the_cache_is_refused_while_one_runs(tmp_path):
    clocks, feed, script = recording()
    refused = []

    def second_recording():
        with pytest.raises(LiveError, match="another recording") as caught:
            run(tmp_path, Clocks(clocks.wall_ns), FakeFeed(), Script(clocks), (INFY,))
        refused.append(caught.type)

    script.at(ist_ns("10:00:01"), second_recording)

    run(tmp_path, clocks, feed, script)

    assert refused == [LiveError]


def test_finalize_is_refused_while_a_recording_runs(tmp_path):
    clocks, feed, script = recording()
    refused = []

    def finalize_now():
        with pytest.raises(LiveError, match="another recording") as caught:
            finalize(DAY, cache_dir=tmp_path)
        refused.append(caught.type)

    script.at(ist_ns("10:00:01"), finalize_now)

    run(tmp_path, clocks, feed, script)

    assert refused == [LiveError]


# ----- what an earlier recording left -------------------------------------------------


def stopped_without_finalizing(tmp_path) -> None:
    """A recording of SBIN from 09:30 that ends by an exception the recorder does not
    handle: its raw file stays, not finalized."""
    clocks, feed, script = recording(start="09:30:00")
    push_at(script, feed, "09:30:01", quote("SBIN", 810.0))

    def killed():
        raise SystemExit

    script.at(ist_ns("09:30:02"), killed)
    with pytest.raises(SystemExit):
        run(tmp_path, clocks, feed, script)


def test_a_raw_file_an_earlier_recording_left_is_merged_before_the_new_ticks(tmp_path):
    stopped_without_finalizing(tmp_path)
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    run(tmp_path, clocks, feed, script)

    assert ltps(tmp_path) == [810.0, 812.4]


def test_the_span_of_a_recording_that_did_not_finalize_ended_by_a_crash(tmp_path):
    stopped_without_finalizing(tmp_path)
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script)

    spans = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans
    assert [(span.start_ns, span.stop_ns, span.ended_by) for span in spans] == [
        (ist_ns("09:30:00"), ist_ns("09:30:01"), Ending.CRASH),
        (ist_ns(START), ist_ns(STOP), Ending.STOP),
    ]


def test_a_day_with_a_recording_that_did_not_finalize_has_no_known_outages(tmp_path):
    stopped_without_finalizing(tmp_path)
    clocks, feed, script = recording()

    run(tmp_path, clocks, feed, script)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages is None


def test_a_recording_that_did_not_finalize_leaves_its_raw_file_and_no_day(tmp_path):
    stopped_without_finalizing(tmp_path)

    with pytest.raises(DataError, match="its raw file is waiting"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


# ----- where the cache is -------------------------------------------------------------


@pytest.mark.parametrize("folder", ["ticks", "./ticks", "~"])
def test_a_relative_cache_dir_is_refused(tmp_path, monkeypatch, folder):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", "relative-home")
    monkeypatch.setenv("USERPROFILE", "relative-home")
    clocks, feed, script = recording()

    with pytest.raises(ConfigError, match="absolute folder"):
        record(feed, [SBIN], now=clocks.wall, monotonic=clocks.mono, cache_dir=folder)


def test_a_relative_cache_folder_in_the_environment_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALGO_BT_CACHE", "ticks")
    clocks, feed, script = recording()

    with pytest.raises(ConfigError, match="OPENALGO_BT_CACHE"):
        record(feed, [SBIN], now=clocks.wall, monotonic=clocks.mono, stop_at_ns=ist_ns(STOP))


def test_a_cache_dir_under_the_home_folder_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    clocks, feed, script = recording()
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    record(
        feed,
        [SBIN],
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        cache_dir="~/ticks",
        stop_at_ns=ist_ns(STOP),
    )

    assert ltps(tmp_path / "ticks") == [812.4]
