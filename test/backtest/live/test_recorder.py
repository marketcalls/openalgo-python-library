"""The tick recorder (FR-LIVE-19, recorder half): a recording connects the client's feed,
subscribes, writes every tick to the cache until the stop (the day's close, a given
instant, or Ctrl+C), keeps the feed's outages, then unsubscribes, disconnects and
finalizes. On a manual clock with a fake feed that a script plays; no socket is opened.
"""

from __future__ import annotations

import threading

import pytest
from recorder_fakes import (
    DAY,
    INFY,
    NS,
    SBIN,
    FakeFeed,
    ManualClock,
    Script,
    interrupt,
    ist_ns,
    quote,
)

from openalgo.backtest._live._tick_files import RawFile, raw_path, read_raw, ticks_root
from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.recorder import (
    Outage,
    Stop,
    finalize,
    read_day,
    read_ticks,
    record,
)
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import CalendarError, ConfigError, DataError, LiveError

START = "10:00:00"
STOP = "10:01:00"


def recording(tmp_path, *, start: str = START, **feed_options):
    """A clock at `start`, a fake feed and the script that plays it."""
    clock = ManualClock(ist_ns(start))
    return clock, FakeFeed(**feed_options), Script(clock)


def run(tmp_path, clock, feed, script, instruments=(SBIN,), **options):
    options.setdefault("stop_at_ns", ist_ns(STOP))
    return record(feed, list(instruments), now=clock, wait=script, cache_dir=tmp_path, **options)


def push_at(script: Script, feed: FakeFeed, clock: str, message: dict) -> None:
    script.at(ist_ns(clock), lambda: feed.push(message))


# ----- recording and reading back ---------------------------------------------------


def test_a_recording_reads_back_every_tick_of_a_symbol_in_receive_order(tmp_path):
    clock, feed, script = recording(tmp_path)
    trade_s = ist_ns("10:00:00.500") // NS
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4, volume=1_000, ltt=trade_s))
    push_at(script, feed, "10:00:02", quote("INFY", 1_500.0, volume=40))
    push_at(script, feed, "10:00:03", quote("SBIN", 812.5, volume=1_100))

    run(tmp_path, clock, feed, script, (SBIN, INFY))

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == [
        Tick("SBIN", "NSE", 812.4, ist_ns("10:00:01"), trade_s * NS, 1_000.0),
        Tick("SBIN", "NSE", 812.5, ist_ns("10:00:03"), None, 1_100.0),
    ]


def test_the_summary_counts_the_ticks_of_every_instrument_asked_for(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    push_at(script, feed, "10:00:02", quote("SBIN", 812.5))

    summary = run(tmp_path, clock, feed, script, (SBIN, INFY))

    assert dict(summary.ticks) == {("NSE", "SBIN"): 2, ("NSE", "INFY"): 0}


def test_an_instrument_without_a_tick_reads_back_as_a_recorded_day_with_none(tmp_path):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script, (SBIN, INFY))

    assert read_ticks("NSE", "INFY", DAY, cache_dir=tmp_path) == []


def test_the_summary_lists_the_day_file_of_each_instrument(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    summary = run(tmp_path, clock, feed, script, (SBIN, INFY))

    written = {(file.exchange, file.symbol, file.ticks) for file in summary.files}
    assert written == {("NSE", "SBIN", 1), ("NSE", "INFY", 0)}


def test_the_receive_time_is_when_the_writer_takes_a_tick_and_the_queued_time_when_it_came(
    tmp_path,
):
    clock, feed, script = recording(tmp_path)

    def tick_then_busy_writer():
        feed.push(quote("SBIN", 812.4))
        clock.ns += NS // 4  # the writer takes the tick a quarter second later

    script.at(ist_ns("10:00:01"), tick_then_busy_writer)

    run(tmp_path, clock, feed, script)

    records = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records
    assert (int(records["ts_queued_ns"][0]), int(records["ts_recv_ns"][0])) == (
        ist_ns("10:00:01"),
        ist_ns("10:00:01.250"),
    )


def test_a_tick_is_in_its_raw_file_within_a_second_of_its_receipt(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:00.200", quote("SBIN", 812.4))
    seen = []
    path = raw_path(ticks_root(tmp_path), "NSE", "SBIN", DAY)
    script.at(ist_ns("10:00:01.200"), lambda: seen.append(len(read_raw(path).records)))

    run(tmp_path, clock, feed, script)

    assert seen == [1]


def test_finalizing_leaves_no_raw_file(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    run(tmp_path, clock, feed, script)

    assert list(tmp_path.rglob("*.raw")) == []


def test_messages_that_make_no_tick_are_counted_as_dropped(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 0))
    push_at(script, feed, "10:00:02", quote("SBIN", 812.4))

    summary = run(tmp_path, clock, feed, script)

    assert (summary.dropped, summary.ticks[("NSE", "SBIN")]) == (1, 1)


def test_a_tick_with_an_implausible_trade_time_is_kept_without_it_and_counted(tmp_path):
    clock, feed, script = recording(tmp_path)
    counted_from_1980 = ist_ns("10:00:00") // NS - 315_532_800
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4, ltt=counted_from_1980))

    summary = run(tmp_path, clock, feed, script)

    (tick,) = read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)
    assert (tick.ts_event_ns, summary.event_times_rejected) == (None, 1)


# ----- stopping -----------------------------------------------------------------------


def test_a_recording_stops_at_the_days_close_by_default(tmp_path):
    clock, feed, script = recording(tmp_path, start="15:29:50")
    push_at(script, feed, "15:29:55", quote("SBIN", 812.4))

    summary = run(tmp_path, clock, feed, script, stop_at_ns=None)

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.CLOSE, ist_ns("15:30:00"))


def test_the_ticks_before_the_close_are_finalized_when_it_stops_there(tmp_path):
    clock, feed, script = recording(tmp_path, start="15:29:50")
    push_at(script, feed, "15:29:55", quote("SBIN", 812.4))

    run(tmp_path, clock, feed, script, stop_at_ns=None)

    assert [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)] == [812.4]


def test_the_default_stop_is_the_latest_close_of_the_instruments_exchanges(tmp_path):
    clock, feed, script = recording(tmp_path, start="23:54:58")
    crude = {"exchange": "MCX", "symbol": "CRUDEOIL25NOVFUT"}

    summary = run(tmp_path, clock, feed, script, (SBIN, crude), stop_at_ns=None)

    assert summary.stopped_ns == default_calendar().session("MCX", DAY)[1]


def test_a_recording_stops_at_the_instant_given(tmp_path):
    clock, feed, script = recording(tmp_path)

    summary = run(tmp_path, clock, feed, script, stop_at_ns=ist_ns("10:00:30"))

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.STOP_AT, ist_ns("10:00:30"))


def test_ctrl_c_stops_the_recording(tmp_path):
    clock, feed, script = recording(tmp_path)
    script.at(ist_ns("10:00:05"), interrupt)

    summary = run(tmp_path, clock, feed, script)

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.INTERRUPT, ist_ns("10:00:05"))


def test_ctrl_c_still_finalizes_the_ticks_recorded_before_it(tmp_path):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))
    script.at(ist_ns("10:00:01.500"), interrupt)

    run(tmp_path, clock, feed, script)

    assert [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)] == [812.4]


def test_a_ctrl_c_right_after_a_batch_was_written_never_writes_it_twice(tmp_path, monkeypatch):
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:00.500", quote("SBIN", 812.4))
    write = RawFile.append
    writes = []

    def write_then_ctrl_c(raw, records):
        write(raw, records)
        writes.append(len(records))
        if len(writes) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(RawFile, "append", write_then_ctrl_c)

    run(tmp_path, clock, feed, script)

    assert len(read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == 1


def test_stopping_unsubscribes_then_disconnects(tmp_path):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script)

    assert feed.calls[:4] == ["connect", "subscribe_quote", "unsubscribe_quote", "disconnect"]


def test_the_recording_subscribes_and_unsubscribes_the_instruments_asked_for(tmp_path):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script, (SBIN, INFY, SBIN))

    assert feed.requests["subscribe_quote"] == feed.requests["unsubscribe_quote"] == [SBIN, INFY]


def test_stopping_removes_the_recordings_hooks_from_the_client(tmp_path):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script)

    assert feed.hooks == {"disconnect": [], "reconnect": []}


@pytest.mark.parametrize("mode", ["ltp", "quote", "depth"])
def test_the_recording_subscribes_in_the_mode_asked_for(tmp_path, mode):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script, mode=mode)

    assert feed.calls[1] == f"subscribe_{mode}"


# ----- outages ------------------------------------------------------------------------


def test_an_outage_runs_from_the_disconnect_to_the_reconnect(tmp_path):
    clock, feed, script = recording(tmp_path)
    script.at(ist_ns("10:00:10"), feed.drop)
    script.at(ist_ns("10:00:25"), feed.restore)

    summary = run(tmp_path, clock, feed, script)

    assert summary.outages == (Outage(ist_ns("10:00:10"), ist_ns("10:00:25")),)


def test_the_days_outages_are_kept_with_its_ticks(tmp_path):
    clock, feed, script = recording(tmp_path)
    script.at(ist_ns("10:00:10"), feed.drop)
    script.at(ist_ns("10:00:25"), feed.restore)

    run(tmp_path, clock, feed, script)

    outages = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages
    assert outages == (Outage(ist_ns("10:00:10"), ist_ns("10:00:25")),)


def test_an_outage_still_under_way_ends_with_the_recording(tmp_path):
    clock, feed, script = recording(tmp_path)
    script.at(ist_ns("10:00:50"), feed.drop)

    summary = run(tmp_path, clock, feed, script)

    assert summary.outages == (Outage(ist_ns("10:00:50"), ist_ns(STOP)),)


def test_the_disconnect_the_recording_asks_for_is_no_outage(tmp_path):
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages == ()


# ----- the feed's thread never waits --------------------------------------------------


def test_the_data_callback_never_blocks_on_a_full_queue_and_counts_what_it_lost(tmp_path):
    clock, feed, script = recording(tmp_path)
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

    summary = run(tmp_path, clock, feed, script, queue_size=2)

    assert (finished, summary.overflowed, summary.ticks[("NSE", "SBIN")]) == ([True], 3, 2)


# ----- what a recording refuses -------------------------------------------------------


def test_a_feed_that_does_not_connect_raises_live_error(tmp_path):
    clock, feed, script = recording(tmp_path, connects=False)

    with pytest.raises(LiveError, match="did not connect"):
        run(tmp_path, clock, feed, script)


def test_a_feed_that_does_not_connect_leaves_no_tick_file(tmp_path):
    clock, feed, script = recording(tmp_path, connects=False)

    with pytest.raises(LiveError):
        run(tmp_path, clock, feed, script)

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
    clock, feed, script = recording(tmp_path)
    options.setdefault("stop_at_ns", ist_ns(STOP))

    with pytest.raises(ConfigError, match=message):
        record(feed, instruments, now=clock, wait=script, cache_dir=tmp_path, **options)


def test_a_client_without_the_feed_calls_raises_config_error(tmp_path):
    clock = ManualClock(ist_ns(START))

    with pytest.raises(ConfigError, match="subscribe_quote"):
        record(object(), [SBIN], now=clock, cache_dir=tmp_path, stop_at_ns=ist_ns(STOP))


def test_the_default_stop_on_a_day_without_a_session_raises_calendar_error(tmp_path):
    clock = ManualClock(ist_ns("10:00:00", DAY.replace(day=15)))  # a Sunday

    with pytest.raises(CalendarError, match="no session on 2026-11-15"):
        record(FakeFeed(), [SBIN], now=clock, cache_dir=tmp_path)


def test_the_default_stop_after_the_close_raises_config_error(tmp_path):
    clock = ManualClock(ist_ns("16:00:00"))

    with pytest.raises(ConfigError, match="closed at 2026-11-16 15:30 IST"):
        record(FakeFeed(), [SBIN], now=clock, cache_dir=tmp_path)


def test_a_second_recording_on_the_cache_is_refused_while_one_runs(tmp_path):
    clock, feed, script = recording(tmp_path)
    refused = []

    def second_recording():
        with pytest.raises(LiveError, match="another recording") as caught:
            record(FakeFeed(), [INFY], now=clock, cache_dir=tmp_path, stop_at_ns=ist_ns(STOP))
        refused.append(caught.type)

    script.at(ist_ns("10:00:01"), second_recording)

    run(tmp_path, clock, feed, script)

    assert refused == [LiveError]


def test_finalize_is_refused_while_a_recording_runs(tmp_path):
    clock, feed, script = recording(tmp_path)
    refused = []

    def finalize_now():
        with pytest.raises(LiveError, match="another recording") as caught:
            finalize(DAY, cache_dir=tmp_path)
        refused.append(caught.type)

    script.at(ist_ns("10:00:01"), finalize_now)

    run(tmp_path, clock, feed, script)

    assert refused == [LiveError]


# ----- what an earlier recording left -------------------------------------------------


def stopped_without_finalizing(tmp_path) -> None:
    """A recording of SBIN from 09:30 that was killed: its raw file stays."""
    clock, feed, script = recording(tmp_path, start="09:30:00")
    push_at(script, feed, "09:30:01", quote("SBIN", 810.0))

    def killed():
        raise SystemExit  # not a stop the recorder handles: it finalizes nothing

    script.at(ist_ns("09:30:02"), killed)
    with pytest.raises(SystemExit):
        run(tmp_path, clock, feed, script)


def test_a_raw_file_an_earlier_recording_left_is_merged_before_the_new_ticks(tmp_path):
    stopped_without_finalizing(tmp_path)
    clock, feed, script = recording(tmp_path)
    push_at(script, feed, "10:00:01", quote("SBIN", 812.4))

    run(tmp_path, clock, feed, script)

    assert [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)] == [
        810.0,
        812.4,
    ]


def test_a_day_merged_with_what_a_killed_recording_left_has_no_known_outages(tmp_path):
    stopped_without_finalizing(tmp_path)
    clock, feed, script = recording(tmp_path)

    run(tmp_path, clock, feed, script)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages is None


def test_a_killed_recording_leaves_its_raw_file_and_no_day(tmp_path):
    stopped_without_finalizing(tmp_path)

    with pytest.raises(DataError, match="its raw file is waiting"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)
