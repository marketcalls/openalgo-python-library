"""How a recording stops and what it never loses: a second Ctrl+C while it stops, a Ctrl+C
in the middle of a write, a write that fails, a wall clock that steps back. On manual
clocks with a fake feed; the write failures are injected at the raw file's one write
call (`RawFile._write_bytes`) and its one truncation (`RawFile._truncate`).
"""

from __future__ import annotations

import errno

from recorder_fakes import (
    DAY,
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

from openalgo.backtest._live._tick_raw import RawFile, read_raw
from openalgo.backtest._live.recorder import Stop, read_ticks, record

STOP = "10:01:00"


def recording():
    clocks = Clocks(ist_ns("10:00:00"))
    return clocks, FakeFeed(), Script(clocks)


def run(tmp_path, clocks, feed, script):
    return record(
        feed,
        [SBIN],
        now=clocks.wall,
        monotonic=clocks.mono,
        wait=script,
        cache_dir=tmp_path,
        stop_at_ns=ist_ns(STOP),
    )


def ltps(tmp_path) -> list[float]:
    return [tick.ltp for tick in read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)]


def a_tick_in_memory_then_ctrl_c(script: Script, feed: FakeFeed) -> None:
    """A tick taken at 10:00:00.200, held until the flush at 10:00:01, and a Ctrl+C at
    10:00:00.500 before that flush."""
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.500"), interrupt)


def disk_full() -> OSError:
    return OSError(errno.ENOSPC, "No space left on device")


def failing_writes(monkeypatch, *, fail: int, partial: bool = False, error=disk_full):
    """The first `fail` record writes raise `error()`; with `partial` each failing one
    first writes half its bytes, as a write cut short by a full disk does."""
    write = RawFile._write_bytes
    calls = []

    def flaky(raw, data):
        calls.append(len(data))
        if len(calls) <= fail:
            if partial:
                write(raw, data[: len(data) // 2])
            raise error()
        write(raw, data)

    monkeypatch.setattr(RawFile, "_write_bytes", flaky)
    return calls


# ----- a second Ctrl+C while stopping -------------------------------------------------


def interrupted_twice(tmp_path):
    """A first Ctrl+C stops the recording; a second lands inside `disconnect()`."""
    clocks, feed, script = recording()
    a_tick_in_memory_then_ctrl_c(script, feed)
    disconnect = feed.disconnect

    def disconnect_then_ctrl_c():
        disconnect()
        raise KeyboardInterrupt

    feed.disconnect = disconnect_then_ctrl_c
    return feed, run(tmp_path, clocks, feed, script)


def test_a_second_ctrl_c_while_disconnecting_loses_no_tick(tmp_path):
    interrupted_twice(tmp_path)

    assert ltps(tmp_path) == [812.4]


def test_a_second_ctrl_c_while_disconnecting_leaves_no_raw_file_open_or_unfinalized(tmp_path):
    _, summary = interrupted_twice(tmp_path)

    # On Windows a raw file still open could not be removed once finalized.
    assert (raw_files(tmp_path), [file.ticks for file in summary.files]) == ([], [1])


def test_a_second_ctrl_c_while_disconnecting_still_removes_the_hooks(tmp_path):
    feed, _ = interrupted_twice(tmp_path)

    assert feed.hooks == {"disconnect": [], "reconnect": []}


def test_a_second_ctrl_c_while_disconnecting_is_noted_in_the_summary_it_returns(tmp_path):
    _, summary = interrupted_twice(tmp_path)

    assert summary.stopped_by is Stop.INTERRUPT
    assert any("disconnect" in note and "KeyboardInterrupt" in note for note in summary.notes)


def test_a_second_ctrl_c_while_the_last_ticks_are_written_loses_no_tick(tmp_path, monkeypatch):
    clocks, feed, script = recording()
    a_tick_in_memory_then_ctrl_c(script, feed)
    failing_writes(monkeypatch, fail=1, error=KeyboardInterrupt)

    run(tmp_path, clocks, feed, script)

    assert ltps(tmp_path) == [812.4]


def test_a_batch_written_twice_after_a_ctrl_c_right_after_its_append_is_kept_once(
    tmp_path, monkeypatch
):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.500"), lambda: feed.push(quote("SBIN", 812.4)))
    append = RawFile.append
    appends = []

    def append_then_ctrl_c(raw, records):
        append(raw, records)
        appends.append(len(records))
        if len(appends) == 1:
            raise KeyboardInterrupt  # the frame is whole on disk; the batch is still held

    monkeypatch.setattr(RawFile, "append", append_then_ctrl_c)

    summary = run(tmp_path, clocks, feed, script)

    # The raw file held the batch twice; the second copy does not increase the sequence.
    assert (len(appends), ltps(tmp_path), summary.duplicates, summary.rejected) == (
        2,
        [812.4],
        1,
        0,
    )


def test_a_ctrl_c_in_the_middle_of_a_write_cuts_the_frame_off_before_writing_it_again(
    tmp_path, monkeypatch
):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.500"), lambda: feed.push(quote("SBIN", 812.4)))
    write = RawFile._write_bytes
    writes = []

    def write_then_ctrl_c(raw, data):
        write(raw, data)
        writes.append(len(data))
        if len(writes) == 1:
            raise KeyboardInterrupt  # before the write was synced and counted whole

    monkeypatch.setattr(RawFile, "_write_bytes", write_then_ctrl_c)

    summary = run(tmp_path, clocks, feed, script)

    assert (len(writes), ltps(tmp_path), summary.duplicates, summary.rejected) == (
        2,
        [812.4],
        0,
        0,
    )


# ----- write failures -----------------------------------------------------------------


def test_a_batch_whose_write_failed_stays_and_is_written_when_the_recording_stops(
    tmp_path, monkeypatch
):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    failing_writes(monkeypatch, fail=1)

    summary = run(tmp_path, clocks, feed, script)

    assert (ltps(tmp_path), summary.lost) == ([812.4], 0)


def test_a_write_failure_stops_the_recording_and_the_summary_holds_the_error(tmp_path, monkeypatch):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    failing_writes(monkeypatch, fail=1)

    summary = run(tmp_path, clocks, feed, script)

    assert (summary.stopped_by, summary.stopped_ns) == (Stop.WRITE_FAILED, ist_ns("10:00:01"))
    assert isinstance(summary.error, OSError) and summary.error.errno == errno.ENOSPC


def test_ticks_that_could_never_be_written_are_counted_as_lost(tmp_path, monkeypatch):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.300"), lambda: feed.push(quote("SBIN", 812.5)))
    failing_writes(monkeypatch, fail=10**6)

    summary = run(tmp_path, clocks, feed, script)

    assert (summary.lost, summary.ticks[("NSE", "SBIN")], ltps(tmp_path)) == (2, 0, [])


def test_after_a_write_cut_short_the_raw_file_is_cut_back_to_its_last_whole_record(
    tmp_path, monkeypatch
):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    failing_writes(monkeypatch, fail=1, partial=True)

    summary = run(tmp_path, clocks, feed, script)

    # Half a record was cut off before the batch was written again: nothing is skipped.
    assert (ltps(tmp_path), summary.rejected) == ([812.4], 0)


def test_after_the_final_write_failed_too_the_summary_counts_what_reached_the_disk(
    tmp_path, monkeypatch
):
    # Two torn writes, and no tear can be cut off: the recording goes on in part 1, whose
    # one frame is torn as well. Nothing whole reached the disk.
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.400"), lambda: feed.push(quote("SBIN", 812.5)))
    failing_writes(monkeypatch, fail=2, partial=True)

    def cannot_truncate(raw, size):
        raise OSError(errno.EACCES, "Access is denied")

    monkeypatch.setattr(RawFile, "_truncate", cannot_truncate)

    summary = run(tmp_path, clocks, feed, script)

    read_back = ltps(tmp_path)
    assert (summary.ticks[("NSE", "SBIN")], summary.lost) == (len(read_back), 2 - len(read_back))


def test_when_a_tear_cannot_be_cut_off_writing_moves_to_a_new_raw_file(tmp_path, monkeypatch):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.200"), lambda: feed.push(quote("SBIN", 812.4)))
    failing_writes(monkeypatch, fail=1, partial=True)

    def cannot_truncate(raw, size):
        raise OSError(errno.EACCES, "Access is denied")

    monkeypatch.setattr(RawFile, "_truncate", cannot_truncate)

    summary = run(tmp_path, clocks, feed, script)

    assert ltps(tmp_path) == [812.4]
    assert any("new raw file" in note for note in summary.notes)


# ----- the clocks ---------------------------------------------------------------------


def test_flushing_goes_on_when_the_wall_clock_steps_back(tmp_path):
    clocks, feed, script = recording()
    script.at(ist_ns("10:00:00.500"), lambda: feed.push(quote("SBIN", 812.4)))
    script.at(ist_ns("10:00:00.600"), lambda: clocks.step_wall(-600 * NS))
    for second in (1, 2, 3):
        script.at(ist_ns(f"10:00:0{second}"), lambda n=second: feed.push(quote("SBIN", 812.4 + n)))
    seen = []
    script.at(
        ist_ns("10:00:04.500"),
        lambda: seen.append(sum(len(read_raw(path).records) for path in raw_files(tmp_path))),
    )
    script.at(ist_ns("10:00:05"), interrupt)

    run(tmp_path, clocks, feed, script)

    assert seen == [4]
