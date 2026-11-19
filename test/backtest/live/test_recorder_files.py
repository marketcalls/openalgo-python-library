"""The recorder's files, written once: one raw file per recording that a crash or a
power loss can tear, finalize turning it into one verified file and sidecar per
instrument and removing it only then, readers that verify and never write, and the
explicit repair that moves a damaged file aside. Raw files are written here as a
recording writes them, so these tests need no feed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import pathlib
import shutil

import numpy as np
import pytest
from recorder_fakes import DAY, NS, ist_ns

import openalgo.backtest._live._tick_finalize as tick_finalize
from openalgo.backtest._live._tick_files import (
    NO_EVENT_TIME,
    RAW_DTYPE,
    TICK_DTYPE,
    finalized_paths,
    instrument_folder,
    next_recording_number,
    raw_path,
    records_of,
    start_path,
    stop_path,
    tick_row,
    ticks_root,
)
from openalgo.backtest._live._tick_finalize import StartFacts, StopFacts, write_start, write_stop
from openalgo.backtest._live._tick_raw import (
    FRAME_HEADER_SIZE,
    RawFile,
    RawHeader,
    frame_of,
    read_raw,
)
from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.recorder import (
    Coverage,
    Outage,
    Stop,
    finalize,
    read_day,
    read_merged_ticks,
    read_ticks,
    repair,
)
from openalgo.backtest.errors import DataError

#: The hash in a sidecar covers this versioned header, then the records' bytes.
HASH_HEADER = b"openalgo.backtest recorded ticks v4\n"
SBIN, INFY = ("NSE", "SBIN"), ("NSE", "INFY")


def sbin(clock: str, ltp: float, **fields) -> Tick:
    return Tick("SBIN", "NSE", ltp, ist_ns(clock), **fields)


def infy(clock: str, ltp: float) -> Tick:
    return Tick("INFY", "NSE", ltp, ist_ns(clock))


def root_of(tmp_path):
    return ticks_root(tmp_path)


def folder_of(tmp_path, symbol="SBIN"):
    return instrument_folder(root_of(tmp_path), "NSE", symbol)


def facts_of(number=1, start="10:00:00", instruments=(SBIN,)) -> StartFacts:
    """What a recording records; its identity, made from its number, is fixed here."""
    return StartFacts(number, f"{number:016x}", DAY, ist_ns(start), tuple(instruments))


def header_of(number=1, start="10:00:00", instruments=(SBIN,), part=0) -> RawHeader:
    facts = facts_of(number, start, instruments)
    return RawHeader(number, facts.identity, part, DAY, facts.start_ns, facts.instruments)


def rows_of(ticks, *, instruments=(SBIN,), number=1, seq=1) -> list[tuple]:
    """One raw row per tick (of the instrument its symbol names), sequence numbers from
    `seq`."""
    index = {symbol: n for n, (_, symbol) in enumerate(instruments)}
    return [
        tick_row(
            tick,
            instrument=index[tick.symbol],
            recording=number,
            seq=seq + n,
            queued_ns=tick.ts_recv_ns,
        )
        for n, tick in enumerate(ticks)
    ]


def write_raw(tmp_path, ticks, *, number=1, start="10:00:00", instruments=(SBIN,), part=0, seq=1):
    """A recording's start sidecar and raw file as the writer leaves them: the header,
    then one frame holding one record per tick."""
    if part == 0:
        write_start(root_of(tmp_path), facts_of(number, start, instruments))
    header = header_of(number, start, instruments, part)
    raw = RawFile.create(raw_path(root_of(tmp_path), DAY, number, part), header)
    rows = rows_of(ticks, instruments=instruments, number=number, seq=seq)
    if rows:
        raw.append(records_of(rows))
    raw.close()
    return raw.path


def stopped(
    tmp_path,
    *,
    number=1,
    start="10:00:00",
    instruments=(SBIN,),
    stopped_by=Stop.STOP_AT,
    stop="10:01:00",
    outages=(),
):
    """The stop sidecar a recording writes before it finalizes."""
    facts = StopFacts(stopped_by, ist_ns(stop), tuple(outages))
    write_stop(root_of(tmp_path), facts_of(number, start, instruments), facts)


def files_of(tmp_path, number=1, symbol="SBIN"):
    return finalized_paths(folder_of(tmp_path, symbol), DAY, number)


def sidecar_of(tmp_path, number=1, symbol="SBIN") -> dict:
    return json.loads(files_of(tmp_path, number, symbol)[1].read_text("utf-8"))


def damage(path) -> None:
    """Change the last byte of a file, which is inside its last record."""
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))


def snapshot(folder) -> dict:
    """Every file of a folder and its bytes."""
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


TWO = [sbin("10:00:01", 812.4, volume=1_000.0), sbin("10:00:02", 812.5, volume=1_100.0)]


def finalized_two(tmp_path) -> None:
    write_raw(tmp_path, TWO)
    stopped(tmp_path)
    finalize(DAY, cache_dir=tmp_path)


def ltps(tmp_path, symbol="SBIN") -> list[float]:
    return [tick.ltp for tick in read_ticks("NSE", symbol, DAY, cache_dir=tmp_path)]


# ----- raw files ----------------------------------------------------------------------


def test_a_records_layouts_are_fixed_fields_in_utc_nanoseconds_and_rupees():
    assert [(name, TICK_DTYPE[name].str) for name in TICK_DTYPE.names] == [
        ("recording", "<u8"),
        ("seq", "<i8"),
        ("ts_recv_ns", "<i8"),
        ("ts_queued_ns", "<i8"),
        ("ts_event_ns", "<i8"),
        ("ltp", "<f8"),
        ("volume", "<f8"),
        ("bid", "<f8"),
        ("ask", "<f8"),
    ]
    assert RAW_DTYPE.names == ("instrument", *TICK_DTYPE.names)


def test_a_raw_files_header_lists_the_recordings_instruments(tmp_path):
    path = write_raw(tmp_path, [], instruments=(SBIN, INFY))

    assert read_raw(path).header.instruments == (SBIN, INFY)


def test_a_torn_last_record_is_cut_off_on_read(tmp_path):
    path = write_raw(tmp_path, TWO)
    with open(path, "ab") as handle:
        handle.write(b"\x01" * 10)  # a crash in the middle of a record

    content = read_raw(path)

    assert (len(content.records), content.torn_bytes) == (2, 10)


def test_a_zero_filled_tail_left_by_a_power_loss_ends_the_file_and_is_counted(tmp_path):
    path = write_raw(tmp_path, TWO[:1])
    with open(path, "ab") as handle:
        handle.write(bytes(3 * RAW_DTYPE.itemsize))
    stopped(tmp_path)

    report = finalize(DAY, cache_dir=tmp_path)

    assert (ltps(tmp_path), report.torn_bytes) == ([812.4], 3 * RAW_DTYPE.itemsize)


@pytest.mark.parametrize("cut", [2, 6, 10, 30, 44, 50, 60, 74])
def test_a_flush_a_power_loss_zero_filled_inside_is_dropped_whole_never_a_wrong_tick(tmp_path, cut):
    # The page boundary falls `cut` bytes into the frame's last record; the zeros go on.
    path = write_raw(tmp_path, TWO)
    blob = bytearray(path.read_bytes())
    last = len(blob) - RAW_DTYPE.itemsize
    blob[last + cut :] = bytes(len(blob) - last - cut)
    path.write_bytes(bytes(blob) + bytes(2 * RAW_DTYPE.itemsize))

    report = finalize(DAY, cache_dir=tmp_path)

    assert (ltps(tmp_path), report.rejected) == ([], 2)


def test_reading_goes_on_after_a_frame_whose_records_do_not_match_their_crc(tmp_path):
    path = write_raw(tmp_path, TWO[:1])
    later = [sbin("10:00:02", 812.5), sbin("10:00:03", 812.6)]
    torn, whole = (
        frame_of(records_of(rows_of([tick], seq=n))) for n, tick in ((2, later[0]), (3, later[1]))
    )
    damaged = torn[:FRAME_HEADER_SIZE] + bytes(len(torn) - FRAME_HEADER_SIZE)
    with open(path, "ab") as handle:
        handle.write(damaged + whole)

    content = read_raw(path)

    assert (content.records["ltp"].tolist(), content.dropped_frames, content.rejected) == (
        [812.4, 812.6],
        1,
        1,
    )


def test_records_that_cannot_be_the_recordings_are_skipped_and_counted(tmp_path):
    path = write_raw(tmp_path, TWO)
    stray = [
        tick_row(sbin("10:00:03", 1.0), instrument=0, recording=9, seq=3, queued_ns=0),
        tick_row(sbin("10:00:05", math.nan), instrument=0, recording=1, seq=4, queued_ns=0),
        tick_row(sbin("10:00:06", 3.0), instrument=7, recording=1, seq=5, queued_ns=0),
    ]
    with open(path, "ab") as handle:
        handle.write(frame_of(records_of(stray)))

    content = read_raw(path)

    assert (content.records["ltp"].tolist(), content.rejected) == ([812.4, 812.5], 3)


def test_a_batch_written_twice_is_kept_once_and_counted_as_duplicates_not_rejected(tmp_path):
    path = write_raw(tmp_path, TWO)
    with open(path, "ab") as handle:
        handle.write(frame_of(records_of(rows_of(TWO))))

    report = finalize(DAY, cache_dir=tmp_path)

    assert (ltps(tmp_path), report.duplicates, report.rejected) == ([812.4, 812.5], 2, 0)


def test_after_a_write_cut_short_the_next_append_first_cuts_the_file_to_whole_records(
    tmp_path, monkeypatch
):
    raw = RawFile.create(raw_path(root_of(tmp_path), DAY, 1, 0), header_of())
    rows = [
        tick_row(t, instrument=0, recording=1, seq=n + 1, queued_ns=0) for n, t in enumerate(TWO)
    ]
    raw.append(records_of(rows[:1]))
    write = RawFile._write_bytes

    def half_then_fail(file, data):
        write(file, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(RawFile, "_write_bytes", half_then_fail)
    with pytest.raises(OSError):
        raw.append(records_of(rows[1:]))
    monkeypatch.setattr(RawFile, "_write_bytes", write)

    raw.append(records_of(rows[1:]))
    raw.close()

    content = read_raw(raw.path)
    assert (content.records["ltp"].tolist(), content.torn_bytes, content.rejected) == (
        [812.4, 812.5],
        0,
        0,
    )


def test_every_append_ends_with_a_sync_to_the_disk(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(RawFile, "_sync", lambda raw: synced.append(raw.path))
    raw = RawFile.create(raw_path(root_of(tmp_path), DAY, 1, 0), header_of())

    for tick in TWO:
        raw.append(records_of([tick_row(tick, instrument=0, recording=1, seq=1, queued_ns=0)]))
    raw.close()

    assert synced == [raw.path, raw.path]


# ----- finalize -----------------------------------------------------------------------


def test_finalize_writes_one_file_and_sidecar_per_instrument_of_the_recording(tmp_path):
    write_raw(
        tmp_path, [sbin("10:00:01", 812.4), infy("10:00:02", 1_500.0)], instruments=(SBIN, INFY)
    )
    stopped(tmp_path)

    finalize(DAY, cache_dir=tmp_path)

    assert all(path.exists() for path in (*files_of(tmp_path), *files_of(tmp_path, symbol="INFY")))
    assert (ltps(tmp_path), ltps(tmp_path, "INFY")) == ([812.4], [1_500.0])


def test_finalize_removes_the_raw_file_and_its_stop_sidecar(tmp_path):
    path = write_raw(tmp_path, TWO)
    stopped(tmp_path)

    finalize(DAY, cache_dir=tmp_path)

    assert list(path.parent.iterdir()) == []


def test_the_sidecar_holds_the_files_hash_over_its_records(tmp_path):
    finalized_two(tmp_path)

    records = np.load(files_of(tmp_path)[0], allow_pickle=False)
    expected = "sha256:" + hashlib.sha256(HASH_HEADER + records.tobytes()).hexdigest()
    assert sidecar_of(tmp_path)["sha256"] == expected


def test_the_sidecar_holds_the_tick_count_and_the_first_and_last_receive_time(tmp_path):
    finalized_two(tmp_path)

    sidecar = sidecar_of(tmp_path)

    assert (sidecar["ticks"], sidecar["first_recv_ns"], sidecar["last_recv_ns"]) == (
        2,
        ist_ns("10:00:01"),
        ist_ns("10:00:02"),
    )


def test_a_recording_that_wrote_how_it_stopped_spans_to_its_stop_with_its_outages(tmp_path):
    write_raw(tmp_path, TWO)
    outage = Outage(ist_ns("10:00:10"), ist_ns("10:00:20"))
    stopped(tmp_path, stopped_by=Stop.INTERRUPT, stop="10:00:30", outages=[outage])
    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.recording, span.start_ns, span.stop_ns, span.ended_by, span.outages) == (
        1,
        ist_ns("10:00:00"),
        ist_ns("10:00:30"),
        Stop.INTERRUPT,
        (outage,),
    )


def test_a_recording_that_did_not_say_how_it_stopped_crashed_at_its_greatest_receive_time(
    tmp_path,
):
    # The wall clock stepped back: the last record was received before an earlier one.
    write_raw(tmp_path, [sbin("10:00:02", 812.4), sbin("10:00:01.500", 812.5)])
    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.stop_ns, span.ended_by, span.outages) == (ist_ns("10:00:02"), Stop.CRASH, None)


def test_a_crashed_recording_without_a_tick_spans_no_time(tmp_path):
    write_raw(tmp_path, [], start="10:00:00")
    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.start_ns, span.stop_ns) == (ist_ns("10:00:00"), ist_ns("10:00:00"))


def test_a_tick_taken_after_the_stop_is_inside_the_span(tmp_path):
    write_raw(tmp_path, [sbin("10:01:00.050", 812.4)])
    stopped(tmp_path, stop="10:01:00")
    finalize(DAY, cache_dir=tmp_path)

    day = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert day.coverage(ist_ns("10:01:00.050")) is Coverage.RECORDED


def test_a_later_recording_never_rewrites_an_earlier_recordings_file(tmp_path):
    finalized_two(tmp_path)
    earlier = files_of(tmp_path)[0]
    before = (earlier.read_bytes(), earlier.stat().st_mtime_ns)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], number=2, start="10:59:00")
    stopped(tmp_path, number=2, start="10:59:00", stop="11:01:00")

    finalize(DAY, cache_dir=tmp_path)

    assert (earlier.read_bytes(), earlier.stat().st_mtime_ns) == before


def test_a_day_reads_every_recordings_file_in_the_order_of_their_numbers(tmp_path):
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], number=2, start="10:59:00")
    write_raw(tmp_path, TWO, number=1)
    finalize(DAY, cache_dir=tmp_path)

    assert ltps(tmp_path) == [812.4, 812.5, 815.0]


def test_finalizing_a_raw_file_whose_files_exist_and_verify_only_removes_it(tmp_path):
    path = write_raw(tmp_path, TWO)
    kept = shutil.copy(path, tmp_path / "copy.raw")
    finalize(DAY, cache_dir=tmp_path)
    written = files_of(tmp_path)[0]
    before = (written.read_bytes(), written.stat().st_mtime_ns)
    shutil.copy(kept, path)  # the crash came before the raw file was removed

    finalize(DAY, cache_dir=tmp_path)

    assert ((written.read_bytes(), written.stat().st_mtime_ns), path.exists()) == (before, False)


def test_a_finalize_cut_short_between_instruments_is_finished_by_the_next(tmp_path):
    write_raw(
        tmp_path, [sbin("10:00:01", 812.4), infy("10:00:02", 1_500.0)], instruments=(SBIN, INFY)
    )
    finalize(DAY, cache_dir=tmp_path)
    for path in files_of(tmp_path, symbol="INFY"):
        path.unlink()  # as if the crash came before INFY's file was written
    raw = write_raw(
        tmp_path, [sbin("10:00:01", 812.4), infy("10:00:02", 1_500.0)], instruments=(SBIN, INFY)
    )

    finalize(DAY, cache_dir=tmp_path)

    assert (ltps(tmp_path), ltps(tmp_path, "INFY"), raw.exists()) == ([812.4], [1_500.0], False)


def test_the_raw_file_stays_while_a_finalized_file_of_it_is_damaged(tmp_path):
    path = write_raw(tmp_path, TWO)
    kept = shutil.copy(path, tmp_path / "copy.raw")
    finalize(DAY, cache_dir=tmp_path)
    damage(files_of(tmp_path)[0])
    shutil.copy(kept, path)

    report = finalize(DAY, cache_dir=tmp_path)

    assert (path.exists(), report.pending, any("damaged" in note for note in report.notes)) == (
        True,
        (1,),
        True,
    )


def test_a_raw_file_that_cannot_be_removed_stays_and_its_span_does_not_change(
    tmp_path, monkeypatch
):
    path = write_raw(tmp_path, TWO)
    outage = Outage(ist_ns("10:00:10"), ist_ns("10:00:20"))
    stopped(tmp_path, outages=[outage])
    unlink = pathlib.Path.unlink

    def held_open_elsewhere(file, missing_ok=False):
        if file == path:
            raise PermissionError(13, "The process cannot access the file", str(file))
        unlink(file, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", held_open_elsewhere)
    report = finalize(DAY, cache_dir=tmp_path)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)

    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans
    assert (report.pending, path.exists(), span.ended_by, span.outages) == (
        (1,),
        False,
        Stop.STOP_AT,
        (outage,),
    )


def test_finalize_moves_aside_a_file_that_is_not_a_raw_file(tmp_path):
    path = raw_path(root_of(tmp_path), DAY, 1, 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not ticks")

    report = finalize(DAY, cache_dir=tmp_path)

    moved = list(path.parent.glob(f"{path.name}.damaged-*"))
    assert (report.files, len(moved), "not a raw tick file" in report.notes[0]) == ((), 1, True)


@pytest.mark.parametrize("symbol", ["M&M", "BAJAJ-AUTO", "NIFTY 50", "SBIN/EQ"])
def test_a_symbol_with_characters_unsafe_in_a_path_round_trips(tmp_path, symbol):
    tick = Tick(symbol, "NSE", 100.0, ist_ns("10:00:01"))
    write_raw(tmp_path, [tick], instruments=(("NSE", symbol),))

    (written,) = finalize(DAY, cache_dir=tmp_path).files

    assert (written.symbol, read_ticks("NSE", symbol, DAY, cache_dir=tmp_path)) == (symbol, [tick])


def test_recording_numbers_count_up_across_the_cache(tmp_path):
    root = root_of(tmp_path)

    assert [next_recording_number(root, DAY) for _ in range(3)] == [1, 2, 3]


def test_a_lost_counter_never_gives_a_number_a_file_of_the_cache_has(tmp_path):
    write_raw(tmp_path, TWO, number=7)
    finalize(DAY, cache_dir=tmp_path)

    assert next_recording_number(root_of(tmp_path), DAY + dt.timedelta(days=1)) == 8


def test_a_counter_behind_the_days_files_never_gives_a_number_of_that_day(tmp_path):
    root = root_of(tmp_path)
    next_recording_number(root, DAY)  # the counter says 1, as a copy of the cache did
    write_raw(tmp_path, TWO, number=5)
    finalize(DAY, cache_dir=tmp_path)
    (root / "recordings.json").write_text('{"last": 1}', "utf-8")

    assert next_recording_number(root, DAY) == 6


def test_a_counter_that_cannot_be_parsed_is_warned_of_and_the_files_decide(tmp_path, caplog):
    write_raw(tmp_path, TWO, number=4)
    finalize(DAY, cache_dir=tmp_path)
    (root_of(tmp_path) / "recordings.json").write_text("{torn", "utf-8")

    with caplog.at_level(logging.WARNING, logger="openalgo.backtest.live"):
        number = next_recording_number(root_of(tmp_path), DAY + dt.timedelta(days=1))

    assert (number, any("cannot be parsed" in r.getMessage() for r in caplog.records)) == (5, True)


def test_a_finalized_file_another_recording_wrote_under_the_same_number_is_a_conflict(
    tmp_path, caplog
):
    finalized_two(tmp_path)
    # A recording numbered 1 again (the counter restored from a copy), started at 11:00.
    later = [sbin("11:00:01", 815.0)]
    raw = RawFile.create(raw_path(root_of(tmp_path), DAY, 1, 0), header_of(start="11:00:00"))
    raw.append(records_of(rows_of(later)))
    raw.close()

    with caplog.at_level(logging.WARNING, logger="openalgo.backtest.live"):
        report = finalize(DAY, cache_dir=tmp_path)

    logged = any("conflict" in record.getMessage() for record in caplog.records)
    assert (report.pending, raw.path.exists(), ltps(tmp_path), logged) == (
        (1,),
        True,
        [812.4, 812.5],
        True,
    )


def test_a_stop_sidecar_that_cannot_be_parsed_leaves_the_recording_crashed(tmp_path):
    write_raw(tmp_path, TWO)
    stop_path(root_of(tmp_path), DAY, 1).write_text("{torn", "utf-8")

    report = finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans
    noted = any("stop sidecar" in note for note in report.notes)
    assert (span.ended_by, span.stop_ns, noted) == (Stop.CRASH, ist_ns("10:00:02"), True)


def test_sidecars_left_once_their_raw_file_is_gone_are_removed_by_the_next_finalize(
    tmp_path, monkeypatch
):
    write_raw(tmp_path, TWO)
    stopped(tmp_path)
    sidecar = stop_path(root_of(tmp_path), DAY, 1)
    unlink = pathlib.Path.unlink

    def held_open_elsewhere(file, missing_ok=False):
        if file == sidecar:
            raise PermissionError(13, "The process cannot access the file", str(file))
        unlink(file, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", held_open_elsewhere)
    finalize(DAY, cache_dir=tmp_path)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)

    report = finalize(DAY, cache_dir=tmp_path)

    raw_folder = sidecar.parent
    assert (report.pending, list(raw_folder.iterdir())) == ((), [])


# ----- readers verify and never write -------------------------------------------------


def test_a_damaged_file_fails_verification_on_read_naming_it(tmp_path):
    finalized_two(tmp_path)
    damage(files_of(tmp_path)[0])

    with pytest.raises(DataError, match=r"2026-11-16\.00000001\.npy.*do not match the hash"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_reader_leaves_a_damaged_file_where_it_is(tmp_path):
    finalized_two(tmp_path)
    damage(files_of(tmp_path)[0])
    before = snapshot(folder_of(tmp_path))

    with pytest.raises(DataError):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert snapshot(folder_of(tmp_path)) == before


def test_a_reader_of_a_sidecar_that_cannot_be_parsed_raises_and_moves_nothing(tmp_path):
    finalized_two(tmp_path)
    files_of(tmp_path)[1].write_text("{not json", "utf-8")
    before = snapshot(folder_of(tmp_path))

    with pytest.raises(DataError, match="cannot be parsed"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert snapshot(folder_of(tmp_path)) == before


def test_a_ticks_file_without_a_sidecar_and_no_raw_file_cannot_be_verified(tmp_path):
    finalized_two(tmp_path)
    files_of(tmp_path)[1].unlink()

    with pytest.raises(DataError, match="no sidecar"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_file_that_cannot_be_read_now_raises_an_os_error_naming_it(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    read_bytes = pathlib.Path.read_bytes

    def busy(path):
        if path.suffix == ".json":
            raise PermissionError(13, "The process cannot access the file")
        return read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", busy)

    with pytest.raises(PermissionError, match=r"2026-11-16\.00000001\.json"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_day_names_the_recordings_of_it_still_pending(tmp_path):
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], number=2, start="10:59:00")

    pending = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending
    finalize(DAY, cache_dir=tmp_path)

    assert (pending, read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending) == ((2,), ())


def test_a_reader_learns_what_is_pending_without_opening_a_raw_file(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], number=2, start="10:59:00")
    opened = []
    real_open, read_bytes = open, pathlib.Path.read_bytes

    def watched_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def watched_read(path):
        opened.append(str(path))
        return read_bytes(path)

    monkeypatch.setattr("builtins.open", watched_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", watched_read)

    pending = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending

    assert (pending, [name for name in opened if name.endswith(".raw")]) == ((2,), [])


def test_a_raw_file_without_a_start_sidecar_is_pending_for_every_instrument(tmp_path):
    finalized_two(tmp_path)
    write_raw(
        tmp_path, [infy("11:00:00", 1_500.0)], number=2, start="10:59:00", instruments=(INFY,)
    )
    start_path(root_of(tmp_path), DAY, 2).unlink()

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending == (2,)


def test_a_day_read_skipping_damaged_files_returns_what_verifies_and_lists_them(tmp_path):
    damaged_first_of_two(tmp_path)

    day = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path, skip_damaged=True)

    assert (day.records["ltp"].tolist(), day.damaged) == ([815.0], (files_of(tmp_path)[0],))


def test_reading_a_day_never_recorded_says_to_record_it(tmp_path):
    with pytest.raises(DataError, match="record that day first"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_reading_a_day_whose_raw_file_waits_says_to_finalize_it(tmp_path):
    write_raw(tmp_path, TWO)

    with pytest.raises(DataError, match="finalize 2026-11-16 first"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_an_instrument_reads_the_same_in_any_case(tmp_path):
    finalized_two(tmp_path)

    assert read_ticks("nse", "sbin", DAY, cache_dir=tmp_path) == TWO


def test_ticks_read_back_carry_their_event_time_volume_bid_and_ask(tmp_path):
    tick = sbin("10:00:01", 812.4, ts_event_ns=ist_ns("10:00:00"), volume=5.0, bid=812.3, ask=812.5)
    write_raw(tmp_path, [tick])
    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", "2026-11-16", cache_dir=tmp_path) == [tick]


def test_missing_values_are_stored_as_0_and_nan_and_read_back_as_none(tmp_path):
    write_raw(tmp_path, [sbin("10:00:01", 812.4)])
    finalize(DAY, cache_dir=tmp_path)

    (stored,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records.tolist()
    (tick,) = read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert stored[4] == NO_EVENT_TIME and all(math.isnan(value) for value in stored[6:])
    assert (tick.ts_event_ns, tick.volume, tick.bid, tick.ask) == (None, None, None, None)


def test_a_volume_of_0_is_stored_and_read_back_as_0(tmp_path):
    write_raw(tmp_path, [sbin("10:00:01", 812.4, volume=0.0)])
    finalize(DAY, cache_dir=tmp_path)

    (tick,) = read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert tick.volume == 0.0


def test_the_queued_time_is_kept_beside_the_receive_time(tmp_path):
    raw = RawFile.create(raw_path(root_of(tmp_path), DAY, 1, 0), header_of())
    tick = sbin("10:00:01.250", 812.4)
    raw.append(
        records_of([tick_row(tick, instrument=0, recording=1, seq=1, queued_ns=ist_ns("10:00:01"))])
    )
    raw.close()
    finalize(DAY, cache_dir=tmp_path)

    records = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records

    assert int(records["ts_recv_ns"][0] - records["ts_queued_ns"][0]) == NS // 4


def test_the_merged_read_follows_the_sequence_across_instruments(tmp_path):
    # Three ticks received at one instant: SBIN, INFY, SBIN, by their sequence numbers.
    at = "10:00:01"
    write_raw(
        tmp_path, [sbin(at, 812.4), infy(at, 1_500.0), sbin(at, 812.5)], instruments=(SBIN, INFY)
    )
    finalize(DAY, cache_dir=tmp_path)
    instruments = [{"exchange": "NSE", "symbol": "INFY"}, {"exchange": "NSE", "symbol": "SBIN"}]

    merged = read_merged_ticks(instruments, DAY, cache_dir=tmp_path)

    assert [tick.ltp for tick in merged.ticks] == [812.4, 1_500.0, 812.5]


def half_finalized_second_recording(tmp_path, monkeypatch) -> None:
    """Recording 1 of SBIN and INFY finalized; recording 2's finalize cut short after
    SBIN's file, so its raw file waits."""
    both = (SBIN, INFY)
    write_raw(tmp_path, [sbin("10:00:01", 801.0), infy("10:00:02", 1501.0)], instruments=both)
    stopped(tmp_path, instruments=both)
    finalize(DAY, cache_dir=tmp_path)
    second = [sbin("11:00:01", 802.0), infy("11:00:02", 1502.0)]
    write_raw(tmp_path, second, number=2, start="11:00:00", instruments=both)
    stopped(tmp_path, number=2, start="11:00:00", instruments=both, stop="11:01:00")
    finalize_instrument = tick_finalize._finalize_instrument

    def cut_after_sbin(root, start, exchange, symbol, *rest):
        if symbol == "INFY":
            raise KeyboardInterrupt
        return finalize_instrument(root, start, exchange, symbol, *rest)

    monkeypatch.setattr(tick_finalize, "_finalize_instrument", cut_after_sbin)
    with pytest.raises(KeyboardInterrupt):
        finalize(DAY, cache_dir=tmp_path)
    monkeypatch.setattr(tick_finalize, "_finalize_instrument", finalize_instrument)


def test_the_merged_read_leaves_out_a_recording_pending_for_any_instrument(tmp_path, monkeypatch):
    half_finalized_second_recording(tmp_path, monkeypatch)
    instruments = [{"exchange": "NSE", "symbol": "SBIN"}, {"exchange": "NSE", "symbol": "INFY"}]

    merged = read_merged_ticks(instruments, DAY, cache_dir=tmp_path)

    assert ([(tick.symbol, tick.ltp) for tick in merged.ticks], merged.left_out) == (
        [("SBIN", 801.0), ("INFY", 1501.0)],
        (2,),
    )


def test_the_merged_read_skipping_damaged_files_lists_them(tmp_path):
    damaged_first_of_two(tmp_path)

    merged = read_merged_ticks(
        [{"exchange": "NSE", "symbol": "SBIN"}], DAY, cache_dir=tmp_path, skip_damaged=True
    )

    assert ([tick.ltp for tick in merged.ticks], merged.damaged) == (
        [815.0],
        (files_of(tmp_path)[0],),
    )


def test_a_moment_inside_a_crashed_recordings_span_has_unknown_coverage(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)

    coverage = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).coverage(ist_ns("10:00:01"))

    assert coverage is Coverage.UNKNOWN


def test_a_moment_outside_every_span_is_not_recorded(tmp_path):
    finalized_two(tmp_path)

    coverage = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).coverage(ist_ns("12:00:00"))

    assert coverage is Coverage.NOT_RECORDED


# ----- repair -------------------------------------------------------------------------


def damaged_first_of_two(tmp_path) -> None:
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], number=2, start="10:59:00")
    stopped(tmp_path, number=2, start="10:59:00", stop="11:01:00")
    finalize(DAY, cache_dir=tmp_path)
    damage(files_of(tmp_path)[0])


def test_repair_moves_a_damaged_file_aside_and_the_days_other_recordings_read(tmp_path):
    damaged_first_of_two(tmp_path)

    report = repair(DAY, cache_dir=tmp_path)

    moved = report.moved
    assert (len(moved), all(path.exists() for path in moved), ltps(tmp_path)) == (2, True, [815.0])


def held_open_by_a_reader(monkeypatch, held):
    replace = os.replace

    def held_open(source, target):
        if pathlib.Path(source) == held:
            raise PermissionError(13, "The process cannot access the file", str(source))
        replace(source, target)

    monkeypatch.setattr(os, "replace", held_open)


def test_repair_reports_a_move_that_failed_and_still_makes_the_others(tmp_path, monkeypatch):
    damaged_first_of_two(tmp_path)
    ticks_path, sidecar_path = files_of(tmp_path)
    held_open_by_a_reader(monkeypatch, sidecar_path)

    report = repair(DAY, cache_dir=tmp_path)

    (failed, why) = report.failed[0]
    assert (failed, ticks_path.exists(), len(report.moved)) == (sidecar_path, False, 1)
    assert "delete" not in why.lower()


def test_repair_logs_every_file_it_moved_and_every_move_that_failed(tmp_path, monkeypatch, caplog):
    damaged_first_of_two(tmp_path)
    ticks_path, sidecar_path = files_of(tmp_path)
    held_open_by_a_reader(monkeypatch, sidecar_path)

    with caplog.at_level(logging.WARNING, logger="openalgo.backtest.live"):
        repair(DAY, cache_dir=tmp_path)

    messages = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert any(level == "WARNING" and "was moved aside" in text for level, text in messages)
    assert any(level == "ERROR" and sidecar_path.name in text for level, text in messages)


def test_a_recording_whose_damaged_file_was_moved_aside_finalizes_again_from_its_raw_file(
    tmp_path,
):
    path = write_raw(tmp_path, TWO)
    kept = shutil.copy(path, tmp_path / "copy.raw")
    finalize(DAY, cache_dir=tmp_path)
    damage(files_of(tmp_path)[0])
    shutil.copy(kept, path)
    repair(DAY, cache_dir=tmp_path)

    finalize(DAY, cache_dir=tmp_path)

    assert (ltps(tmp_path), path.exists()) == ([812.4, 812.5], False)
