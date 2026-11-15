"""The recorder's files: raw files of fixed-size records that a crash can tear, finalize
turning them into one verified day file per instrument with its index, readers that
verify and never write, and the repairs (moving a damaged file aside, rebuilding an
index) that only a locked finalize makes. Raw files are written here as a recording
writes them, so these tests need no feed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import shutil

import numpy as np
import pytest
from recorder_fakes import DAY, NS, ist_ns, raw_files

import openalgo.backtest._live._tick_days as tick_days
from openalgo.backtest._live._tick_files import (
    NO_EVENT_TIME,
    TICK_DTYPE,
    RawFile,
    instrument_folder,
    read_raw,
    recording_number,
    records_of,
    tick_row,
    ticks_root,
)
from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.recorder import (
    Coverage,
    Ending,
    finalize,
    read_day,
    read_merged_ticks,
    read_ticks,
)
from openalgo.backtest.errors import DataError

#: The hash of a day file covers this versioned header, then the records' bytes.
HASH_HEADER = b"openalgo.backtest recorded ticks v2\n"
#: Two recordings' ids, as a recording names itself (16 hex digits).
EARLY, LATE = "00000000000000a1", "00000000000000b2"


def sbin(clock: str, ltp: float, **fields) -> Tick:
    return Tick("SBIN", "NSE", ltp, ist_ns(clock), **fields)


def folder_of(tmp_path, symbol="SBIN"):
    return instrument_folder(ticks_root(tmp_path), "NSE", symbol)


def write_raw(tmp_path, ticks, *, symbol="SBIN", recording=EARLY, start="10:00:00", seq=1):
    """A raw file as a recording leaves it: the header, then one record per tick, with
    sequence numbers from `seq`."""
    raw = RawFile.create(
        folder_of(tmp_path, symbol), DAY, recording=recording, start_ns=ist_ns(start)
    )
    number = recording_number(recording)
    rows = [
        tick_row(tick, recording=number, seq=seq + n, queued_ns=tick.ts_recv_ns)
        for n, tick in enumerate(ticks)
    ]
    if rows:
        raw.append(records_of(rows))
    raw.close()
    return raw.path


def index_path(tmp_path, symbol="SBIN"):
    return folder_of(tmp_path, symbol) / "index.json"


def index_of(tmp_path, symbol="SBIN") -> dict:
    return json.loads(index_path(tmp_path, symbol).read_text("utf-8"))


def day_file(tmp_path):
    (path,) = folder_of(tmp_path).glob("*.npy")
    return path


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
    finalize(DAY, cache_dir=tmp_path)


# ----- raw files ----------------------------------------------------------------------


def test_a_torn_last_record_is_cut_off_on_read(tmp_path):
    path = write_raw(tmp_path, TWO)
    with open(path, "ab") as handle:
        handle.write(b"\x01" * 10)  # a crash in the middle of a record

    content = read_raw(path)

    assert (len(content.records), content.torn_bytes) == (2, 10)


def test_finalize_drops_a_torn_last_record_and_reports_its_bytes(tmp_path):
    path = write_raw(tmp_path, TWO)
    with open(path, "ab") as handle:
        handle.write(b"\x01" * (TICK_DTYPE.itemsize - 1))

    (written,) = finalize(DAY, cache_dir=tmp_path).files

    assert (written.ticks, written.torn_bytes) == (2, TICK_DTYPE.itemsize - 1)


def test_after_a_write_cut_short_the_next_append_first_cuts_the_file_to_whole_records(
    tmp_path, monkeypatch
):
    raw = RawFile.create(folder_of(tmp_path), DAY, recording=EARLY, start_ns=ist_ns("10:00:00"))
    rows = [
        tick_row(tick, recording=recording_number(EARLY), seq=n, queued_ns=0)
        for n, tick in enumerate(TWO)
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
    assert (content.records["ltp"].tolist(), content.torn_bytes) == ([812.4, 812.5], 0)


def test_a_record_is_72_bytes_of_fixed_fields_in_utc_nanoseconds_and_rupees():
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


# ----- finalize -----------------------------------------------------------------------


def test_finalize_turns_the_days_raw_file_into_a_day_file_that_reads_back(tmp_path):
    finalized_two(tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == TWO


def test_finalize_removes_the_raw_file(tmp_path):
    path = write_raw(tmp_path, TWO)

    finalize(DAY, cache_dir=tmp_path)

    assert not path.exists()


def test_the_index_holds_the_files_hash_over_its_records(tmp_path):
    finalized_two(tmp_path)

    records = np.load(day_file(tmp_path), allow_pickle=False)
    expected = "sha256:" + hashlib.sha256(HASH_HEADER + records.tobytes()).hexdigest()
    assert index_of(tmp_path)["days"]["2026-11-16"]["sha256"] == expected


def test_the_index_holds_the_tick_count_and_the_first_and_last_receive_time(tmp_path):
    finalized_two(tmp_path)

    entry = index_of(tmp_path)["days"]["2026-11-16"]

    assert (entry["ticks"], entry["first_recv_ns"], entry["last_recv_ns"]) == (
        2,
        ist_ns("10:00:01"),
        ist_ns("10:00:02"),
    )


def test_a_recording_finalized_from_its_raw_file_alone_ended_by_a_crash_at_its_last_tick(
    tmp_path,
):
    finalized_two(tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.recording, span.start_ns, span.stop_ns, span.ended_by, span.outages) == (
        EARLY,
        ist_ns("10:00:00"),
        ist_ns("10:00:02"),
        Ending.CRASH,
        None,
    )


def test_a_crashed_recording_without_a_tick_spans_no_time(tmp_path):
    write_raw(tmp_path, [], start="10:00:00")
    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.start_ns, span.stop_ns) == (ist_ns("10:00:00"), ist_ns("10:00:00"))


def test_a_later_recordings_raw_file_is_appended_after_the_first(tmp_path):
    finalized_two(tmp_path)
    later = sbin("11:00:00", 815.0)
    write_raw(tmp_path, [later], recording=LATE, start="10:59:00")

    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == [*TWO, later]


def test_a_merged_day_keeps_one_day_file(tmp_path):
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")

    finalize(DAY, cache_dir=tmp_path)

    assert len(list(folder_of(tmp_path).glob("*.npy"))) == 1


def test_a_record_found_in_two_raw_files_of_its_recording_is_kept_once(tmp_path):
    write_raw(tmp_path, TWO)
    write_raw(tmp_path, TWO[1:], seq=2)  # the batch a recording wrote again elsewhere

    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == TWO


def test_a_raw_file_finalized_before_a_crash_removed_it_is_not_merged_twice(tmp_path):
    path = write_raw(tmp_path, TWO)
    kept = shutil.copy(path, tmp_path / "copy.raw")
    finalize(DAY, cache_dir=tmp_path)
    shutil.copy(kept, path)  # the crash came after the index was saved

    finalize(DAY, cache_dir=tmp_path)

    assert (len(read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)), path.exists()) == (2, False)


def test_a_day_without_a_tick_reads_back_as_an_empty_list(tmp_path):
    write_raw(tmp_path, [])

    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == []


def test_finalize_moves_aside_a_file_that_is_not_a_raw_file(tmp_path):
    path = folder_of(tmp_path) / "2026-11-16.0123456789abcdef.raw"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not ticks")

    report = finalize(DAY, cache_dir=tmp_path)

    moved = list(path.parent.glob(f"{path.name}.quarantined-*"))
    assert (report.files, len(moved), "not a raw tick file" in report.notes[0]) == ((), 1, True)


@pytest.mark.parametrize("symbol", ["M&M", "BAJAJ-AUTO", "NIFTY 50", "SBIN/EQ"])
def test_a_symbol_with_characters_unsafe_in_a_path_round_trips(tmp_path, symbol):
    tick = Tick(symbol, "NSE", 100.0, ist_ns("10:00:01"))
    write_raw(tmp_path, [tick], symbol=symbol)

    (written,) = finalize(DAY, cache_dir=tmp_path).files

    assert (written.symbol, read_ticks("NSE", symbol, DAY, cache_dir=tmp_path)) == (symbol, [tick])


def test_an_old_day_file_that_cannot_be_removed_stays_until_a_later_finalize(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    old = day_file(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")
    unlink = pathlib.Path.unlink

    def held_open_elsewhere(path, missing_ok=False):
        if path == old:
            raise PermissionError(13, "The process cannot access the file", str(path))
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", held_open_elsewhere)
    report = finalize(DAY, cache_dir=tmp_path)
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    stayed = old.exists()

    finalize(DAY, cache_dir=tmp_path)

    assert (stayed, any(old.name in note for note in report.notes), old.exists()) == (
        True,
        True,
        False,
    )


def test_the_day_reads_back_whole_while_an_old_day_file_stays(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    old = day_file(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")
    unlink = pathlib.Path.unlink

    def held_open_elsewhere(path, missing_ok=False):
        if path == old:
            raise PermissionError(13, "The process cannot access the file", str(path))
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", held_open_elsewhere)

    finalize(DAY, cache_dir=tmp_path)

    assert len(read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == 3


# ----- spans, pending files, the merged read -------------------------------------------


def test_a_day_says_when_a_raw_file_of_it_is_still_pending(tmp_path):
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")

    pending = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending
    finalize(DAY, cache_dir=tmp_path)

    assert (pending, read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).pending) == (True, False)


def test_a_moment_inside_a_span_whose_outages_are_not_known_has_unknown_coverage(tmp_path):
    finalized_two(tmp_path)

    coverage = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).coverage(ist_ns("10:00:01"))

    assert coverage is Coverage.UNKNOWN


def test_a_moment_outside_every_known_span_is_not_recorded(tmp_path):
    finalized_two(tmp_path)

    coverage = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).coverage(ist_ns("12:00:00"))

    assert coverage is Coverage.NOT_RECORDED


def test_the_merged_read_follows_the_sequence_across_instruments(tmp_path):
    # Three ticks received at one instant: SBIN, INFY, SBIN, by their sequence numbers.
    write_raw(tmp_path, [sbin("10:00:01", 812.4)], seq=1)
    write_raw(tmp_path, [Tick("INFY", "NSE", 1_500.0, ist_ns("10:00:01"))], symbol="INFY", seq=2)
    write_raw(tmp_path, [sbin("10:00:01", 812.5)], seq=3)
    finalize(DAY, cache_dir=tmp_path)
    instruments = [{"exchange": "NSE", "symbol": "INFY"}, {"exchange": "NSE", "symbol": "SBIN"}]

    merged = read_merged_ticks(instruments, DAY, cache_dir=tmp_path)

    assert [tick.ltp for tick in merged] == [812.4, 1_500.0, 812.5]


# ----- readers verify and never write -------------------------------------------------


def test_a_changed_day_file_fails_verification_on_read(tmp_path):
    finalized_two(tmp_path)
    damage(day_file(tmp_path))

    with pytest.raises(DataError, match="does not match the hash"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_reader_leaves_a_changed_day_file_and_the_index_as_they_are(tmp_path):
    finalized_two(tmp_path)
    damage(day_file(tmp_path))
    before = snapshot(folder_of(tmp_path))

    with pytest.raises(DataError):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert snapshot(folder_of(tmp_path)) == before


def test_finalize_moves_a_changed_day_file_aside_never_deleting_it(tmp_path):
    finalized_two(tmp_path)
    path = day_file(tmp_path)
    damage(path)

    report = finalize(DAY, cache_dir=tmp_path)

    moved = list(path.parent.glob(f"{path.name}.quarantined-*"))
    assert (len(moved), any("does not match the hash" in note for note in report.notes)) == (
        1,
        True,
    )


def test_a_day_whose_file_was_moved_aside_is_no_longer_read(tmp_path):
    finalized_two(tmp_path)
    damage(day_file(tmp_path))
    finalize(DAY, cache_dir=tmp_path)

    with pytest.raises(DataError, match="no ticks of NSE:SBIN were recorded on 2026-11-16"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_day_file_replaced_by_other_ticks_fails_verification(tmp_path):
    finalized_two(tmp_path)
    with open(day_file(tmp_path), "wb") as handle:
        np.save(handle, np.zeros(2, dtype=TICK_DTYPE), allow_pickle=False)

    with pytest.raises(DataError, match="does not match the hash"):
        read_day("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_reader_of_an_unparseable_index_raises_and_moves_nothing(tmp_path):
    finalized_two(tmp_path)
    index_path(tmp_path).write_text("{not json", "utf-8")
    before = snapshot(folder_of(tmp_path))

    with pytest.raises(DataError, match="cannot be parsed"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert snapshot(folder_of(tmp_path)) == before


def test_finalize_moves_an_unparseable_index_aside_and_rebuilds_it_from_the_day_files(
    tmp_path, caplog
):
    finalized_two(tmp_path)
    index_path(tmp_path).write_text("{not json", "utf-8")

    with caplog.at_level(logging.WARNING, logger="openalgo.backtest.live"):
        finalize(DAY, cache_dir=tmp_path)

    moved = list(folder_of(tmp_path).glob("index.json.quarantined-*"))
    rebuilt = any("rebuilt" in record.getMessage() for record in caplog.records)
    assert (len(moved), rebuilt, read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == (
        1,
        True,
        TWO,
    )


def test_a_rebuilt_index_marks_the_spans_and_outages_of_the_day_unknown(tmp_path):
    finalized_two(tmp_path)
    index_path(tmp_path).write_text("[]", "utf-8")
    finalize(DAY, cache_dir=tmp_path)

    (span,) = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).spans

    assert (span.recording, span.start_ns, span.stop_ns, span.ended_by, span.outages) == (
        EARLY,
        None,
        None,
        Ending.UNKNOWN,
        None,
    )


def test_a_rebuilt_index_takes_the_newest_day_file_of_a_day(tmp_path):
    finalized_two(tmp_path)
    two_ticks = day_file(tmp_path).read_bytes()
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")
    finalize(DAY, cache_dir=tmp_path)
    current = day_file(tmp_path)
    # An older day file of the day that a removal missed, written ten seconds earlier.
    stale = current.with_name("2026-11-16.00000000000000ff.npy")
    stale.write_bytes(two_ticks)
    earlier_ns = current.stat().st_mtime_ns - 10 * NS
    os.utime(stale, ns=(earlier_ns, earlier_ns))
    index_path(tmp_path).write_text("{not json", "utf-8")

    finalize(DAY, cache_dir=tmp_path)

    assert len(read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == 3


def test_an_index_entry_pointing_outside_its_folder_fails_verification(tmp_path):
    finalized_two(tmp_path)
    index = index_of(tmp_path)
    index["days"]["2026-11-16"]["file"] = "../../elsewhere.npy"
    index_path(tmp_path).write_text(json.dumps(index), "utf-8")

    with pytest.raises(DataError, match="cannot be parsed"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_file_that_cannot_be_opened_now_raises_an_os_error_naming_it(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    read_bytes = pathlib.Path.read_bytes

    def busy(path):
        if path.name == "index.json":
            raise PermissionError(13, "The process cannot access the file")
        return read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", busy)

    with pytest.raises(PermissionError, match="index.json"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_file_that_cannot_be_opened_now_is_neither_moved_nor_lost(tmp_path, monkeypatch):
    finalized_two(tmp_path)
    before = snapshot(folder_of(tmp_path))
    read_bytes = pathlib.Path.read_bytes

    def busy(path):
        if path.suffix == ".npy":
            raise PermissionError(13, "The process cannot access the file")
        return read_bytes(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", busy)
    with pytest.raises(PermissionError):
        finalize(DAY, cache_dir=tmp_path)
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)

    assert (snapshot(folder_of(tmp_path)), read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == (
        before,
        TWO,
    )


def test_a_reader_whose_day_file_a_finalize_replaced_meanwhile_reads_the_new_one(
    tmp_path, monkeypatch
):
    finalized_two(tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)], recording=LATE, start="10:59:00")
    read_index = tick_days.read_index

    def then_a_finalize(*args):
        index = read_index(*args)
        monkeypatch.setattr(tick_days, "read_index", read_index)
        finalize(DAY, cache_dir=tmp_path)  # it removes the day file this index names
        return index

    monkeypatch.setattr(tick_days, "read_index", then_a_finalize)

    assert len(read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)) == 3


def test_reading_a_day_never_recorded_says_to_record_it(tmp_path):
    with pytest.raises(DataError, match="record that day first"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_reading_a_day_whose_raw_file_waits_says_to_finalize_it(tmp_path):
    write_raw(tmp_path, TWO)

    with pytest.raises(DataError, match="finalize 2026-11-16 first"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_ticks_read_back_carry_their_event_time_volume_bid_and_ask(tmp_path):
    tick = sbin("10:00:01", 812.4, ts_event_ns=ist_ns("10:00:00"), volume=5.0, bid=812.3, ask=812.5)
    write_raw(tmp_path, [tick])
    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", "2026-11-16", cache_dir=tmp_path) == [tick]


def test_the_queued_time_is_kept_beside_the_receive_time(tmp_path):
    raw = RawFile.create(folder_of(tmp_path), DAY, recording=EARLY, start_ns=ist_ns("10:00:00"))
    tick = sbin("10:00:01.250", 812.4)
    row = tick_row(tick, recording=recording_number(EARLY), seq=1, queued_ns=ist_ns("10:00:01"))
    raw.append(records_of([row]))
    raw.close()
    finalize(DAY, cache_dir=tmp_path)

    records = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records

    assert int(records["ts_recv_ns"][0] - records["ts_queued_ns"][0]) == NS // 4


def test_raw_files_are_found_wherever_a_recording_put_them(tmp_path):
    write_raw(tmp_path, TWO)

    assert len(raw_files(tmp_path)) == 1
