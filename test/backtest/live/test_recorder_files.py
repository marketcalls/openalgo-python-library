"""The recorder's files: raw files of fixed-size records that a crash can tear, finalize
turning them into one verified day file per instrument with its index, and reads that
verify the hash and move a changed file aside. Raw files are written here as a recording
writes them, so these tests need no feed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil

import numpy as np
import pytest
from recorder_fakes import DAY, NS, ist_ns

from openalgo.backtest._live._tick_files import (
    NO_EVENT_TIME,
    TICK_DTYPE,
    RawFile,
    raw_path,
    read_raw,
    records_of,
    tick_row,
    ticks_root,
)
from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.recorder import finalize, read_day, read_ticks
from openalgo.backtest.errors import DataError

#: The hash of a day file covers this versioned header, then the records' bytes.
HASH_HEADER = b"openalgo.backtest recorded ticks v1\n"


def sbin(clock: str, ltp: float, **fields) -> Tick:
    return Tick("SBIN", "NSE", ltp, ist_ns(clock), **fields)


def write_raw(tmp_path, ticks: list[Tick], *, exchange="NSE", symbol="SBIN"):
    """A raw file as a recording leaves it: the header, then one record per tick."""
    raw = RawFile.create(raw_path(ticks_root(tmp_path), exchange, symbol, DAY))
    if ticks:
        raw.append(records_of(tick_row(tick, tick.ts_recv_ns) for tick in ticks))
    raw.close()
    return raw.path


def index_of(tmp_path, symbol="SBIN") -> dict:
    path = raw_path(ticks_root(tmp_path), "NSE", symbol, DAY).parent / "index.json"
    return json.loads(path.read_text("utf-8"))


def day_file(tmp_path):
    (path,) = raw_path(ticks_root(tmp_path), "NSE", "SBIN", DAY).parent.glob("*.npy")
    return path


TWO = [sbin("10:00:01", 812.4, volume=1_000.0), sbin("10:00:02", 812.5, volume=1_100.0)]


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


def test_a_record_is_56_bytes_of_fixed_fields_in_utc_nanoseconds_and_rupees():
    assert [(name, TICK_DTYPE[name].str) for name in TICK_DTYPE.names] == [
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

    assert stored[2] == NO_EVENT_TIME and all(math.isnan(value) for value in stored[4:])
    assert (tick.ts_event_ns, tick.volume, tick.bid, tick.ask) == (None, None, None, None)


# ----- finalize -----------------------------------------------------------------------


def test_finalize_turns_the_days_raw_file_into_a_day_file_that_reads_back(tmp_path):
    write_raw(tmp_path, TWO)

    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == TWO


def test_finalize_removes_the_raw_file(tmp_path):
    path = write_raw(tmp_path, TWO)

    finalize(DAY, cache_dir=tmp_path)

    assert not path.exists()


def test_the_index_holds_the_files_hash_over_its_records(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)

    records = np.load(day_file(tmp_path), allow_pickle=False)
    expected = "sha256:" + hashlib.sha256(HASH_HEADER + records.tobytes()).hexdigest()
    assert index_of(tmp_path)["days"]["2026-11-16"]["sha256"] == expected


def test_the_index_holds_the_tick_count_and_the_first_and_last_receive_time(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)

    entry = index_of(tmp_path)["days"]["2026-11-16"]

    assert (entry["ticks"], entry["first_recv_ns"], entry["last_recv_ns"]) == (
        2,
        ist_ns("10:00:01"),
        ist_ns("10:00:02"),
    )


def test_a_day_finalized_from_raw_files_alone_has_no_known_outages(tmp_path):
    write_raw(tmp_path, TWO)

    finalize(DAY, cache_dir=tmp_path)

    assert read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).outages is None


def test_a_second_raw_file_of_the_day_is_appended_after_the_first(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    later = sbin("11:00:00", 815.0)
    write_raw(tmp_path, [later])

    finalize(DAY, cache_dir=tmp_path)

    assert read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path) == [*TWO, later]


def test_a_merged_day_keeps_one_day_file(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    write_raw(tmp_path, [sbin("11:00:00", 815.0)])

    finalize(DAY, cache_dir=tmp_path)

    assert len(list(day_file(tmp_path).parent.glob("*.npy"))) == 1


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
    path = raw_path(ticks_root(tmp_path), "NSE", "SBIN", DAY)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not ticks")

    report = finalize(DAY, cache_dir=tmp_path)

    moved = list(path.parent.glob("2026-11-16.raw.quarantined-*"))
    assert (report.files, len(moved), "not a raw tick file" in report.notes[0]) == ((), 1, True)


@pytest.mark.parametrize("symbol", ["M&M", "BAJAJ-AUTO", "NIFTY 50", "SBIN/EQ"])
def test_a_symbol_with_characters_unsafe_in_a_path_round_trips(tmp_path, symbol):
    tick = Tick(symbol, "NSE", 100.0, ist_ns("10:00:01"))
    write_raw(tmp_path, [tick], symbol=symbol)

    (written,) = finalize(DAY, cache_dir=tmp_path).files

    assert (written.symbol, read_ticks("NSE", symbol, DAY, cache_dir=tmp_path)) == (symbol, [tick])


# ----- reads verify -------------------------------------------------------------------


def test_a_changed_day_file_fails_verification_on_read(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    blob = bytearray(day_file(tmp_path).read_bytes())
    blob[-1] ^= 0xFF  # one byte of the last record's ask
    day_file(tmp_path).write_bytes(bytes(blob))

    with pytest.raises(DataError, match="does not match the hash"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_changed_day_file_is_moved_aside_never_deleted(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    path = day_file(tmp_path)
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")

    with pytest.raises(DataError):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert len(list(path.parent.glob(f"{path.name}.quarantined-*"))) == 1


def test_a_changed_day_file_is_no_longer_read_after_it_was_moved_aside(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    path = day_file(tmp_path)
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    with pytest.raises(DataError):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    with pytest.raises(DataError, match="no ticks of NSE:SBIN were recorded on 2026-11-16"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_a_day_file_replaced_by_other_ticks_fails_verification(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    other = np.zeros(2, dtype=TICK_DTYPE)
    with open(day_file(tmp_path), "wb") as handle:
        np.save(handle, other, allow_pickle=False)

    with pytest.raises(DataError, match="does not match the hash"):
        read_day("NSE", "SBIN", DAY, cache_dir=tmp_path)


def test_an_unreadable_index_is_moved_aside_and_the_day_is_not_read(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    index = day_file(tmp_path).parent / "index.json"
    index.write_text("{not json", "utf-8")

    with pytest.raises(DataError, match="the tick index .* failed verification"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)

    assert len(list(index.parent.glob("index.json.quarantined-*"))) == 1


def test_an_index_entry_pointing_outside_its_folder_fails_verification(tmp_path):
    write_raw(tmp_path, TWO)
    finalize(DAY, cache_dir=tmp_path)
    index_path = day_file(tmp_path).parent / "index.json"
    index = json.loads(index_path.read_text("utf-8"))
    index["days"]["2026-11-16"]["file"] = "../../elsewhere.npy"
    index_path.write_text(json.dumps(index), "utf-8")

    with pytest.raises(DataError, match="its index entry is malformed"):
        read_ticks("NSE", "SBIN", DAY, cache_dir=tmp_path)


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
    raw = RawFile.create(raw_path(ticks_root(tmp_path), "NSE", "SBIN", DAY))
    raw.append(records_of([tick_row(sbin("10:00:01.250", 812.4), ist_ns("10:00:01"))]))
    raw.close()
    finalize(DAY, cache_dir=tmp_path)

    records = read_day("NSE", "SBIN", DAY, cache_dir=tmp_path).records

    assert int(records["ts_recv_ns"][0] - records["ts_queued_ns"][0]) == NS // 4
