"""The local cache of raw rows (FR-DATA-08): what is stored, when a session counts as
complete, how damage is repaired, and what a load through the cache returns.

Every test writes under pytest's `tmp_path`; no test touches the user's cache. The
reference series is 5-minute NSE bars (75 per session, 09:15 to 15:30 IST) from
Monday 2024-01-01; the fake calendar has weekday sessions.
"""

import datetime as dt
import hashlib
import json
import struct

import numpy as np
import pytest
from data_fixtures.fakes import (
    NS,
    FakeCalendar,
    FakeClient,
    at_ist,
    broker_shape,
    daily_rows,
    days_between,
    historify_shape,
    ist_date_of,
    open_ns,
    server_load,
    truth_rows,
)

from openalgo.backtest.data import resolve_cache_dir
from openalgo.backtest.data.cache import (
    CACHE_ENV,
    SETTLE_MARGIN_NANOS,
    RawCache,
    SeriesKey,
    is_settled,
    safe_name,
)
from openalgo.backtest.data.fetching import rows_to_records
from openalgo.backtest.errors import ConfigError, DataError, NoDataError

CAL = FakeCalendar()
JAN_1 = dt.date(2024, 1, 1)  # a Monday
JAN_5 = dt.date(2024, 1, 5)
JAN_8 = dt.date(2024, 1, 8)
JAN_12 = dt.date(2024, 1, 12)
JAN_19 = dt.date(2024, 1, 19)
LONG_AFTER = at_ist(dt.date(2024, 6, 3), 18)  # every session of the tests has settled
BARS_PER_DAY = 75
KEY = SeriesKey("openalgo-db", "NSE", "X", "5m")
SIX_HOURS = 6 * 3600


def _rows(first, last):
    return truth_rows(CAL, "NSE", days_between(first, last), step=300)


def _db(rows, **options):
    return FakeClient(db={("X", "NSE", "5m"): historify_shape(rows)}, **options)


def _broker(rows, **options):
    return FakeClient(api={("X", "NSE", "5m"): broker_shape(rows)}, **options)


def _load(client, tmp_path, first=JAN_1, last=JAN_5, source="db", now_ns=LONG_AFTER, **options):
    return server_load(
        client, "5m", first, last, calendar=CAL, now_ns=now_ns, source=source,
        cache_dir=tmp_path, **options,
    )  # fmt: skip


def _windows(client, source="db"):
    return [(first, last) for _, _, first, last in client.history_calls(source)]


def _series_dir(tmp_path, namespace="openalgo-db"):
    return tmp_path / "v2" / namespace / "exchange=NSE" / "symbol=X" / "interval=5m"


# --- where the cache lives ----------------------------------------------------------


def test_cache_dir_argument_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "from_env"))
    assert resolve_cache_dir(tmp_path / "from_arg") == tmp_path / "from_arg"


def test_cache_dir_comes_from_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "from_env"))
    assert CACHE_ENV == "OPENALGO_BT_CACHE"
    assert resolve_cache_dir() == tmp_path / "from_env"


def test_cache_dir_defaults_to_a_folder_under_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(CACHE_ENV, raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert resolve_cache_dir() == tmp_path / ".openalgo" / "backtest" / "cache"


def test_cache_directory_that_cannot_be_created_is_a_config_error(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    with pytest.raises(ConfigError, match="cache directory .* cannot be used"):
        _load(_db(_rows(JAN_1, JAN_5)), blocker)


def test_cache_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    result = server_load(
        _db(_rows(JAN_1, JAN_5)), "5m", JAN_1, JAN_5, calendar=CAL, now_ns=LONG_AFTER,
        source="db", cache=False,
    )  # fmt: skip
    assert result.meta.cache == "off"
    assert list(tmp_path.iterdir()) == []


# --- what is stored: the rows as the source returned them ---------------------------


def test_series_is_stored_as_one_array_file_per_month_with_an_index(tmp_path):
    _load(_db(_rows(JAN_1, dt.date(2024, 2, 2))), tmp_path, last=dt.date(2024, 2, 2))
    assert sorted(path.name for path in _series_dir(tmp_path).iterdir()) == [
        "2024-01.npy", "2024-02.npy", "index.json",
    ]  # fmt: skip


def test_partition_holds_raw_epoch_seconds_and_the_open_interest_the_server_sent(tmp_path):
    _load(_db(_rows(JAN_1, JAN_5)), tmp_path)
    records = np.load(_series_dir(tmp_path) / "2024-01.npy", allow_pickle=False)
    assert records.dtype.names == ("timestamp", "open", "high", "low", "close", "volume", "oi")
    assert records["timestamp"][0] == at_ist(JAN_1, 9, 15) // NS
    assert records["oi"][0] == 0.0  # not NaN: turning 0 into "unknown" is the load's job


def test_index_records_a_hash_per_file_and_the_fetch_time_per_date(tmp_path):
    fetched_at = at_ist(JAN_8, 18)
    _load(_db(_rows(JAN_1, JAN_5)), tmp_path, now_ns=fetched_at)
    index = json.loads((_series_dir(tmp_path) / "index.json").read_text(encoding="utf-8"))
    january = index["partitions"]["2024-01"]
    assert january["sha256"].startswith("sha256:") and len(january["sha256"]) == 7 + 64
    assert january["rows"] == 5 * BARS_PER_DAY
    assert january["fetched"] == {f"2024-01-0{day}": fetched_at // NS for day in range(1, 6)}
    assert (index["source"], index["broker"]) == ("db", None)


def test_sources_and_brokers_are_cached_apart(tmp_path):
    rows = _rows(JAN_1, JAN_5)
    _load(_db(rows), tmp_path, source="db")
    _load(_broker(rows, broker="zerodha"), tmp_path, source="api")
    _load(_broker(rows, broker="angel"), tmp_path, source="api")
    assert sorted(path.name for path in (tmp_path / "v2").iterdir()) == [
        "openalgo-api-angel", "openalgo-api-zerodha", "openalgo-db",
    ]  # fmt: skip


def test_no_temporary_file_is_left_behind_after_a_write(tmp_path):
    _load(_db(_rows(JAN_1, JAN_5)), tmp_path)
    assert [path for path in tmp_path.rglob("*") if path.name.endswith(".tmp")] == []


def test_failed_write_leaves_the_previous_partition_in_place(tmp_path, monkeypatch):
    store = RawCache(tmp_path)
    week_1 = rows_to_records(historify_shape(_rows(JAN_1, JAN_5)))
    store.write(KEY, week_1, [(JAN_1, JAN_5)], 1_704_800_000, source="db", broker=None)

    def disk_full(handle, records, allow_pickle):
        handle.write(b"half a file")
        raise OSError("no space left on device")

    monkeypatch.setattr(np, "save", disk_full)
    week_2 = rows_to_records(historify_shape(_rows(JAN_8, JAN_12)))
    with pytest.raises(OSError, match="no space left"):
        store.write(KEY, week_2, [(JAN_8, JAN_12)], 1_705_400_000, source="db", broker=None)
    monkeypatch.undo()
    found = store.read(KEY, JAN_1, JAN_12)
    assert (len(found.records), found.notes) == (5 * BARS_PER_DAY, [])


@pytest.mark.parametrize(
    "symbol, folder_name",
    [("M&M", "symbol=M%26M"), ("NIFTY 50", "symbol=NIFTY%2050"), ("../x", "symbol=%2E%2E%2Fx"),
     ("BAJAJ-AUTO", "symbol=BAJAJ-AUTO"), ("CON", "symbol=CON")],
)  # fmt: skip
def test_symbol_becomes_a_safe_reversible_directory_name(tmp_path, symbol, folder_name):
    key = SeriesKey("openalgo-db", "NSE", symbol, "D")
    assert RawCache(tmp_path).series_dir(key).parent.name == folder_name
    assert safe_name("a-b_c9") == "a-b_c9"


def test_write_replaces_the_dates_that_received_rows_and_keeps_the_others(tmp_path):
    store = RawCache(tmp_path)
    week = rows_to_records(historify_shape(_rows(JAN_1, JAN_5)))
    store.write(KEY, week, [(JAN_1, JAN_5)], 1_704_800_000, source="db", broker=None)
    monday_only = rows_to_records(historify_shape(_rows(JAN_1, JAN_1)[:10]))
    store.write(KEY, monday_only, [(JAN_1, JAN_5)], 1_705_400_000, source="db", broker=None)
    found = store.read(KEY, JAN_1, JAN_5)
    assert len(found.records) == 10 + 4 * BARS_PER_DAY  # Monday replaced, Tuesday to Friday kept
    assert found.fetched_at[JAN_1] == 1_705_400_000
    assert found.fetched_at[JAN_5] == 1_704_800_000


# --- when a cached session counts as complete ---------------------------------------


@pytest.mark.parametrize(
    "source, fetched, settled",
    [
        ("db", None, False),
        ("db", (JAN_5, 15, 30, 5), False),  # five seconds after the close
        ("db", (JAN_5, 21, 29, 59), False),  # one second before the margin has passed
        ("db", (JAN_5, 21, 30, 0), True),  # six hours after the close, the same evening
        ("api", (JAN_5, 23, 59, 59), False),  # a broker is never trusted on the same IST date
        ("api", (dt.date(2024, 1, 6), 0, 0, 0), True),
    ],
    ids=["never fetched", "just closed", "inside the margin", "past the margin",
         "broker same date", "broker next date"],
)  # fmt: skip
def test_session_is_settled_only_a_margin_after_its_close(source, fetched, settled):
    close_ns = at_ist(JAN_5, 15, 30)
    fetched_at_s = None if fetched is None else at_ist(*fetched) // NS
    assert SETTLE_MARGIN_NANOS == SIX_HOURS * NS
    assert is_settled(JAN_5, fetched_at_s, close_ns, source) is settled


def test_load_of_settled_sessions_is_a_hit_without_any_request(tmp_path):
    rows = _rows(JAN_1, JAN_5)
    client = _db(rows)
    assert _load(client, tmp_path).meta.cache == "miss"
    client.calls.clear()
    again = _load(client, tmp_path)
    assert (again.meta.cache, client.calls) == ("hit", [])
    assert again.bars["ts_open"].tolist() == open_ns(rows)


def test_session_loaded_before_it_settled_is_fetched_again(tmp_path):
    rows = _rows(JAN_1, JAN_5)
    until_1100 = [row for row in rows if row["timestamp"] < at_ist(JAN_5, 11) // NS]
    client = _db(until_1100)
    _load(client, tmp_path, now_ns=at_ist(JAN_5, 16))
    client.db[("X", "NSE", "5m")] = historify_shape(rows)  # the store completes the day later
    client.calls.clear()
    later = _load(client, tmp_path, now_ns=at_ist(JAN_8, 18))
    assert _windows(client) == [("2024-01-05", "2024-01-05")]
    assert later.bars["ts_open"].tolist() == open_ns(rows)


def test_broker_session_asked_on_its_own_date_is_fetched_again_the_next_day(tmp_path):
    client = _broker(_rows(JAN_1, JAN_5))
    _load(client, tmp_path, source="api", now_ns=at_ist(JAN_5, 23, 0))
    client.calls.clear()
    _load(client, tmp_path, source="api", now_ns=at_ist(JAN_8, 18))
    assert _windows(client, "api") == [("2024-01-05", "2024-01-05")]


def test_todays_unfinished_session_is_fetched_on_every_load(tmp_path):
    mid_session = at_ist(JAN_5, 11, 0)
    client = _db(_rows(JAN_1, JAN_5))
    _load(client, tmp_path, now_ns=mid_session)
    client.calls.clear()
    again = _load(client, tmp_path, now_ns=mid_session)
    assert _windows(client) == [("2024-01-05", "2024-01-05")]
    assert again.bars["ts_close"].iloc[-1] == mid_session  # the 10:55 bar, closed at 11:00


def test_nothing_is_asked_for_today_before_the_session_opens(tmp_path):
    client = _db(_rows(JAN_1, JAN_5))
    _load(client, tmp_path, last=dt.date(2024, 1, 4), now_ns=at_ist(JAN_5, 7, 0))
    client.calls.clear()
    result = _load(client, tmp_path, now_ns=at_ist(JAN_5, 8, 0))
    assert (result.meta.cache, client.calls) == ("hit", [])


@pytest.mark.parametrize(
    "first_load, second_load, asked",
    [
        ((JAN_1, JAN_5), (JAN_1, JAN_12), ("2024-01-08", "2024-01-12")),
        ((JAN_8, JAN_12), (JAN_1, JAN_12), ("2024-01-01", "2024-01-05")),
    ],
    ids=["later days", "earlier days"],
)
def test_top_up_fetches_only_the_sessions_the_cache_lacks(tmp_path, first_load, second_load, asked):
    rows = _rows(JAN_1, JAN_12)
    client = _db(rows)
    _load(client, tmp_path, *first_load)
    client.calls.clear()
    result = _load(client, tmp_path, *second_load)
    assert (result.meta.cache, _windows(client)) == ("top-up", [asked])
    assert result.bars["ts_open"].tolist() == open_ns(rows)


def test_refresh_fetches_the_requested_range_again(tmp_path):
    client = _db(_rows(JAN_1, JAN_5))
    _load(client, tmp_path)
    client.calls.clear()
    result = _load(client, tmp_path, refresh=True)
    assert (result.meta.cache, _windows(client)) == ("refresh", [("2024-01-01", "2024-01-05")])


def test_sessions_a_broker_does_not_hold_are_asked_again_with_one_request(tmp_path):
    held = [row for row in _rows(JAN_1, JAN_19) if ist_date_of(row["timestamp"]) >= JAN_8]
    client = _broker(held)
    first = _load(client, tmp_path, last=JAN_19, source="api")
    client.calls.clear()
    second = _load(client, tmp_path, last=JAN_19, source="api")
    assert _windows(client, "api") == [("2024-01-01", "2024-01-05")]
    assert first.meta.short_history and second.meta.short_history
    assert second.meta.warnings[0].startswith("Short history: 2024-01-01 to 2024-01-19")


def test_empty_source_with_an_empty_cache_is_a_no_data_error(tmp_path):
    with pytest.raises(NoDataError, match="returned no bars"):
        _load(FakeClient(), tmp_path)


# --- verification: every read, and repair from the source ---------------------------


def _alter_first_close(tmp_path):
    path = _series_dir(tmp_path) / "2024-01.npy"
    records = np.load(path)
    records["close"][0] = 99999.0
    records["high"][0] = 99999.0
    np.save(path, records)
    return path


@pytest.mark.parametrize("last", [JAN_5, JAN_12], ids=["plain re-read", "top-up of the same month"])
def test_altered_partition_is_never_served(tmp_path, last):
    rows = _rows(JAN_1, JAN_12)
    client = _db(rows)
    _load(client, tmp_path)
    _alter_first_close(tmp_path)
    result = _load(client, tmp_path, last=last)
    assert result.bars["close"].iloc[0] == rows[0]["close"]
    assert "failed verification" in result.meta.warnings[0]


def test_altered_partition_is_moved_aside_not_deleted(tmp_path):
    client = _db(_rows(JAN_1, JAN_5))
    _load(client, tmp_path)
    _alter_first_close(tmp_path)
    _load(client, tmp_path)
    names = sorted(path.name for path in _series_dir(tmp_path).iterdir())
    assert names[0] == "2024-01.npy" and names[1].startswith("2024-01.npy.quarantined-")


@pytest.mark.parametrize("refresh", [False, True])
def test_truncated_partition_is_fetched_again(tmp_path, refresh):
    rows = _rows(JAN_1, JAN_5)
    client = _db(rows)
    _load(client, tmp_path)
    path = _series_dir(tmp_path) / "2024-01.npy"
    path.write_bytes(path.read_bytes()[:400])
    result = _load(client, tmp_path, refresh=refresh)
    assert result.bars["ts_open"].tolist() == open_ns(rows)


@pytest.mark.parametrize("refresh", [False, True])
def test_broken_index_is_recovered_from_the_source(tmp_path, refresh):
    rows = _rows(JAN_1, JAN_5)
    client = _db(rows)
    _load(client, tmp_path)
    (_series_dir(tmp_path) / "index.json").write_bytes(b'{"version": 2, "exch')
    result = _load(client, tmp_path, refresh=refresh)
    assert result.bars["ts_open"].tolist() == open_ns(rows)
    assert "the cache index" in result.meta.warnings[0]


def test_damage_that_cannot_be_repaired_is_a_data_error_that_names_the_file(tmp_path):
    client = _db(_rows(JAN_1, JAN_5))
    _load(client, tmp_path)
    path = _alter_first_close(tmp_path)
    client.queue(503, {"status": "error", "message": "maintenance"})
    with pytest.raises(DataError) as caught:
        _load(client, tmp_path)
    assert str(path) in str(caught.value)
    assert "Fetching them again failed" in str(caught.value) and "HTTP 503" in str(caught.value)


def test_two_writers_on_one_series_never_serve_a_mixed_partition(tmp_path):
    rows = _rows(JAN_1, JAN_12)
    week_1 = rows_to_records(historify_shape(_rows(JAN_1, JAN_5)))
    week_2 = rows_to_records(historify_shape(_rows(JAN_8, JAN_12)))
    first, second = RawCache(tmp_path), RawCache(tmp_path)
    save_index = second._save_index

    def save_after_the_other_writer(folder, index):
        # `second` has written its partition from an index it read before `first` wrote
        first.write(KEY, week_1, [(JAN_1, JAN_5)], 1_704_800_000, source="db", broker=None)
        save_index(folder, index)

    second._save_index = save_after_the_other_writer
    second.write(KEY, week_2, [(JAN_8, JAN_12)], 1_705_400_000, source="db", broker=None)

    client = _db(rows)
    result = _load(client, tmp_path, last=JAN_12)
    assert result.bars["ts_open"].tolist() == open_ns(rows)
    assert "failed verification" in result.meta.warnings[0]
    assert _windows(client) == [("2024-01-01", "2024-01-12")]


# --- every load normalises the cached rows with its own calendar and options --------


def _with_preopen(rows):
    return FakeClient(db={("X", "NSE", "5m"): historify_shape(rows, CAL, "NSE", preopen_minutes=5)})


def test_default_load_after_a_keep_all_load_serves_no_preopen_bars(tmp_path):
    rows = _rows(JAN_1, JAN_5)
    client = _with_preopen(rows)
    _load(client, tmp_path, provider_options={"regular_session_only": False})
    default = _load(client, tmp_path)
    assert default.meta.cache == "hit"
    assert default.bars["ts_open"].tolist() == open_ns(rows)


def test_cached_daily_bars_take_their_session_times_from_the_calendar_of_the_load(tmp_path):
    week = days_between(JAN_1, JAN_5)
    client = FakeClient(db={("X", "NSE", "D"): historify_shape(daily_rows(CAL, "NSE", week))})
    arguments = {"now_ns": LONG_AFTER, "source": "db", "cache_dir": tmp_path}
    server_load(client, "D", JAN_1, JAN_5, calendar=CAL, **arguments)
    closes_at_1700 = FakeCalendar(
        special={("NSE", day): (at_ist(day, 9, 15), at_ist(day, 17, 0)) for day in week}
    )
    result = server_load(client, "D", JAN_1, JAN_5, calendar=closes_at_1700, **arguments)
    assert result.meta.cache == "hit"
    assert result.bars["ts_close"].tolist() == [at_ist(day, 17, 0) for day in week]


def test_counts_do_not_grow_when_the_same_range_is_loaded_again(tmp_path):
    client = _with_preopen(_rows(JAN_1, JAN_5))
    counts = []
    for options in ({}, {"refresh": True}, {"refresh": True}, {}):
        result = _load(client, tmp_path, **options)
        counts.append(result.meta.quality.count("preopen_dropped"))
    assert counts == [25, 25, 25, 25]


def test_counts_of_a_one_day_load_describe_that_day_only(tmp_path):
    client = _with_preopen(_rows(JAN_1, JAN_5))
    _load(client, tmp_path)
    friday = _load(client, tmp_path, first=JAN_5)
    assert friday.meta.normalisation.preopen_dropped == 5
    assert friday.meta.normalisation.rows_out == BARS_PER_DAY


def test_open_interest_rule_is_applied_to_the_whole_series_of_the_load(tmp_path):
    daily = broker_shape(daily_rows(CAL, "NSE", days_between(JAN_1, JAN_12)))
    for row in daily:  # a new contract: no open interest in week 1, real values in week 2
        row["oi"] = 0 if ist_date_of(row["timestamp"]) <= JAN_5 else 5000
    client = FakeClient(api={("X", "NSE", "D"): daily})
    arguments = {"calendar": CAL, "source": "api", "cache_dir": tmp_path}
    first = server_load(client, "D", JAN_1, JAN_5, now_ns=at_ist(JAN_8, 9), **arguments)
    both = server_load(client, "D", JAN_1, JAN_12, now_ns=LONG_AFTER, **arguments)
    assert first.bars["oi"].isna().all()  # week 1 alone: the source supplies none
    assert both.bars["oi"].tolist() == [0.0] * 5 + [5000.0] * 5  # with week 2 the zeros are real


# --- brokers: nothing cached is thrown away -----------------------------------------


def test_failed_ping_bypasses_the_cache_and_keeps_what_it_holds(tmp_path):
    rows = _rows(JAN_1, dt.date(2024, 1, 9))
    client = _broker(rows)
    _load(client, tmp_path, source="api", now_ns=at_ist(JAN_8, 9))
    client.broker = None  # ping answers HTTP 500
    during = _load(
        client, tmp_path, last=JAN_8, source="api", now_ns=at_ist(dt.date(2024, 1, 9), 9)
    )
    client.broker = "zerodha"
    client.calls.clear()
    after = _load(client, tmp_path, last=dt.date(2024, 1, 9), source="api")
    assert during.meta.cache == "bypassed"
    assert "the local cache was not" in during.meta.warnings[0]
    assert _windows(client, "api") == [("2024-01-08", "2024-01-09")]  # week 1 is still cached
    assert after.bars["ts_open"].tolist() == open_ns(rows)


def test_refresh_keeps_cached_sessions_the_broker_no_longer_serves(tmp_path):
    rows = _rows(JAN_1, JAN_19)
    client = _broker(rows)
    _load(client, tmp_path, last=JAN_19, source="api")
    last_week = [row for row in rows if ist_date_of(row["timestamp"]) >= dt.date(2024, 1, 15)]
    client.api[("X", "NSE", "5m")] = broker_shape(last_week)
    result = _load(client, tmp_path, last=JAN_19, source="api", refresh=True)
    assert result.bars["ts_open"].tolist() == open_ns(rows)
    assert result.meta.warnings == [
        "the source returned no rows for 10 sessions that are in the cache (2024-01-01 to "
        "2024-01-12); the cached rows were kept"
    ]


def test_declared_timestamp_shift_bypasses_the_cache(tmp_path):
    shifted_late = [dict(row, timestamp=row["timestamp"] + 19_800) for row in _rows(JAN_1, JAN_5)]
    client = _broker(shifted_late, broker="compositedge")
    result = _load(
        client, tmp_path, source="api", provider_options={"intraday_shift_seconds": 19_800}
    )
    assert result.meta.cache == "bypassed"
    assert list(tmp_path.iterdir()) == []
    assert any("the local cache" in warning for warning in result.meta.warnings)


# --- the probe decides the source of auto; the cache never does ---------------------


def test_auto_is_not_settled_by_a_cached_store_series_the_probe_rejects(tmp_path):
    all_days = days_between(JAN_1, JAN_19)
    without_week_2 = [day for day in all_days if not JAN_8 <= day <= JAN_12]
    client = FakeClient(
        db={("X", "NSE", "D"): historify_shape(daily_rows(CAL, "NSE", without_week_2))},
        api={("X", "NSE", "D"): broker_shape(daily_rows(CAL, "NSE", all_days))},
    )
    arguments = {"calendar": CAL, "now_ns": LONG_AFTER, "cache_dir": tmp_path}
    server_load(client, "D", JAN_1, JAN_19, source="db", **arguments)
    result = server_load(client, "D", JAN_1, JAN_19, source="auto", **arguments)
    assert (result.meta.source, len(result.bars)) == ("api", 15)
    assert result.meta.auto_probe.decision == "api"


def test_auto_asks_the_probe_even_when_the_chosen_series_is_cached(tmp_path):
    days = days_between(JAN_1, JAN_5)
    client = FakeClient(db={("X", "NSE", "D"): historify_shape(daily_rows(CAL, "NSE", days))})
    arguments = {"calendar": CAL, "now_ns": LONG_AFTER, "cache_dir": tmp_path, "source": "auto"}
    server_load(client, "D", JAN_1, JAN_5, **arguments)
    client.calls.clear()
    again = server_load(client, "D", JAN_1, JAN_5, **arguments)
    assert (again.meta.cache, again.meta.source) == ("hit", "db")
    assert client.history_calls() == [("db", "D", "2024-01-01", "2024-01-05")]  # the probe


# --- manifest hashes: what the run used ---------------------------------------------


def test_manifest_lists_one_hash_per_month_of_the_returned_bars(tmp_path):
    result = _load(
        _db(_rows(JAN_1, dt.date(2024, 2, 9))), tmp_path, first=dt.date(2024, 1, 29),
        last=dt.date(2024, 2, 2),
    )  # fmt: skip
    assert list(result.meta.partition_hashes()) == [
        "openalgo-db/NSE/X/5m/2024-01", "openalgo-db/NSE/X/5m/2024-02",
    ]  # fmt: skip
    assert [entry["rows"] for entry in result.meta.partitions] == [
        3 * BARS_PER_DAY,
        2 * BARS_PER_DAY,
    ]


def test_manifest_hash_is_sha256_of_a_version_header_and_the_little_endian_bars(tmp_path):
    one_bar = _rows(JAN_1, JAN_1)[:1]
    result = _load(_db(one_bar), tmp_path, last=JAN_1)
    row = one_bar[0]
    times = struct.pack("<qq", at_ist(JAN_1, 9, 15), at_ist(JAN_1, 9, 20))
    prices = struct.pack("<4d", row["open"], row["high"], row["low"], row["close"])
    volume_and_oi = struct.pack("<2d", row["volume"], float("nan"))
    header = b"openalgo.backtest bars v1\n"
    expected = hashlib.sha256(header + times + prices + volume_and_oi).hexdigest()
    assert result.meta.partition_hashes() == {"openalgo-db/NSE/X/5m/2024-01": f"sha256:{expected}"}


def test_manifest_hash_of_a_range_is_not_changed_by_an_unrelated_top_up(tmp_path):
    client = _db(_rows(JAN_1, JAN_19))
    first_week = _load(client, tmp_path)
    _load(client, tmp_path, last=JAN_19)  # the same month file grows
    again = _load(client, tmp_path)
    assert again.meta.partition_hashes() == first_week.meta.partition_hashes()


def test_manifest_hash_changes_when_one_price_of_the_range_changes(tmp_path):
    rows = _rows(JAN_1, JAN_5)
    before = _load(_db(rows), tmp_path / "a").meta.partition_hashes()
    rows[-1] = dict(rows[-1], close=rows[-1]["close"] + 0.05)
    after = _load(_db(rows), tmp_path / "b").meta.partition_hashes()
    assert list(before) == list(after)
    assert before != after


def test_cached_and_uncached_loads_of_the_same_bars_hash_alike(tmp_path):
    client = _db(_rows(JAN_1, JAN_5))
    cached = _load(client, tmp_path)
    uncached = server_load(
        client, "5m", JAN_1, JAN_5, calendar=CAL, now_ns=LONG_AFTER, source="db", cache=False
    )
    assert cached.meta.partition_hashes() == uncached.meta.partition_hashes()
