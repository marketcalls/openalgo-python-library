"""Own data without a server: `Frame`, `Files` and `load_bars(data=...)` (FR-DATA-04).

No test here uses a client. The reference series is one NSE week of 1-minute bars
(09:15 to 15:30 IST, Monday 2024-01-01 to Friday 2024-01-05).
"""

import datetime as dt
import sys

import numpy as np
import pandas as pd
import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    at_ist,
    canonical_frame,
    daily_rows,
    days_between,
    ist_frame,
    open_ns,
    shifted,
    truth_rows,
)

from openalgo.backtest.data import Files, Frame, load_bars
from openalgo.backtest.errors import (
    ConfigError,
    DataError,
    MissingDependencyError,
    NoDataError,
    TimestampError,
)

MON = dt.date(2024, 1, 1)
FRI = dt.date(2024, 1, 5)
WEEK = days_between(MON, FRI)
CAL = FakeCalendar()
LONG_AFTER = at_ist(dt.date(2024, 3, 15), 18, 0)
ROWS = truth_rows(CAL, "NSE", WEEK)
BARS_IN_WEEK = 375 * 5
FIRST_OPEN = at_ist(MON, 9, 15)
PRICES = ("open", "high", "low", "close", "volume")


def _bars(data, interval="1m", start=MON, end=FRI, **options):
    frame = Frame(data, calendar=CAL, now=LONG_AFTER, **options)
    return frame.bars("X", "NSE", interval, start, end)


def _naive_ist_frame(rows=ROWS):
    """IST clock times with no time zone attached, as most vendor files have them."""
    frame = ist_frame(rows)
    frame.index = frame.index.tz_localize(None)
    return frame


def _columns(rows=ROWS):
    return {name: [row[name] for row in rows] for name in PRICES}


def _epoch_seconds(rows=ROWS):
    return [row["timestamp"] for row in rows]


def _write_csv(path, frame, **options):
    frame.to_csv(path, **options)
    return path


# --- time zone: aware, or declared, never assumed -----------------------------------


def test_tz_aware_index_needs_no_tz_argument():
    bars = _bars(ist_frame(ROWS))
    assert len(bars) == BARS_IN_WEEK
    assert bars["ts_open"].iloc[0] == FIRST_OPEN


def test_tz_aware_index_in_utc_gives_the_same_instants():
    in_utc = ist_frame(ROWS)
    in_utc.index = in_utc.index.tz_convert("UTC")
    assert _bars(in_utc)["ts_open"].iloc[0] == FIRST_OPEN


def test_naive_index_without_tz_is_a_config_error_with_no_default():
    with pytest.raises(ConfigError, match="carry no time zone.*there is no default"):
        _bars(_naive_ist_frame())


def test_naive_index_with_declared_tz_is_placed_in_that_zone():
    bars = _bars(_naive_ist_frame(), tz="Asia/Kolkata")
    assert bars["ts_open"].iloc[0] == FIRST_OPEN
    assert bars.attrs["timestamp_convention"] == "naive, declared tz=Asia/Kolkata/open"


def test_ist_clock_times_declared_as_utc_are_caught_by_the_calendar():
    with pytest.raises(TimestampError) as caught:
        _bars(_naive_ist_frame(), tz="UTC")
    assert "5 hours 30 minutes late" in str(caught.value)
    assert "declare tz='Asia/Kolkata'" in str(caught.value)


def test_tz_argument_that_contradicts_tz_aware_data_is_a_config_error():
    with pytest.raises(ConfigError, match="already time-zone aware"):
        _bars(ist_frame(ROWS), tz="UTC")


def test_unknown_time_zone_is_a_timestamp_error():
    with pytest.raises(TimestampError, match="cannot be placed in time zone 'Mars/Olympus'"):
        _bars(_naive_ist_frame(), tz="Mars/Olympus")


# --- the supported in-memory inputs -------------------------------------------------


def test_dict_of_columns_with_epoch_seconds_needs_unit_and_tz():
    data = {"timestamp": _epoch_seconds(), **_columns()}
    bars = _bars(data, unit="s", tz="UTC")
    assert (len(bars), bars["ts_open"].iloc[0]) == (BARS_IN_WEEK, FIRST_OPEN)


def test_numeric_timestamps_without_unit_are_a_config_error():
    data = {"timestamp": _epoch_seconds(), **_columns()}
    with pytest.raises(ConfigError, match="Pass unit="):
        _bars(data, tz="UTC")


def test_structured_numpy_array_is_read_by_field_name():
    dtype = [("timestamp", "i8")] + [(name, "f8") for name in PRICES]
    records = np.array([tuple(row[name] for name, _ in dtype) for row in ROWS], dtype=dtype)
    bars = _bars(records, unit="s", tz="UTC")
    assert (len(bars), bars["ts_open"].iloc[0]) == (BARS_IN_WEEK, FIRST_OPEN)


def test_plain_numpy_array_needs_the_column_names_in_order():
    names = ["timestamp", *PRICES]
    matrix = np.array([[row[name] for name in names] for row in ROWS], dtype="float64")
    bars = _bars(matrix, columns=names, unit="s", tz="UTC")
    assert bars["close"].iloc[0] == ROWS[0]["close"]
    with pytest.raises(ConfigError, match="needs columns="):
        _bars(matrix, unit="s", tz="UTC")


def test_polars_like_frame_is_read_through_to_dict_without_pyarrow():
    class PolarsLike:
        """The two members of a polars DataFrame that the data layer relies on."""

        columns = ["timestamp", *PRICES]

        def to_dict(self, *, as_series):
            assert as_series is False
            stamps = [dt.datetime.fromtimestamp(s, tz=dt.UTC) for s in _epoch_seconds()]
            return {"timestamp": stamps, **_columns()}

    bars = _bars(PolarsLike())
    assert (len(bars), bars["ts_open"].iloc[0]) == (BARS_IN_WEEK, FIRST_OPEN)


def test_frame_already_in_the_canonical_layout_is_accepted_without_tz():
    bars = _bars(canonical_frame(ROWS))
    assert bars["ts_open"].tolist() == open_ns(ROWS)
    assert bars.attrs["timestamp_convention"] == "utc_ns/open (canonical layout)"


@pytest.mark.parametrize("data", [pd.Series([1.0, 2.0]), 42, [1.0, 2.0, 3.0]])
def test_unsupported_data_is_a_config_error(data):
    with pytest.raises(ConfigError):
        _bars(data, tz="UTC")


def test_frame_whose_to_dict_does_not_convert_is_an_error_that_says_why():
    class OtherFrame:
        columns = ["timestamp", *PRICES]

        def to_dict(self, orient="dict"):  # the pandas signature, not the polars one
            return {}

    with pytest.raises(ConfigError, match=r"to_dict\(as_series=False\) failed"):
        _bars(OtherFrame(), tz="UTC")


# --- columns ------------------------------------------------------------------------


def test_columns_are_found_by_canonical_name_in_any_letter_case():
    frame = ist_frame(ROWS).rename(columns=str.upper)
    assert _bars(frame)["close"].iloc[0] == ROWS[0]["close"]


def test_column_mapping_names_the_source_columns():
    frame = ist_frame(ROWS).reset_index()
    frame.columns = ["When", "O", "H", "L", "C", "Vol"]
    mapping = {
        "timestamp": "When",
        "open": "O",
        "high": "H",
        "low": "L",
        "close": "C",
        "volume": "Vol",
    }
    bars = _bars(frame, columns=mapping)
    assert (bars["open"].iloc[0], bars["volume"].iloc[0]) == (ROWS[0]["open"], ROWS[0]["volume"])


def test_date_and_time_columns_are_joined_into_the_timestamp():
    naive = _naive_ist_frame()
    frame = naive.reset_index(drop=True)
    frame["Date"] = naive.index.strftime("%Y-%m-%d")
    frame["Time"] = naive.index.strftime("%H:%M:%S")
    bars = _bars(frame, columns={"timestamp": ("Date", "Time")}, tz="Asia/Kolkata")
    assert bars["ts_open"].iloc[0] == FIRST_OPEN


@pytest.mark.parametrize(
    "change, mapping, message",
    [
        (lambda f: f.drop(columns=["high"]), None, r"no column for \['high'\]"),
        (lambda f: f.assign(OPEN=f["open"]), None, "several columns could be 'open'"),
        (lambda f: f.reset_index().assign(date=1), None, "several columns could be the timestamp"),
        (lambda f: f.reset_index(drop=True), None, "no timestamp found"),
        (lambda f: f, {"price": "close"}, "unknown keys"),
        (lambda f: f, {"close": "last"}, "names 'last', which is not in the data"),
    ],
    ids=["missing price", "two opens", "two timestamps", "no timestamp", "bad key", "bad name"],
)
def test_columns_that_cannot_be_resolved_are_a_config_error(change, mapping, message):
    with pytest.raises(ConfigError, match=message):
        _bars(change(ist_frame(ROWS)), columns=mapping)


def test_symbol_column_selects_the_rows_of_the_requested_symbol():
    mine = ist_frame(ROWS).assign(symbol="X")
    other = ist_frame(ROWS).assign(symbol="Y", close=1.0)
    bars = _bars(pd.concat([mine, other]))
    assert len(bars) == BARS_IN_WEEK
    assert bars["close"].iloc[0] == ROWS[0]["close"]


def test_symbol_that_is_not_in_the_symbol_column_is_a_no_data_error():
    with pytest.raises(NoDataError, match=r"no rows with symbol 'X'.*\['Y'\]"):
        _bars(ist_frame(ROWS).assign(symbol="Y"))


def test_non_numeric_price_column_is_a_data_error():
    frame = ist_frame(ROWS).astype({"close": "object"})
    frame.iloc[0, frame.columns.get_loc("close")] = "1,234.50"
    with pytest.raises(DataError, match="column 'close' is not numeric"):
        _bars(frame)


# --- what the frame holds -----------------------------------------------------------


@pytest.mark.parametrize(
    "held, message",
    [
        ({"interval": "5m"}, "holds interval '5m' but '1m' was requested.*base="),
        ({"symbol": "Y"}, "holds symbol 'Y' but 'X' was requested"),
        ({"exchange": "MCX"}, "holds exchange 'MCX' but 'NSE' was requested"),
    ],
)
def test_request_for_something_the_frame_does_not_hold_is_a_config_error(held, message):
    frame = Frame(ist_frame(ROWS), calendar=CAL, now=LONG_AFTER, **held)
    with pytest.raises(ConfigError, match=message):
        frame.bars("X", "NSE", "1m", MON, FRI)


def test_request_outside_the_data_is_a_no_data_error_that_says_what_is_held():
    with pytest.raises(NoDataError) as caught:
        _bars(ist_frame(ROWS), start=dt.date(2024, 2, 1), end=dt.date(2024, 2, 5))
    assert "holds no rows between 2024-02-01 and 2024-02-05" in str(caught.value)
    assert "2024-01-01 09:15 IST to 2024-01-05 15:29 IST" in str(caught.value)


def test_only_the_requested_dates_are_returned():
    bars = _bars(ist_frame(ROWS), start=dt.date(2024, 1, 2), end=dt.date(2024, 1, 3))
    assert bars["ts_open"].iloc[0] == at_ist(dt.date(2024, 1, 2), 9, 15)
    assert bars["ts_open"].iloc[-1] == at_ist(dt.date(2024, 1, 3), 15, 29)


# --- the bar label ------------------------------------------------------------------


def test_label_close_reads_stamps_as_bar_closes():
    close_stamped = ist_frame(shifted(ROWS, 60))  # 09:16 ... 15:30
    bars = _bars(close_stamped, label="close")
    assert bars["ts_open"].iloc[0] == FIRST_OPEN
    assert bars["ts_close"].iloc[0] == at_ist(MON, 9, 16)


def test_close_stamped_data_read_with_the_default_label_is_an_error():
    with pytest.raises(TimestampError, match="label='open' was declared"):
        _bars(ist_frame(shifted(ROWS, 60)))


def test_label_other_than_open_or_close_is_a_config_error():
    with pytest.raises(ConfigError, match="label must be 'open' or 'close'"):
        Frame(ist_frame(ROWS), label="mid")


# --- daily own data -----------------------------------------------------------------


def test_daily_dates_with_declared_tz_map_to_their_sessions():
    rows = daily_rows(CAL, "NSE", WEEK)
    frame = pd.DataFrame(
        {"date": [day.isoformat() for day in WEEK], **{n: [r[n] for r in rows] for n in PRICES}}
    )
    bars = _bars(frame, interval="D", tz="Asia/Kolkata")
    assert bars["ts_open"].tolist() == [at_ist(day, 9, 15) for day in WEEK]
    assert bars["ts_close"].tolist() == [at_ist(day, 15, 30) for day in WEEK]


# --- files --------------------------------------------------------------------------


def test_csv_with_utc_offsets_in_its_timestamps_needs_no_tz(tmp_path):
    path = _write_csv(tmp_path / "x.csv", ist_frame(ROWS))  # "2024-01-01 09:15:00+05:30"
    bars = Files(path, calendar=CAL, now=LONG_AFTER).bars("X", "NSE", "1m", MON, FRI)
    assert (len(bars), bars["ts_open"].iloc[0]) == (BARS_IN_WEEK, FIRST_OPEN)


def test_csv_with_naive_timestamps_requires_tz(tmp_path):
    path = _write_csv(tmp_path / "x.csv", _naive_ist_frame())
    with pytest.raises(ConfigError, match="carry no time zone"):
        Files(path, calendar=CAL, now=LONG_AFTER).bars("X", "NSE", "1m", MON, FRI)
    with_tz = Files(path, tz="Asia/Kolkata", calendar=CAL, now=LONG_AFTER)
    assert with_tz.bars("X", "NSE", "1m", MON, FRI)["ts_open"].iloc[0] == FIRST_OPEN


def test_csv_timestamps_that_are_not_iso_need_an_explicit_format(tmp_path):
    frame = _naive_ist_frame()
    frame.index = frame.index.strftime("%d-%m-%Y %H:%M")  # "01-01-2024 09:15": day first
    path = _write_csv(tmp_path / "x.csv", frame)
    with pytest.raises(ConfigError, match="not ISO 8601.*never guessed"):
        Files(path, tz="Asia/Kolkata", calendar=CAL, now=LONG_AFTER).bars(
            "X", "NSE", "1m", MON, FRI
        )
    declared = Files(
        path, tz="Asia/Kolkata", timestamp_format="%d-%m-%Y %H:%M", calendar=CAL, now=LONG_AFTER
    )
    assert declared.bars("X", "NSE", "1m", MON, FRI)["ts_open"].iloc[0] == FIRST_OPEN


def test_path_placeholders_are_filled_in_per_request(tmp_path):
    _write_csv(tmp_path / "X_NSE_1m.csv", ist_frame(ROWS))
    files = Files(
        str(tmp_path / "{symbol}_{exchange}_{interval}.csv"), calendar=CAL, now=LONG_AFTER
    )
    assert len(files.bars("X", "NSE", "1m", MON, FRI)) == BARS_IN_WEEK
    with pytest.raises(NoDataError, match="Y_NSE_1m.csv does not exist"):
        files.bars("Y", "NSE", "1m", MON, FRI)


def test_gzipped_csv_is_read_as_csv(tmp_path):
    path = _write_csv(tmp_path / "x.csv.gz", ist_frame(ROWS))
    assert (
        len(Files(path, calendar=CAL, now=LONG_AFTER).bars("X", "NSE", "1m", MON, FRI))
        == BARS_IN_WEEK
    )


@pytest.mark.parametrize(
    "name, options, message",
    [
        ("x.dat", {}, "cannot tell the format"),
        ("x.csv", {"format": "xlsx"}, "format must be one of"),
        ("x.csv", {"columns": ["a", "b"]}, "columns must be a mapping"),
        (
            "x.csv",
            {"read_options": {"parse_dates": True}},
            "do not ask the CSV reader to parse dates",
        ),
    ],
    ids=["unknown suffix", "unknown format", "column list", "reader date parsing"],
)
def test_file_that_cannot_be_read_as_declared_is_a_config_error(tmp_path, name, options, message):
    path = _write_csv(tmp_path / name, ist_frame(ROWS))
    with pytest.raises(ConfigError, match=message):
        Files(path, calendar=CAL, now=LONG_AFTER, **options).bars("X", "NSE", "1m", MON, FRI)


@pytest.mark.parametrize("name", ["x.parquet", "x.feather"])
def test_parquet_and_feather_without_pyarrow_say_which_package_is_missing(
    tmp_path, monkeypatch, name
):
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # makes `import pyarrow` fail
    path = tmp_path / name
    path.write_bytes(b"not read: the import fails first")
    with pytest.raises(MissingDependencyError, match="needs the optional package pyarrow"):
        Files(path, calendar=CAL, now=LONG_AFTER).bars("X", "NSE", "1m", MON, FRI)


@pytest.mark.parametrize("name, writer", [("x.parquet", "to_parquet"), ("x.feather", "to_feather")])
def test_parquet_and_feather_are_read_when_pyarrow_is_installed(tmp_path, name, writer):
    pytest.importorskip("pyarrow")
    path = tmp_path / name
    getattr(ist_frame(ROWS).reset_index(), writer)(path)
    bars = Files(path, calendar=CAL, now=LONG_AFTER).bars("X", "NSE", "1m", MON, FRI)
    assert (len(bars), bars["ts_open"].iloc[0]) == (BARS_IN_WEEK, FIRST_OPEN)


# --- load_bars with own data --------------------------------------------------------


def test_load_bars_with_a_frame_needs_no_client():
    result = load_bars(
        "X", "NSE", "1m", MON, FRI, data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER
    )
    assert (result.meta.provider, result.meta.source) == ("frame", "frame")
    assert (result.meta.cache, result.meta.rows) == ("not-applicable", BARS_IN_WEEK)
    assert result.meta.quality.ok


def test_load_bars_with_a_file_path_uses_the_files_provider(tmp_path):
    path = _write_csv(tmp_path / "x.csv", _naive_ist_frame())
    result = load_bars(
        "X", "NSE", "1m", MON, FRI, data=path, tz="Asia/Kolkata", calendar=CAL, now=LONG_AFTER
    )
    assert result.meta.provider == "files"
    assert result.bars["ts_open"].iloc[0] == FIRST_OPEN


def test_load_bars_hashes_own_data_per_month_for_the_manifest():
    result = load_bars(
        "X", "NSE", "1m", MON, FRI, data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER
    )
    assert list(result.meta.partition_hashes()) == ["frame/NSE/X/1m/2024-01"]
    assert result.meta.partitions[0]["rows"] == BARS_IN_WEEK


def test_load_bars_announces_own_data_that_is_shorter_than_the_request():
    result = load_bars(
        "X",
        "NSE",
        "1m",
        MON,
        dt.date(2024, 1, 12),
        data=ist_frame(ROWS),
        calendar=CAL,
        now=LONG_AFTER,
    )
    assert result.meta.short_history is True
    assert result.meta.warnings[0] == (
        "Short history: 2024-01-01 to 2024-01-12 was requested but the supplied data returned "
        "2024-01-01 to 2024-01-05, 5 of 10 sessions. The supplied data does not cover the rest "
        "of the requested range."
    )


def test_load_bars_passes_tz_to_the_frame():
    with pytest.raises(ConfigError, match="carry no time zone"):
        load_bars("X", "NSE", "1m", MON, FRI, data=_naive_ist_frame(), calendar=CAL, now=LONG_AFTER)


# --- rows are selected by date after normalisation ----------------------------------

_HOW = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _own_daily_frame(first, last):
    days = [day for day in days_between(first, last) if CAL.session("NSE", day)]
    values = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}
    return pd.DataFrame(values, index=pd.DatetimeIndex(days))


def _load_own(frame, interval, first, last):
    return load_bars(
        "X", "NSE", interval, first, last, data=frame, tz="Asia/Kolkata", calendar=CAL,
        now=at_ist(dt.date(2024, 6, 3), 18),
    )  # fmt: skip


def test_month_labelled_at_its_last_day_is_kept_when_the_label_is_after_the_range_end():
    last_session = dt.date(2024, 3, 29)
    monthly = _own_daily_frame(MON, last_session).resample("ME").agg(_HOW)  # Jan 31, Feb 29, Mar 31
    bars = _load_own(monthly, "M", MON, last_session).bars
    assert bars["ts_open"].tolist() == [
        at_ist(MON, 9, 15), at_ist(dt.date(2024, 2, 1), 9, 15), at_ist(dt.date(2024, 3, 1), 9, 15),
    ]  # fmt: skip
    assert bars["ts_close"].iloc[-1] == at_ist(last_session, 15, 30)


def test_week_labelled_at_its_sunday_is_kept_when_the_range_ends_on_the_friday():
    weekly = _own_daily_frame(MON, dt.date(2024, 1, 12)).resample("W").agg(_HOW)  # Jan 7, Jan 14
    bars = _load_own(weekly, "W", MON, dt.date(2024, 1, 12)).bars
    assert bars["ts_close"].tolist() == [at_ist(FRI, 15, 30), at_ist(dt.date(2024, 1, 12), 15, 30)]


def test_own_period_bar_that_reaches_beyond_the_requested_dates_is_dropped_and_counted():
    monthly = _own_daily_frame(MON, dt.date(2024, 3, 29)).resample("ME").agg(_HOW)
    result = _load_own(monthly, "M", dt.date(2024, 1, 15), dt.date(2024, 3, 15))
    assert result.bars["ts_open"].tolist() == [at_ist(dt.date(2024, 2, 1), 9, 15)]
    assert result.meta.quality.count("incomplete_period_dropped") == 2


def test_option_the_provider_does_not_know_is_a_config_error():
    with pytest.raises(ConfigError, match=r"provider_options \['bogus'\] do not fit"):
        load_bars(
            "X", "NSE", "1m", MON, FRI, data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER,
            provider_options={"bogus": 1},
        )  # fmt: skip
