"""`load_bars`: argument handling, the metadata it returns and the calendar it uses.
Source-specific behaviour is covered in the tests of each provider, of the cache
and of resampling.
"""

import datetime as dt
import json

import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    FakeClient,
    at_ist,
    broker_shape,
    days_between,
    historify_shape,
    ist_frame,
    server_load,
    truth_rows,
)

from openalgo.backtest._protocols import SessionCalendar
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.data import BarsResult, Frame, LoadMetadata, load_bars, partition_hashes
from openalgo.backtest.errors import ConfigError, NoDataError

MON = dt.date(2024, 1, 1)
FRI = dt.date(2024, 1, 5)
WEEK = days_between(MON, FRI)
CAL = FakeCalendar()
LONG_AFTER = at_ist(dt.date(2024, 3, 15), 18, 0)
ROWS = truth_rows(CAL, "NSE", WEEK)


def _own(**options):
    arguments = {"data": ist_frame(ROWS), "calendar": CAL, "now": LONG_AFTER, **options}
    return load_bars("X", "NSE", "1m", MON, FRI, **arguments)


# --- what comes back ----------------------------------------------------------------


def test_result_unpacks_into_bars_and_metadata():
    result = _own()
    bars, meta = result
    assert isinstance(result, BarsResult) and isinstance(meta, LoadMetadata)
    assert bars is result.bars and meta is result.meta


def test_bars_carry_the_series_and_its_source_in_attrs():
    assert _own().bars.attrs == {
        "symbol": "X",
        "exchange": "NSE",
        "interval": "1m",
        "provider": "frame",
        "source": "frame",
        "broker": None,
        "timestamp_convention": "tz-aware (Asia/Kolkata)/open",
    }


def test_metadata_reports_the_requested_and_the_obtained_range():
    meta = _own().meta
    assert (meta.requested_start, meta.requested_end) == ("2024-01-01", "2024-01-05")
    assert (meta.obtained_start, meta.obtained_end) == ("2024-01-01", "2024-01-05")
    assert (meta.rows, meta.short_history, meta.warnings) == (375 * 5, False, [])


def test_metadata_is_json_serialisable_for_the_run_manifest():
    as_dict = json.loads(json.dumps(_own().meta.to_dict()))
    assert as_dict["source"] == "frame"
    assert list(as_dict["partition_hashes"]) == ["frame/NSE/X/1m/2024-01"]
    assert as_dict["quality"]["ok"] is True
    assert as_dict["normalisation"]["rows_out"] == 375 * 5


def test_partition_hashes_of_a_result_are_those_of_its_metadata():
    result = _own()
    assert partition_hashes(result) == result.meta.partition_hashes()
    assert list(partition_hashes(result)) == ["frame/NSE/X/1m/2024-01"]


def test_metadata_of_a_server_load_names_source_broker_and_probe():
    client = FakeClient(broker="angel", api={("X", "NSE", "1m"): broker_shape(ROWS)})
    meta = server_load(client, "1m", MON, FRI, calendar=CAL, now_ns=LONG_AFTER, cache=False).meta
    assert (meta.source, meta.source_requested, meta.broker) == ("api", "auto", "angel")
    assert meta.auto_probe.decision == "api"
    assert meta.timestamp_convention == "utc_open/ist_date"
    assert (meta.fetch.windows_planned, meta.fetch.windows_fetched) == (1, 1)


def test_dates_may_be_strings_dates_or_datetimes():
    by_string = load_bars(
        "X",
        "NSE",
        "1m",
        "2024-01-02",
        "2024-01-03",
        data=ist_frame(ROWS),
        calendar=CAL,
        now=LONG_AFTER,
    )
    by_datetime = load_bars(
        "X", "NSE", "1m", dt.datetime(2024, 1, 2, 9, 15), dt.date(2024, 1, 3),
        data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER,
    )  # fmt: skip
    assert by_string.meta.rows == by_datetime.meta.rows == 375 * 2


def test_own_data_shorter_than_the_request_is_announced_first():
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


# --- arguments that do not fit together ---------------------------------------------


@pytest.mark.parametrize(
    "options, message",
    [
        ({"source": "db", "data": None}, "source='db' reads from an OpenAlgo server"),
        ({"source": "db"}, "data= and source='db' are alternatives"),
        ({"data": None, "client": FakeClient(), "tz": "UTC"}, "tz describe own data"),
        ({"data": None, "source": "nowhere"}, "unknown data provider 'nowhere'"),
        ({"data": None, "source": 42}, "must have a non-empty string attribute 'name'"),
        ({"data": Frame(ist_frame(ROWS)), "tz": "UTC"}, "tz describe own data"),
    ],
    ids=["server source without client", "data with a source", "tz without data",
         "unknown provider name", "source of the wrong type", "tz with a provider object"],
)  # fmt: skip
def test_arguments_that_do_not_fit_together_are_a_config_error(options, message):
    with pytest.raises(ConfigError, match=message):
        _own(**options)


@pytest.mark.parametrize(
    "start, end, interval, message",
    [
        (FRI, MON, "1m", "start 2024-01-05 is after end 2024-01-01"),
        ("01/02/2024", FRI, "1m", "start: cannot read '01/02/2024' as a date"),
        (MON, FRI, "1min", "unknown interval '1min'"),
    ],
)
def test_bad_dates_and_intervals_are_a_config_error(start, end, interval, message):
    with pytest.raises(ConfigError, match=message):
        load_bars(
            "X", "NSE", interval, start, end, data=ist_frame(ROWS), calendar=CAL, now=LONG_AFTER
        )


@pytest.mark.parametrize(
    "provider_options",
    [{"bogus": 1}, {"calendar": CAL}],
    ids=["unknown option", "argument given twice"],
)
def test_server_option_the_provider_does_not_take_is_a_config_error(provider_options):
    client = FakeClient(db={("X", "NSE", "1m"): historify_shape(ROWS)})
    with pytest.raises(ConfigError, match="provider_options"):
        load_bars(
            "X", "NSE", "1m", MON, FRI, client=client, source="db", calendar=CAL, now=LONG_AFTER,
            cache=False, provider_options=provider_options,
        )  # fmt: skip


def test_source_without_bars_in_the_range_is_a_no_data_error():
    with pytest.raises(NoDataError, match="holds no rows between 2024-02-01 and 2024-02-02"):
        load_bars(
            "X", "NSE", "1m", "2024-02-01", "2024-02-02", data=ist_frame(ROWS), calendar=CAL,
            now=LONG_AFTER,
        )  # fmt: skip


# --- the calendar -------------------------------------------------------------------


def test_default_calendar_is_used_when_none_is_given():
    # The built-in NSE calendar closes Monday 2024-01-22 and Friday 2024-01-26 (Republic
    # Day), so this week has three sessions; a weekday rule would expect five.
    week = days_between(dt.date(2024, 1, 22), dt.date(2024, 1, 26))
    rows = truth_rows(default_calendar(), "NSE", week)
    result = load_bars("X", "NSE", "1m", week[0], week[-1], data=ist_frame(rows), now=LONG_AFTER)
    assert result.meta.rows == 375 * 3
    assert result.meta.short_history is False  # the holidays are not expected sessions


def test_fake_calendar_of_these_tests_satisfies_the_protocol():
    assert isinstance(CAL, SessionCalendar)


def test_object_without_a_session_method_is_not_a_calendar():
    with pytest.raises(ConfigError, match="calendar must provide session"):
        _own(calendar=object())
