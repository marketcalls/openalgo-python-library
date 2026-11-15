"""Fetching from an OpenAlgo server: the chunk window table, chunk planning, the walk
over a date range, pacing and HTTP 429 backoff (FR-DATA-07). No test waits: the
clock and sleep are injected.

The window values asserted here are the ones of PRD 06 section 4 and of the
reference note on the OpenAlgo broker adapters, written out by hand.
"""

import datetime as dt

import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    FakeClient,
    FakeClock,
    at_ist,
    broker_shape,
    days_between,
    historify_shape,
    make_provider,
    server_load,
    truth_rows,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from openalgo.backtest.data.brokers import BROKER_PROFILES, DEFAULT_PROFILE, broker_profile
from openalgo.backtest.data.fetching import plan_chunks, rows_to_records
from openalgo.backtest.errors import ConfigError, DataError, TimestampError

START = dt.date(2024, 1, 1)
END = dt.date(2024, 5, 31)  # 152 days after START, inclusive
DAYS = days_between(START, END)
NOW = at_ist(dt.date(2024, 6, 3), 18, 0)
CAL = FakeCalendar()
CONFIRMED_BROKERS = ("zerodha", "angel", "upstox", "fyers", "dhan", "kotak")
XTS_FAMILY = ("compositedge", "fivepaisaxts", "ibulls", "iifl", "jainamxts", "rmoney", "wisdom")
BUSY = {"status": "error", "message": "OpenAlgo is pacing requests for your broker"}
BARS_PER_DAY_5M = 75


def _client(broker="zerodha", days=DAYS, **options):
    rows = broker_shape(truth_rows(CAL, "NSE", days, step=300))
    return FakeClient(broker=broker, api={("X", "NSE", "5m"): rows}, **options)


def _load(client, start=START, end=END, source="api", now_ns=NOW, **options):
    return server_load(
        client, "5m", start, end, calendar=CAL, now_ns=now_ns, source=source, cache=False, **options
    )


def _windows(client, source="api"):
    return [(first, last) for _, _, first, last in client.history_calls(source)]


# --- the per-broker window table ----------------------------------------------------


@pytest.mark.parametrize(
    "broker, interval, days",
    [
        ("zerodha", "1m", 60), ("zerodha", "1h", 60), ("zerodha", "D", 2000),
        ("angel", "1m", 30), ("angel", "3m", 60), ("angel", "5m", 100), ("angel", "10m", 100),
        ("angel", "15m", 200), ("angel", "30m", 200), ("angel", "1h", 400), ("angel", "D", 2000),
        ("upstox", "1m", 30), ("upstox", "15m", 30), ("upstox", "30m", 90), ("upstox", "4h", 90),
        ("upstox", "D", 3650),
        ("fyers", "5s", 25), ("fyers", "1m", 60), ("fyers", "1h", 60), ("fyers", "D", 300),
        ("dhan", "1m", 90), ("dhan", "15m", 90),
        ("kotak", "1m", 30), ("kotak", "5m", 30), ("kotak", "10m", 60), ("kotak", "15m", 60),
        ("kotak", "30m", 90), ("kotak", "1h", 90), ("kotak", "D", 180),
    ],
)  # fmt: skip
def test_chunk_window_of_a_confirmed_broker(broker, interval, days):
    assert broker_profile(broker).window_days(interval) == days


@pytest.mark.parametrize(
    "broker, interval, days", [("angel", "2m", 30), ("upstox", "20m", 30), ("kotak", "2h", 90)]
)
def test_interval_the_table_does_not_list_gets_the_next_finer_window(broker, interval, days):
    assert broker_profile(broker).window_days(interval) == days


@pytest.mark.parametrize("interval, days", [("5s", 25), ("1m", 30), ("1h", 30), ("D", 180)])
def test_unknown_broker_gets_the_conservative_default_window(interval, days):
    assert broker_profile("somebroker").window_days(interval) == days
    assert broker_profile(None).window_days(interval) == days


@pytest.mark.parametrize("interval", ["5s", "1m", "5m", "15m", "1h", "D"])
def test_no_confirmed_broker_has_a_smaller_window_than_the_default(interval):
    smallest = min(BROKER_PROFILES[name].window_days(interval) for name in CONFIRMED_BROKERS)
    assert DEFAULT_PROFILE.window_days(interval) <= smallest


def test_exactly_the_six_prd_brokers_are_marked_verified():
    verified = {name for name, profile in BROKER_PROFILES.items() if profile.verified}
    assert verified == set(CONFIRMED_BROKERS)


@pytest.mark.parametrize("broker", XTS_FAMILY)
def test_xts_family_timestamps_are_flagged_unverified_with_no_shift_assumed(broker):
    profile = BROKER_PROFILES[broker]
    assert profile.timestamps_verified is False
    assert profile.intraday_shift_seconds == 0
    assert "XTS family" in profile.notes


def test_broker_name_is_matched_without_regard_to_case():
    assert broker_profile("Zerodha").name == "zerodha"


# --- chunk planning -----------------------------------------------------------------


def test_chunks_are_anchored_at_the_end_so_only_the_oldest_is_short():
    assert plan_chunks(START, END, 60) == [
        (dt.date(2024, 1, 1), dt.date(2024, 2, 1)),  # the remainder: 32 days
        (dt.date(2024, 2, 2), dt.date(2024, 4, 1)),  # 60 days
        (dt.date(2024, 4, 2), dt.date(2024, 5, 31)),  # 60 days
    ]


@pytest.mark.parametrize(
    "end, window_days, expected_chunks",
    [
        (dt.date(2024, 1, 1), 30, 1),   # one day
        (dt.date(2024, 2, 29), 60, 1),  # exactly one window (60 days)
        (dt.date(2024, 3, 1), 60, 2),   # one day more than a window
    ],
)  # fmt: skip
def test_chunk_count_at_the_window_boundary(end, window_days, expected_chunks):
    assert len(plan_chunks(START, end, window_days)) == expected_chunks


@pytest.mark.parametrize("start, end, window_days", [(END, START, 30), (START, END, 0)])
def test_impossible_chunk_plan_is_a_config_error(start, end, window_days):
    with pytest.raises(ConfigError):
        plan_chunks(start, end, window_days)


@settings(max_examples=100, deadline=None)
@given(span=st.integers(0, 4000), window=st.integers(1, 400))
def test_chunks_cover_every_day_exactly_once_within_the_window(span, window):
    start = dt.date(2015, 1, 1)
    end = start + dt.timedelta(days=span)
    chunks = plan_chunks(start, end, window)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all(low <= high and (high - low).days < window for low, high in chunks)
    one_day = dt.timedelta(days=1)
    assert all(nxt[0] == prev[1] + one_day for prev, nxt in zip(chunks, chunks[1:], strict=False))


# --- the walk over a date range -----------------------------------------------------


@pytest.mark.parametrize(
    "broker, expected",
    [
        (
            "zerodha",  # 60 days per request
            [("2024-04-02", "2024-05-31"), ("2024-02-02", "2024-04-01"),
             ("2024-01-01", "2024-02-01")],
        ),
        (
            "kotak",  # 30 days per request at 5m
            [("2024-05-02", "2024-05-31"), ("2024-04-02", "2024-05-01"),
             ("2024-03-03", "2024-04-01"), ("2024-02-02", "2024-03-02"),
             ("2024-01-03", "2024-02-01"), ("2024-01-01", "2024-01-02")],
        ),
    ],
)  # fmt: skip
def test_broker_range_is_requested_in_table_windows_newest_first(broker, expected):
    client = _client(broker)
    _load(client)
    assert _windows(client) == expected


def test_chunk_days_overrides_the_table():
    client = _client(days=days_between(START, dt.date(2024, 1, 31)))
    _load(client, end=dt.date(2024, 1, 31), provider_options={"chunk_days": 10})
    assert len(_windows(client)) == 4  # 31 days in windows of 10


def test_historify_range_is_read_in_engine_windows_oldest_first():
    rows = historify_shape(truth_rows(CAL, "NSE", days_between(START, dt.date(2024, 3, 31))))
    client = FakeClient(db={("X", "NSE", "1m"): rows})
    server_load(
        client,
        "1m",
        START,
        dt.date(2024, 3, 31),
        calendar=CAL,
        now_ns=NOW,
        source="db",
        cache=False,
    )
    assert _windows(client, "db") == [
        ("2024-01-01", "2024-01-31"),  # the remainder: 31 days
        ("2024-02-01", "2024-03-31"),  # 60 days, the engine window for 1m
    ]


def test_broker_walk_stops_at_the_first_empty_window_that_had_a_session():
    held = [day for day in DAYS if day >= dt.date(2024, 5, 10)]
    client = _client(days=held)
    result = _load(client)
    # zerodha windows: Apr 2 to May 31 has bars, Feb 2 to Apr 1 is empty, the oldest is not asked
    assert _windows(client) == [("2024-04-02", "2024-05-31"), ("2024-02-02", "2024-04-01")]
    assert result.meta.fetch.stopped == (
        "the broker returned no bars for 2024-02-02 to 2024-04-01, so the history is taken "
        "to start after 2024-04-01; nothing older was asked for"
    )


def test_empty_window_between_cached_sessions_does_not_end_the_broker_walk(tmp_path):
    # The broker has no bars for Wednesday 2024-01-10, a session by the calendar (as a
    # holiday the calendar does not know), and the cache holds the sessions around it.
    wednesday, friday = dt.date(2024, 1, 10), dt.date(2024, 1, 19)
    client = _client(days=[day for day in days_between(START, friday) if day != wednesday])
    cached = {"source": "api", "cache_dir": tmp_path}
    for first, last in [(dt.date(2024, 1, 8), dt.date(2024, 1, 9)), (dt.date(2024, 1, 11), friday)]:
        server_load(client, "5m", first, last, calendar=CAL, now_ns=NOW, **cached)
    client.calls.clear()
    result = server_load(client, "5m", START, friday, calendar=CAL, now_ns=NOW, **cached)
    assert _windows(client) == [("2024-01-10", "2024-01-10"), ("2024-01-01", "2024-01-05")]
    assert result.meta.fetch.stopped is None
    assert result.meta.short_history is False
    assert len(result.bars) == 14 * BARS_PER_DAY_5M  # 15 weekdays, Wednesday 10th without bars


@pytest.mark.parametrize("chunk_days", [1, 2, 3])
def test_empty_weekend_window_does_not_end_the_broker_walk(chunk_days):
    three_weeks = days_between(START, dt.date(2024, 1, 19))
    result = _load(
        _client(days=three_weeks), end=dt.date(2024, 1, 19),
        provider_options={"chunk_days": chunk_days},
    )  # fmt: skip
    assert len(result.bars) == 15 * BARS_PER_DAY_5M
    assert result.meta.short_history is False


def test_window_of_todays_unfinished_session_does_not_end_the_broker_walk():
    today = dt.date(2024, 1, 5)
    before_the_first_bar_closes = at_ist(today, 9, 16)
    result = _load(
        _client(days=days_between(START, dt.date(2024, 1, 4))), end=today,
        now_ns=before_the_first_bar_closes, provider_options={"chunk_days": 1},
    )  # fmt: skip
    assert len(result.bars) == 4 * BARS_PER_DAY_5M


@pytest.mark.parametrize(
    "status, body", [(401, {"status": "error", "message": "Session expired"}), (502, None)]
)
def test_failure_of_an_older_window_is_an_error_not_short_history(status, body):
    client = _client()
    newest = broker_shape(truth_rows(CAL, "NSE", [END], step=300))
    client.queue(200, {"status": "success", "data": newest})
    client.queue(status, body)
    with pytest.raises(DataError, match=rf"2024-02-02 to 2024-04-01 .* failed \(HTTP {status}"):
        _load(client, broker="zerodha")


def test_not_json_404_from_the_store_is_reported_as_what_it_is():
    client = FakeClient()
    client.queue(404, None)
    with pytest.raises(DataError, match="HTTP 404: the body is not JSON"):
        _load(client, source="db")


# --- the broker name ----------------------------------------------------------------


def test_broker_is_asked_from_the_server_once_per_provider():
    client = _client("angel", days=DAYS[:10])
    provider = make_provider(client, CAL, NOW, source="api")
    for _ in range(2):
        provider.bars("X", "NSE", "5m", START, dt.date(2024, 1, 10))
    assert [endpoint for endpoint, _ in client.calls].count("ping") == 1


def test_broker_given_explicitly_is_not_asked_from_the_server():
    client = _client("angel", days=DAYS[:10])
    result = _load(client, end=dt.date(2024, 1, 10), broker="Zerodha")
    assert result.meta.broker == "zerodha"
    assert client.endpoints() == {"history"}


def test_unidentified_broker_uses_the_default_window_and_says_so():
    client = _client(broker=None, days=days_between(START, dt.date(2024, 2, 29)))
    result = _load(client, end=dt.date(2024, 2, 29))
    assert len(_windows(client)) == 2  # 60 days in default windows of 30
    assert result.meta.broker is None
    assert "the broker could not be identified" in result.meta.warnings[0]


def test_historify_fetch_does_not_ask_for_the_broker():
    rows = historify_shape(truth_rows(CAL, "NSE", DAYS[:5], step=300))
    client = FakeClient(db={("X", "NSE", "5m"): rows})
    _load(client, end=dt.date(2024, 1, 5), source="db")
    assert client.endpoints() == {"history"}


# --- pacing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broker, pacing, least_gap_seconds",
    [("kotak", None, 1.0), ("zerodha", None, 0.35), ("angel", None, 0.5), ("kotak", 2.5, 2.5)],
    ids=["kotak table", "zerodha table", "angel table", "pacing argument"],
)
def test_history_requests_are_never_closer_together_than_the_pacing(
    broker, pacing, least_gap_seconds
):
    clock = FakeClock()
    client = _client(broker, clock=clock.clock)
    _load(client, clock=clock, broker=broker, provider_options={"pacing": pacing})
    gaps = [b - a for a, b in zip(client.call_times, client.call_times[1:], strict=False)]
    assert gaps  # the range needs more than one request for every broker here
    assert min(gaps) == pytest.approx(least_gap_seconds)


def test_waiting_is_done_through_the_injected_sleep():
    clock = FakeClock()
    _load(_client("kotak", clock=clock.clock), clock=clock, broker="kotak")
    assert clock.sleeps == [1.0] * 5  # six requests, one second apart


@pytest.mark.parametrize("option", [{"pacing": -1}, {"chunk_days": 0}])
def test_negative_pacing_and_empty_window_are_config_errors(option):
    with pytest.raises(ConfigError):
        _load(_client(), provider_options=option)


# --- HTTP 429 -----------------------------------------------------------------------


def test_http_429_is_retried_with_doubling_waits():
    clock = FakeClock()
    client = _client(days=DAYS[:10])
    for _ in range(3):
        client.queue(429, BUSY)
    result = _load(client, end=dt.date(2024, 1, 10), clock=clock, broker="zerodha")
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert (result.meta.fetch.requests, result.meta.fetch.retries) == (4, 3)
    assert len(result.bars) == BARS_PER_DAY_5M * 8


@pytest.mark.parametrize(
    "retry_after, waited", [("7", 7.0), ("600", 20.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0)],
    ids=["longer than the backoff", "capped at backoff_max", "a date is ignored"],
)  # fmt: skip
def test_retry_after_header_lengthens_the_wait_up_to_the_cap(retry_after, waited):
    clock = FakeClock()
    client = _client(days=DAYS[:10])
    client.queue(429, BUSY, {"Retry-After": retry_after})
    _load(
        client, end=dt.date(2024, 1, 10), clock=clock, broker="zerodha",
        provider_options={"backoff_max": 20.0},
    )  # fmt: skip
    assert clock.sleeps == [waited]


def test_backoff_gives_up_after_max_retries_with_a_data_error():
    clock = FakeClock()
    client = _client(days=DAYS[:10])
    for _ in range(3):
        client.queue(429, BUSY)
    with pytest.raises(DataError, match="HTTP 429.*after 2 retries"):
        _load(
            client, end=dt.date(2024, 1, 10), clock=clock, broker="zerodha",
            provider_options={"max_retries": 2},
        )  # fmt: skip
    assert clock.sleeps == [1.0, 2.0]


# --- raw rows -----------------------------------------------------------------------


def test_raw_rows_keep_epoch_seconds_and_values_as_the_server_sent_them():
    rows = broker_shape(truth_rows(CAL, "NSE", [START], step=300))[:2]
    records = rows_to_records(rows)
    assert records["timestamp"].tolist() == [
        at_ist(START, 9, 15) // 10**9,
        at_ist(START, 9, 20) // 10**9,
    ]
    assert records["close"].tolist() == [rows[0]["close"], rows[1]["close"]]
    assert records["oi"].tolist() == [0.0, 0.0]  # the zero the server added is kept as it is


@pytest.mark.parametrize(
    "bad_rows, error, message",
    [
        (lambda rows: [dict(r, timestamp="2024-01-01 09:15:00+05:30") for r in rows],
         TimestampError, "expected epoch seconds"),
        (lambda rows: [dict(r, timestamp=r["timestamp"] * 1000) for r in rows],
         TimestampError, "milliseconds"),
        (lambda rows: [dict(r, timestamp=r["timestamp"] + 0.5) for r in rows],
         TimestampError, "whole epoch seconds"),
        (lambda rows: [{"timestamp": r["timestamp"], "open": 1.0} for r in rows],
         DataError, "lack the fields"),
        (lambda rows: [dict(r, close="n/a") for r in rows], DataError, "not a number"),
    ],
    ids=["date strings", "milliseconds", "fractions", "missing prices", "text price"],
)  # fmt: skip
def test_raw_rows_that_are_not_epoch_seconds_with_prices_are_rejected(bad_rows, error, message):
    rows = truth_rows(CAL, "NSE", [START])[:5]
    with pytest.raises(error, match=message):
        rows_to_records(bad_rows(rows))
