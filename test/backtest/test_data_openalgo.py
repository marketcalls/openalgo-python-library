"""The OpenAlgo provider: the three sources, short history, the partial candle and
the transport (FR-DATA-01, FR-DATA-02, FR-DATA-07). Every test uses an in-process
fake server; nothing here reaches a network.

The fake calendar has weekday sessions and one holiday (2024-01-26), so 2024-01-01
to 2024-05-31 holds 109 sessions: 110 weekdays less the holiday.
"""

import datetime as dt
import json

import httpx
import pytest
from data_fixtures.fakes import (
    NS,
    FakeCalendar,
    FakeClient,
    SdkOnlyClient,
    at_ist,
    broker_shape,
    daily_rows,
    days_between,
    historify_shape,
    make_provider,
    open_ns,
    server_load,
    truth_rows,
)

from openalgo import api as openalgo_api
from openalgo.backtest.data import DataProvider, OpenAlgoProvider
from openalgo.backtest.errors import ConfigError, DataError, NoDataError, TimestampError

START = dt.date(2024, 1, 1)
END = dt.date(2024, 5, 31)
DAYS = days_between(START, END)
SESSIONS_IN_RANGE = 109
TODAY = dt.date(2024, 6, 3)  # the Monday after the range
EVENING = at_ist(TODAY, 18, 0)
CAL = FakeCalendar(holidays=[dt.date(2024, 1, 26)])
BARS_PER_DAY_5M = 75  # 09:15 to 15:30 in five-minute bars
BROKER_VOLUME = 7  # marks bars served by the fake broker, to prove nothing is stitched


def _five_minute(days):
    return truth_rows(CAL, "NSE", days, step=300)


def _broker_client(days=DAYS, **options):
    return FakeClient(api={("X", "NSE", "5m"): broker_shape(_five_minute(days))}, **options)


def _both_sources(db_days, api_days=DAYS, db_daily_days=None):
    """A server whose Historify holds `db_days` and whose broker holds `api_days`."""
    broker_rows = broker_shape(_five_minute(api_days))
    for row in broker_rows:
        row["volume"] = BROKER_VOLUME
    daily_days = db_days if db_daily_days is None else db_daily_days
    return FakeClient(
        db={
            ("X", "NSE", "5m"): historify_shape(_five_minute(db_days)),
            ("X", "NSE", "D"): historify_shape(daily_rows(CAL, "NSE", daily_days)),
        },
        api={("X", "NSE", "5m"): broker_rows},
    )


def _load(client, source="auto", interval="5m", start=START, end=END, now_ns=EVENING, **options):
    return server_load(
        client, interval, start, end, calendar=CAL, now_ns=now_ns, source=source, cache=False,
        **options,
    )  # fmt: skip


# --- the provider contract ----------------------------------------------------------


def test_provider_satisfies_the_data_provider_protocol():
    provider = make_provider(FakeClient(), CAL, EVENING)
    assert isinstance(provider, DataProvider)
    assert provider.name == "openalgo"
    assert provider.capabilities == {"bars", "oi"}
    assert provider.timestamp_convention == "utc_open/ist_date"


@pytest.mark.parametrize(
    "client, options, message",
    [
        (FakeClient(), {"source": "yfinance"}, "source must be one of"),
        (None, {}, "an OpenAlgo client is required"),
        (object(), {}, "openalgo.api instance"),
    ],
    ids=["unknown source", "no client", "not a client"],
)
def test_provider_that_cannot_work_is_a_config_error(client, options, message):
    with pytest.raises(ConfigError, match=message):
        OpenAlgoProvider(client, **options)


def test_provider_bars_are_canonical_bars_of_the_configured_source():
    provider = make_provider(_broker_client(), CAL, EVENING, source="api")
    bars = provider.bars("X", "NSE", "5m", "2024-01-01", "2024-01-05")
    assert bars["ts_open"].iloc[0] == at_ist(START, 9, 15)
    assert bars["ts_close"].iloc[0] == at_ist(START, 9, 20)
    assert len(bars) == BARS_PER_DAY_5M * 5
    assert (bars.attrs["provider"], bars.attrs["source"]) == ("openalgo", "api")


def test_metadata_never_contains_the_api_key():
    client = _broker_client()
    result = _load(client, "api")
    assert client.keys_seen == {client.api_key}  # the key was sent ...
    assert client.api_key not in json.dumps(result.meta.to_dict())  # ... and kept out
    assert client.api_key not in repr(result.bars.attrs)


def test_source_openalgo_is_the_server_path_with_its_cache_and_warnings(tmp_path):
    client = FakeClient(broker="iifl", api={("X", "NSE", "5m"): broker_shape(_five_minute(DAYS))})
    result = server_load(
        client, "5m", START, END, calendar=CAL, now_ns=EVENING, source="openalgo",
        cache_dir=tmp_path, provider_options={"source": "api"},
    )  # fmt: skip
    assert (result.meta.source, result.meta.broker, result.meta.cache) == ("api", "iifl", "miss")
    assert result.meta.warnings == [
        "the timestamp convention of broker 'iifl' is not verified in the broker table; this "
        "series passed the session calendar check"
    ]


def test_source_openalgo_takes_the_providers_arguments_from_provider_options():
    client = _broker_client(DAYS[:5])
    result = server_load(
        None, "5m", START, dt.date(2024, 1, 5), calendar=CAL, now_ns=EVENING, source="openalgo",
        cache=False, provider_options={"client": client, "source": "api", "calendar": CAL},
    )  # fmt: skip
    assert (result.meta.source, len(result.bars)) == ("api", BARS_PER_DAY_5M * 5)


def test_source_openalgo_goes_through_the_cache_and_names_an_unverified_broker(tmp_path):
    client = _broker_client(DAYS[:5], broker="iifl")  # an XTS broker, not verified
    result = server_load(
        None, "5m", START, dt.date(2024, 1, 5), calendar=CAL, now_ns=EVENING, source="openalgo",
        cache_dir=tmp_path, provider_options={"client": client, "source": "api"},
    )  # fmt: skip
    assert (result.meta.cache, result.meta.broker) == ("miss", "iifl")
    assert any("broker 'iifl' is not verified" in note for note in result.meta.warnings)


# --- short history ------------------------------------------------------------------


def test_short_broker_history_is_announced_first_and_points_to_historify():
    held = [day for day in DAYS if day >= dt.date(2024, 4, 10)]
    meta = _load(_broker_client(held), "api").meta
    assert meta.short_history is True
    assert (meta.obtained_start, meta.obtained_end) == ("2024-04-10", "2024-05-31")
    assert meta.warnings[0] == (
        "Short history: 2024-01-01 to 2024-05-31 was requested but the broker (zerodha, "
        "source='api') returned 2024-04-10 to 2024-05-31, 38 of 109 sessions. A broker keeps "
        "only a limited history for this interval. Download the series into Historify and use "
        "source='db' for a longer range. The fetch stopped going back because the broker "
        "returned no bars for 2024-02-02 to 2024-04-01."
    )


def test_short_historify_history_is_announced_with_its_own_advice():
    held = [day for day in DAYS if dt.date(2024, 2, 1) <= day <= dt.date(2024, 4, 30)]
    client = FakeClient(db={("X", "NSE", "D"): historify_shape(daily_rows(CAL, "NSE", held))})
    meta = _load(client, "db", interval="D").meta
    assert meta.warnings[0] == (
        "Short history: 2024-01-01 to 2024-05-31 was requested but Historify (source='db') "
        "returned 2024-02-01 to 2024-04-30, 64 of 109 sessions. Historify does not hold the "
        "rest: download the missing range in Historify."
    )


@pytest.mark.parametrize(
    "source, message",
    [
        ("db", "Download the series in Historify first"),
        ("api", "expired contracts and very old ranges need Historify"),
        ("auto", "chose 'api' because Historify lacks a daily bar for 109 of 109 sessions"),
    ],
)
def test_source_without_bars_is_a_no_data_error_with_advice(source, message):
    with pytest.raises(NoDataError, match=message):
        _load(FakeClient(), source)


@pytest.mark.parametrize("source", ["auto", "db", "api"])
def test_range_that_starts_after_today_is_refused_without_any_request(source):
    client = _both_sources(DAYS)
    with pytest.raises(NoDataError, match="starts after today"):
        _load(client, source, start=dt.date(2030, 1, 1), end=dt.date(2030, 1, 31), broker="zerodha")
    assert client.calls == []


# --- the partial candle of today ----------------------------------------------------


@pytest.mark.parametrize(
    "now_hm, last_day, dropped",
    [((11, 30), "2024-05-31", 1), ((15, 30), "2024-06-03", 0)],
    ids=["during market hours", "after the close"],
)
def test_broker_daily_series_drops_todays_candle_only_while_the_session_runs(
    now_hm, last_day, dropped
):
    rows = broker_shape(daily_rows(CAL, "NSE", days_between(dt.date(2024, 5, 20), TODAY)))
    client = FakeClient(api={("X", "NSE", "D"): rows})
    meta = _load(
        client, "api", interval="D", start=dt.date(2024, 5, 20), end=TODAY,
        now_ns=at_ist(TODAY, *now_hm),
    ).meta  # fmt: skip
    assert meta.obtained_end == last_day
    assert meta.normalisation.partial_dropped == dropped
    assert meta.short_history is False  # today is not expected before its close


# --- source="auto" ------------------------------------------------------------------


def test_auto_uses_historify_when_the_probe_finds_it_complete():
    client = _both_sources(DAYS)
    result = _load(client)
    assert (result.meta.source, result.meta.source_requested) == ("db", "auto")
    assert (result.bars["volume"] != BROKER_VOLUME).all()
    assert client.history_calls("api") == []


def test_auto_probe_asks_historify_for_daily_bars_and_the_first_and_last_intraday_day():
    client = _both_sources(DAYS)
    probe = _load(client).meta.auto_probe
    assert client.history_calls("db")[:3] == [
        ("db", "D", "2024-01-01", "2024-05-31"),
        ("db", "5m", "2024-01-01", "2024-01-01"),
        ("db", "5m", "2024-05-31", "2024-05-31"),
    ]
    assert (probe.daily_expected, probe.daily_found) == (109, 109)
    assert probe.intraday_days_checked == ["2024-01-01", "2024-05-31"]


def test_auto_choosing_historify_for_intraday_bars_states_what_the_probe_cannot_see():
    meta = _load(_both_sources(DAYS)).meta
    assert meta.warnings == [
        "source='auto' chose Historify from its daily bars and the intraday bars of the first "
        "and the last session; the probe does not check the intraday bars of the days in "
        "between (see the missing_bars count of the quality report)"
    ]


def test_auto_uses_the_broker_for_the_whole_range_when_historify_lacks_one_session():
    hole = dt.date(2024, 3, 12)
    result = _load(_both_sources([day for day in DAYS if day != hole]))
    assert result.meta.source == "api"
    assert (result.bars["volume"] == BROKER_VOLUME).all()  # never stitched with Historify
    assert result.meta.obtained_start == "2024-01-01"
    assert result.meta.auto_probe.reason == (
        "Historify lacks a daily bar for 1 of 109 sessions (first missing: 2024-03-12)"
    )


def test_auto_uses_the_broker_when_historify_lacks_intraday_bars_on_the_last_day():
    client = _both_sources([day for day in DAYS if day != END], db_daily_days=DAYS)
    meta = _load(client).meta
    assert meta.source == "api"
    assert meta.auto_probe.reason == (
        "Historify has daily bars but no 5m bars inside the session of 2024-05-31"
    )


def test_auto_with_an_empty_historify_reports_the_brokers_short_history():
    recent = [day for day in DAYS if day >= dt.date(2024, 4, 22)]
    meta = _load(_both_sources([], api_days=recent)).meta
    assert meta.source == "api"
    assert meta.auto_probe.daily_found == 0
    assert (meta.short_history, meta.obtained_start) == (True, "2024-04-22")


def test_auto_does_not_require_todays_unfinished_session_from_historify():
    meta = _load(_both_sources(DAYS), end=TODAY, now_ns=at_ist(TODAY, 11, 0)).meta
    assert meta.source == "db"
    assert meta.auto_probe.daily_expected == SESSIONS_IN_RANGE
    assert meta.short_history is False


def test_probe_can_be_asked_without_fetching():
    forty_days = days_between(START, dt.date(2024, 2, 9))  # 29 sessions
    provider = make_provider(_both_sources(forty_days), CAL, EVENING)
    probe = provider.probe("X", "NSE", "5m", START, END)
    assert (probe.decision, probe.daily_found, probe.daily_expected) == ("api", 29, 109)


def test_failing_historify_probe_falls_to_the_broker_and_says_why():
    client = _both_sources(DAYS)
    client.queue(500, {"status": "error", "message": "database is locked"})
    meta = _load(client).meta
    assert meta.source == "api"
    assert meta.auto_probe.failed is True
    assert "HTTP 500: database is locked" in meta.auto_probe.reason
    assert meta.warnings[0].startswith("source='auto' could not check Historify")


# --- transport ----------------------------------------------------------------------


def test_requests_go_through_the_clients_pooled_http_client():
    client = _broker_client()
    _load(client, "api")
    assert client.options_seen
    assert all(options == (client.headers, client.timeout) for options in client.options_seen)


def test_only_the_read_only_endpoints_are_called():
    client = _both_sources(DAYS)
    _load(client)
    _load(client, "api")
    assert client.endpoints() == {"history", "ping"}


def test_real_sdk_client_is_read_through_its_own_pooled_http_client():
    """Drives the provider through `openalgo.api` itself, so that a rename of the
    attributes it relies on in the SDK's base class fails here."""
    fake = _broker_client(DAYS[:5])
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["content-type"]))
        answer = fake.handle(str(request.url), json.loads(request.content), None, None)
        return httpx.Response(answer.status_code, json=answer.json())

    client = openalgo_api(api_key=fake.api_key, host="http://openalgo.invalid")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = _load(client, "api", end=dt.date(2024, 1, 5))
    client.close()
    assert result.bars["ts_open"].tolist() == open_ns(_five_minute(DAYS[:5]))
    assert result.meta.broker == "zerodha"
    assert set(seen) == {
        ("POST", "/api/v1/ping", "application/json"),
        ("POST", "/api/v1/history", "application/json"),
    }
    assert fake.keys_seen == {fake.api_key}


def test_client_with_only_the_sdk_post_helper_is_supported():
    inner = FakeClient(db={("X", "NSE", "5m"): historify_shape(_five_minute(DAYS[:5]))})
    result = _load(SdkOnlyClient(inner), "db", end=dt.date(2024, 1, 5))
    assert len(result.bars) == BARS_PER_DAY_5M * 5


def test_http_429_is_backed_off_through_the_sdk_post_helper_too():
    inner = FakeClient(db={("X", "NSE", "5m"): historify_shape(_five_minute(DAYS[:5]))})
    inner.queue(429, {"status": "error", "message": "busy"})
    result = _load(SdkOnlyClient(inner), "db", end=dt.date(2024, 1, 5))
    assert result.meta.fetch.retries == 1


@pytest.mark.parametrize(
    "status, body, message",
    [
        (403, {"status": "error", "message": "Invalid openalgo apikey"}, "HTTP 403: Invalid"),
        (400, {"status": "error", "message": {"interval": ["Invalid interval"]}}, "HTTP 400"),
        (502, None, "HTTP 502: the body is not JSON"),
        (200, {"status": "error", "message": "master contract not ready"}, "HTTP 200: master"),
    ],
    ids=["bad key", "validation", "proxy error page", "error with status 200"],
)
def test_error_answer_of_the_server_is_a_data_error_without_the_api_key(status, body, message):
    client = _broker_client()
    client.queue(status, body)
    with pytest.raises(DataError, match=message) as caught:
        _load(client, "api", broker="zerodha")
    assert client.api_key not in str(caught.value)


def test_unreachable_server_is_a_data_error_that_keeps_the_cause():
    class Unreachable(FakeClient):
        def handle(self, url, payload, headers, timeout):
            raise OSError("connection refused")

    with pytest.raises(DataError, match="could not reach the OpenAlgo server") as caught:
        _load(Unreachable(), "api", broker="zerodha")
    assert isinstance(caught.value.__cause__, OSError)


def test_date_string_timestamps_from_the_server_are_rejected_not_parsed():
    served = broker_shape(_five_minute(DAYS[:1]))
    rows = [dict(row, timestamp="2024-01-01 09:15:00+05:30") for row in served]
    client = FakeClient(api={("X", "NSE", "5m"): rows})
    with pytest.raises(TimestampError, match="expected epoch seconds"):
        _load(client, "api", end=START)


def test_bars_close_one_interval_after_they_open():
    bars = _load(_broker_client(), "api", end=dt.date(2024, 1, 5)).bars
    assert (bars["ts_close"] - bars["ts_open"] == 300 * NS).all()
