"""`resample_bars` against an independent pandas implementation of Historify aggregation.

The reference is `calendar_fixtures/historify.py`, written from the SQL of the OpenAlgo
server, `database/historify_db.py` at commit d8cc13f24:

- `_get_aggregated_ohlcv`, lines 1044 to 1145 (bucket expression and aggregates in the
  SELECT at lines 1106 to 1136): 1-minute rows to N-minute and N-hour bars.
- `EXCHANGE_MARKET_OPEN_SECONDS`, lines 1004 to 1014: the bucket anchor per exchange.
- `export_to_parquet`, line 2563: the export's filter that drops pre-open rows.
- `_get_daily_aggregated_ohlcv`, lines 1148 to 1246 (group expressions at lines 1186 to
  1204): daily rows to W, M, Q and Y bars.

This shows equality with Historify's aggregation logic as read from that source, and
pins each documented difference. It is not a comparison with a running server.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from calendar_fixtures import historify
from calendar_fixtures.builders import NS_PER_SECOND, at_ist, daily_frame, opens_between, weekdays
from hypothesis import given, settings
from hypothesis import strategies as st

from openalgo.backtest._layout import parse_interval
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.resample import resample_bars

MONDAY = dt.date(2025, 6, 2)
WEEK = [MONDAY + dt.timedelta(days=offset) for offset in range(5)]
EVENING = dt.date(2024, 11, 1)
VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "oi"]
SESSION_HOURS = {
    "NSE": ((9, 15), (15, 30)),
    "NFO": ((9, 15), (15, 30)),
    "CDS": ((9, 0), (17, 0)),
    "MCX": ((9, 0), (23, 30)),
}


def random_walk_bars(ts_open: np.ndarray, exchange: str, seed: int) -> pd.DataFrame:
    """Canonical 1-minute bars with random prices, whole-number volume and open interest."""
    rng = np.random.default_rng(seed)
    rows = len(ts_open)
    close = 1_000.0 + np.cumsum(rng.normal(0.0, 1.5, rows)).round(2)
    open_ = np.concatenate([[1_000.0], close])[:rows]  # each bar opens at the prior close
    spread = rng.uniform(0.05, 2.0, rows).round(2)
    bars = pd.DataFrame(
        {
            "ts_open": ts_open,
            "ts_close": ts_open + 60 * NS_PER_SECOND,
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(0, 5_000, rows).astype(np.float64),
            "oi": rng.integers(10_000, 90_000, rows).astype(np.float64),
        }
    )
    bars.attrs.update(symbol="TEST", exchange=exchange, interval="1m")
    return bars


def week_with_gaps(exchange: str, seed: int) -> pd.DataFrame:
    """A week of 1-minute session bars with about one bar in six missing at random."""
    open_, close = SESSION_HOURS[exchange]
    ts_open = np.concatenate([opens_between(day, open_, close) for day in WEEK])
    keep = np.random.default_rng(seed).random(len(ts_open)) > 1 / 6
    return random_walk_bars(ts_open[keep], exchange, seed)


def assert_equals_historify(engine: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Same buckets, labelled by the same instant, with identical values."""
    np.testing.assert_array_equal(engine["ts_open"] // NS_PER_SECOND, reference["timestamp"])
    for column in VALUE_COLUMNS:
        np.testing.assert_array_equal(engine[column], reference[column], err_msg=column)


@pytest.fixture(scope="module")
def cal() -> Calendar:
    return Calendar(builtin=False)


@pytest.mark.parametrize("exchange", ["NSE", "NFO", "CDS", "MCX"])
@pytest.mark.parametrize(
    "interval", ["2m", "3m", "5m", "10m", "15m", "25m", "30m", "1h", "2h", "4h"]
)
def test_intraday_bars_equal_historify_on_session_bars(cal, exchange, interval):
    bars = week_with_gaps(exchange, seed=7)
    engine = resample_bars(bars, interval, exchange=exchange, calendar=cal)
    reference = historify.intraday(
        historify.from_canonical(bars), exchange, parse_interval(interval)[1]
    )
    assert_equals_historify(engine, reference)


@pytest.mark.parametrize("interval", ["5m", "15m", "1h"])
def test_with_pre_open_rows_the_engine_equals_the_historify_export(cal, interval):
    pre_open = opens_between(MONDAY, (9, 0), (9, 8))
    bars = random_walk_bars(
        np.concatenate([pre_open, opens_between(MONDAY, (9, 15), (15, 30))]), "NSE", seed=11
    )
    engine = resample_bars(bars, interval, exchange="NSE", calendar=cal)
    export = historify.intraday(
        historify.from_canonical(bars), "NSE", parse_interval(interval)[1], drop_pre_open=True
    )
    assert_equals_historify(engine, export)


def test_pre_open_rows_make_the_historify_rest_read_differ(cal):
    pre_open = opens_between(MONDAY, (9, 0), (9, 8))
    bars = random_walk_bars(
        np.concatenate([pre_open, opens_between(MONDAY, (9, 15), (9, 30))]), "NSE", seed=11
    )
    engine = resample_bars(bars, "5m", exchange="NSE", calendar=cal)
    rest_read = historify.intraday(historify.from_canonical(bars), "NSE", 300)
    assert len(rest_read) - len(engine) == 2  # the 09:00 and 09:05 buckets


def test_post_close_rows_reach_a_historify_session_bucket_but_not_the_engines(cal):
    rows = np.concatenate(
        [opens_between(MONDAY, (15, 15), (15, 30)), opens_between(MONDAY, (15, 30), (15, 40))]
    )
    bars = random_walk_bars(rows, "NSE", seed=13)
    engine = resample_bars(bars, "1h", exchange="NSE", calendar=cal)
    export = historify.intraday(historify.from_canonical(bars), "NSE", 3600, drop_pre_open=True)
    # Both have one 15:15 bucket; Historify's also holds the ten bars from 15:30.
    assert engine["ts_open"].tolist() == [at_ist(MONDAY, 15, 15)]
    assert len(export) == 1
    assert engine["close"].item() == bars["close"].iloc[14]
    assert export["close"].item() == bars["close"].iloc[-1]


@pytest.mark.parametrize("interval", ["5m", "30m", "1h"])
def test_a_special_session_off_the_lattice_equals_historify(interval):
    # 18:00 is 8 hours 45 minutes after 09:15, not a whole number of half hours.
    cal = Calendar(builtin=False)
    cal.add_session("NSE", EVENING, "18:00", "19:00")
    bars = random_walk_bars(opens_between(EVENING, (18, 0), (19, 0)), "NSE", seed=3)
    engine = resample_bars(bars, interval, exchange="NSE", calendar=cal)
    reference = historify.intraday(
        historify.from_canonical(bars), "NSE", parse_interval(interval)[1]
    )
    assert_equals_historify(engine, reference)


def test_a_missing_value_is_skipped_by_the_engine_and_taken_by_historify(cal):
    bars = random_walk_bars(opens_between(MONDAY, (9, 15), (9, 20)), "NSE", seed=17)
    bars.loc[4, "oi"] = np.nan
    engine = resample_bars(bars, "5m", exchange="NSE", calendar=cal)
    reference = historify.intraday(historify.from_canonical(bars), "NSE", 300)
    assert engine["oi"].item() == bars["oi"].iloc[3]  # the last known value
    assert np.isnan(reference["oi"].item())  # LAST(oi ORDER BY timestamp)


def period_start(day: dt.date, interval: str) -> dt.date:
    """First calendar date of the week, month, quarter or year that contains `day`."""
    if interval == "W":
        return day - dt.timedelta(days=day.weekday())
    months = {"M": 1, "Q": 3, "Y": 12}[interval]
    return dt.date(day.year, (day.month - 1) // months * months + 1, 1)


@pytest.mark.parametrize("interval", ["W", "M", "Q", "Y"])
def test_period_bars_equal_historify_grouping_and_values(cal, interval):
    daily = daily_frame(weekdays(dt.date(2024, 11, 18), dt.date(2026, 2, 13)))
    rng = np.random.default_rng(5)
    daily["volume"] = rng.integers(1_000, 9_000, len(daily)).astype(np.float64)
    daily["oi"] = rng.integers(10_000, 90_000, len(daily)).astype(np.float64)

    engine = resample_bars(daily, interval, exchange="NSE", calendar=cal)
    reference = historify.period(historify.from_canonical(daily, daily=True), interval)

    for column in VALUE_COLUMNS:
        np.testing.assert_array_equal(engine[column], reference[column], err_msg=column)
    # Historify labels a bar with the first calendar date of its period; the engine
    # opens it at the first session of that period.
    opened = [
        dt.datetime.fromtimestamp(ts // NS_PER_SECOND, dt.UTC).date() for ts in engine["ts_open"]
    ]
    labels = [dt.datetime.fromtimestamp(ts, dt.UTC).date() for ts in reference["timestamp"]]
    assert [period_start(day, interval) for day in opened] == labels


MINUTES_IN_NSE_SESSION = 375


@settings(max_examples=60, deadline=None)
@given(
    present=st.lists(
        st.booleans(), min_size=MINUTES_IN_NSE_SESSION, max_size=MINUTES_IN_NSE_SESSION
    ),
    minutes=st.sampled_from([2, 3, 5, 15, 25, 60, 120]),
)
def test_any_pattern_of_missing_minutes_equals_historify(present, minutes):
    ts_open = opens_between(MONDAY, (9, 15), (15, 30))[np.array(present)]
    bars = random_walk_bars(ts_open, "NSE", seed=1)
    engine = resample_bars(bars, f"{minutes}m", exchange="NSE", calendar=Calendar(builtin=False))
    reference = historify.intraday(historify.from_canonical(bars), "NSE", minutes * 60)
    assert_equals_historify(engine, reference)
