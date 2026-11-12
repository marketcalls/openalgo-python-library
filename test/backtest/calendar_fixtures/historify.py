"""Historify aggregation in pandas, written from the SQL of the OpenAlgo server.

Source: `database/historify_db.py` of the OpenAlgo server at commit d8cc13f24.

- Intraday buckets: `_get_aggregated_ohlcv`, lines 1044 to 1145. The bucket expression
  and the aggregates are the SELECT at lines 1106 to 1136.
- Market open per exchange: `EXCHANGE_MARKET_OPEN_SECONDS`, lines 1004 to 1014.
- The Parquet export drops rows before the market open: the extra WHERE clause at
  line 2563 of `export_to_parquet`.
- Weekly, monthly, quarterly and yearly bars from daily rows:
  `_get_daily_aggregated_ohlcv`, lines 1148 to 1246; the group expressions are at
  lines 1186 to 1204.

Timestamps are epoch seconds, as Historify stores them: an intraday row is the UTC
instant of the bar open and a daily row is 00:00 UTC of the IST session date.

This is the independent reference the engine's resampler is compared with. It shares
no code with `openalgo.backtest` and follows the SQL exactly: `FIRST(open ORDER BY
timestamp)` and `LAST(close ORDER BY timestamp)`, `LAST(oi ORDER BY timestamp)` take the
value of the first or last row of the bucket as it is, a missing value included, and
`MAX`, `MIN` and `SUM` skip missing values (SQL NULL), giving NULL when all are missing.
"""

import numpy as np
import pandas as pd

IST_OFFSET_SECONDS = 19_800  # ist_offset in both SQL builders
SECONDS_PER_DAY = 86_400

# EXCHANGE_MARKET_OPEN_SECONDS, lines 1004 to 1014.
MARKET_OPEN_SECONDS = {
    "NSE": 33_300,
    "BSE": 33_300,
    "NFO": 33_300,
    "BFO": 33_300,
    "CDS": 32_400,
    "BCD": 32_400,
    "MCX": 32_400,
    "NSE_INDEX": 33_300,
    "BSE_INDEX": 33_300,
}

# 1970-01-05, the Monday the weekly group expression anchors on (anchor = 4).
WEEK_ANCHOR_DAY = 4
PERIOD_MONTHS = {"M": 1, "Q": 3, "Y": 12}

VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "oi")


def _aggregate(rows: pd.DataFrame, bucket: np.ndarray) -> pd.DataFrame:
    """One row per bucket, in bucket order. `rows` are in timestamp order, so the first
    and last row of a bucket are the ones SQL `ORDER BY timestamp` picks."""
    frame = rows.assign(timestamp=bucket)
    grouped = frame.groupby("timestamp", sort=True)
    out = pd.DataFrame(
        {
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "volume": grouped["volume"].sum(min_count=1),
        }
    )
    first = frame.drop_duplicates("timestamp", keep="first").set_index("timestamp")
    last = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    out["open"], out["close"], out["oi"] = first["open"], last["close"], last["oi"]
    return out.reset_index()[["timestamp", *VALUE_COLUMNS]]


def intraday(
    rows: pd.DataFrame, exchange: str, interval_seconds: int, *, drop_pre_open: bool = False
) -> pd.DataFrame:
    """Aggregate 1-minute rows as `_get_aggregated_ohlcv` does.

    `rows` has `timestamp` (epoch seconds) and open, high, low, close, volume, oi.
    `drop_pre_open=True` adds the filter of the Parquet export.
    """
    market_open = MARKET_OPEN_SECONDS[exchange]
    ts = rows["timestamp"].to_numpy(dtype=np.int64)
    seconds_of_day = (ts + IST_OFFSET_SECONDS) % SECONDS_PER_DAY
    if drop_pre_open:
        keep = seconds_of_day >= market_open
        rows, ts, seconds_of_day = rows[keep], ts[keep], seconds_of_day[keep]
    day_start = (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY * SECONDS_PER_DAY - IST_OFFSET_SECONDS
    in_day = (seconds_of_day - market_open) // interval_seconds * interval_seconds
    return _aggregate(rows, day_start + market_open + in_day)


def period(rows: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate daily rows to "W", "M", "Q" or "Y" as `_get_daily_aggregated_ohlcv` does.

    The output `timestamp` is 00:00 UTC of the first calendar date of the period.
    """
    ts = rows["timestamp"].to_numpy(dtype=np.int64)
    day_number = (ts + IST_OFFSET_SECONDS) // SECONDS_PER_DAY
    if interval == "W":
        first_day = day_number - (day_number - WEEK_ANCHOR_DAY) % 7
    else:
        months = PERIOD_MONTHS[interval]
        # Months since 1970-01; 1970 * 12 is a multiple of 3 and 12, so the remainder
        # equals that of the SQL's EXTRACT(YEAR) * 12 + EXTRACT(MONTH) - 1.
        month_number = day_number.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
        first_month = month_number - month_number % months
        first_day = first_month.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    return _aggregate(rows, first_day * SECONDS_PER_DAY)


def from_canonical(bars: pd.DataFrame, *, daily: bool = False) -> pd.DataFrame:
    """Canonical bars as Historify rows: epoch seconds and the six value columns.

    With `daily=True` the timestamp becomes 00:00 UTC of the IST session date.
    """
    seconds = bars["ts_open"].to_numpy(dtype=np.int64) // 1_000_000_000
    if daily:
        seconds = (seconds + IST_OFFSET_SECONDS) // SECONDS_PER_DAY * SECONDS_PER_DAY
    columns = {name: bars[name].to_numpy() for name in VALUE_COLUMNS}
    return pd.DataFrame({"timestamp": seconds, **columns})
