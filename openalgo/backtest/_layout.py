"""Canonical data layout shared by the data layer, resampling and the kernel.

Bars are a pandas DataFrame with a RangeIndex and exactly these columns:

    ts_open   int64    UTC nanoseconds, the instant the bar opens
    ts_close  int64    UTC nanoseconds, the instant the bar closes (a bar is visible
                       to a strategy only at ts_close)
    open, high, low, close   float64 rupees
    volume    float64
    oi        float64  open interest; NaN when unknown (never 0 for unknown)

Rows are sorted by ts_open with no duplicates. `df.attrs` carries "symbol",
"exchange" and "interval" (plus provider metadata such as "source").

Intervals use OpenAlgo strings: "1m", "5m", "15m", "1h", "D", "W", "M", "Q", "Y".
"""

from .errors import ConfigError

BAR_COLUMNS = ("ts_open", "ts_close", "open", "high", "low", "close", "volume", "oi")
PRICE_COLUMNS = ("open", "high", "low", "close")

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MINUTE = 60 * NANOS_PER_SECOND
IST_OFFSET_NANOS = (5 * 3600 + 30 * 60) * NANOS_PER_SECOND

_INTRADAY_UNITS = {"s": 1, "m": 60, "h": 3600}
_CALENDAR_UNITS = ("D", "W", "M", "Q", "Y")


def parse_interval(interval):
    """Split an OpenAlgo interval string into (kind, count).

    Intraday intervals return ("intraday", seconds): "1m" is ("intraday", 60) and
    "1h" is ("intraday", 3600). Calendar intervals return (unit, 1): "D" is ("D", 1).
    """
    if not isinstance(interval, str) or not interval:
        raise ConfigError(f"interval must be a string such as '5m' or 'D', got {interval!r}")
    if interval in _CALENDAR_UNITS:
        return interval, 1
    unit = interval[-1]
    count = interval[:-1]
    if unit in _INTRADAY_UNITS and count.isdecimal() and int(count) > 0:
        return "intraday", int(count) * _INTRADAY_UNITS[unit]
    raise ConfigError(
        f"unknown interval {interval!r}: use forms such as '1m', '5m', '15m', '1h', "
        "'D', 'W' or 'M' (minutes are a lower-case 'm', months an upper-case 'M')"
    )


def is_intraday(interval):
    return parse_interval(interval)[0] == "intraday"


__all__ = [
    "BAR_COLUMNS",
    "IST_OFFSET_NANOS",
    "NANOS_PER_MINUTE",
    "NANOS_PER_SECOND",
    "PRICE_COLUMNS",
    "is_intraday",
    "parse_interval",
]
