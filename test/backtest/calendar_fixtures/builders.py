"""Instants and bar frames for the calendar and resampling tests.

Nothing here calls the code under test: instants are computed from epoch arithmetic
and bars are built row by row, so expected values in the tests stay independent.
"""

import datetime as dt

import numpy as np
import pandas as pd

NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60
SECONDS_PER_DAY = 86_400
EPOCH = dt.date(1970, 1, 1)

BAR_VOLUME = 10.0  # volume of every bar built by `bars_frame`
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "D": 86_400}


def at_ist(day: dt.date, hour: int, minute: int = 0) -> int:
    """UTC nanoseconds of an IST clock time on `day`."""
    seconds = (day - EPOCH).days * SECONDS_PER_DAY + hour * 3600 + minute * 60
    return (seconds - IST_OFFSET_SECONDS) * NS_PER_SECOND


def epoch_ms(day: dt.date, hour: int, minute: int = 0) -> int:
    """Epoch milliseconds of an IST clock time, as OpenAlgo holiday windows use."""
    return at_ist(day, hour, minute) // 1_000_000


def opens_between(
    day: dt.date, start: tuple[int, int], end: tuple[int, int], step_minutes: int = 1
) -> np.ndarray:
    """Bar opens every `step_minutes` from `start` up to, not including, `end` (IST)."""
    first, last = at_ist(day, *start), at_ist(day, *end)
    return np.arange(first, last, step_minutes * NS_PER_MINUTE, dtype=np.int64)


def nse_session(day: dt.date, step_minutes: int = 1) -> np.ndarray:
    """Bar opens across the regular NSE session, 09:15 to 15:30 IST."""
    return opens_between(day, (9, 15), (15, 30), step_minutes)


def bars_frame(
    ts_open, *, exchange: str = "NSE", interval: str = "1m", first_open: float = 100.0
) -> pd.DataFrame:
    """Canonical bars at `ts_open` with prices that count up one rupee per row.

    Row i has open `first_open + i`, high two above it, low two below it and close one
    above it; volume is `BAR_VOLUME` and open interest is unknown. `ts_close` is one
    interval after `ts_open`.
    """
    ts_open = np.asarray(ts_open, dtype=np.int64)
    opens = first_open + np.arange(len(ts_open), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "ts_open": ts_open,
            "ts_close": ts_open + INTERVAL_SECONDS[interval] * NS_PER_SECOND,
            "open": opens,
            "high": opens + 2.0,
            "low": opens - 2.0,
            "close": opens + 1.0,
            "volume": np.full(len(ts_open), BAR_VOLUME),
            "oi": np.full(len(ts_open), np.nan),
        }
    )
    frame.attrs.update(symbol="TEST", exchange=exchange, interval=interval)
    return frame


def daily_frame(days: list[dt.date], *, exchange: str = "NSE", open_at=(9, 15)) -> pd.DataFrame:
    """One canonical daily bar per date, opening at `open_at` IST."""
    ts_open = [at_ist(day, *open_at) for day in days]
    return bars_frame(ts_open, exchange=exchange, interval="D")


def weekdays(first: dt.date, last: dt.date) -> list[dt.date]:
    """Every Monday to Friday from `first` to `last`, both inclusive."""
    span = (last - first).days + 1
    days = (first + dt.timedelta(days=offset) for offset in range(span))
    return [day for day in days if day.weekday() < 5]


def bars_per_output(frame: pd.DataFrame) -> list[int]:
    """How many `bars_frame` rows went into each resampled bar, from its volume."""
    return [int(volume / BAR_VOLUME) for volume in frame["volume"]]
