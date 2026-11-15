"""Candles of a special session that opens off the bar lattice are an error that names
the session and says what to do.

Bars of interval N sit on a lattice from each date's regular open (09:15 IST on NSE), on
every date, special sessions included. A source that starts a special session's candles
at the session's own open puts them off that lattice: the built-in calendar's NSE Muhurat
session of Sunday 2026-11-08 runs from 14:20 to 15:30 IST, so 15-minute candles from
14:20 (14:20, 14:35, 14:50, 15:05 and 15:20, five bars) cannot be placed. The error gives
the session's name and date from the calendar data, its open and the lattice's, and the
two ways out: 1-minute data, from which the engine builds the session's 15-minute bars,
or leaving that session's bars out.

The own data is a 15-minute SBIN series of 1,000.00 flat bars on the regular sessions of
2026-11-05, 06, 09 and 10 and, unless left out, the special session.
"""

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import TimestampError

IST = "Asia/Kolkata"
REGULAR_DAYS = ("2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10")


def candles(stamps: pd.DatetimeIndex) -> pd.DataFrame:
    flat = pd.Series(1_000.0, index=stamps)
    columns = {"open": flat, "high": flat + 1, "low": flat - 1, "close": flat}
    return pd.DataFrame(columns).assign(volume=10_000.0)


def session_stamps(day: str, first: str, step: str) -> pd.DatetimeIndex:
    """Bar opens from `first` to the last one before 15:30 IST on `day`."""
    return pd.date_range(f"{day} {first}", f"{day} 15:29", freq=step, tz=IST)


def own_bars(special: tuple[str, str] | None, step: str = "15min") -> pd.DataFrame:
    """The regular sessions, plus `special` (date, first bar open) when given."""
    days = [session_stamps(day, "09:15", step) for day in REGULAR_DAYS]
    if special is not None:
        days.append(session_stamps(*special, step))
    return candles(days[0].append(days[1:]).sort_values())


class Holds(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "CNC"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(qty=1)


def run(data: pd.DataFrame, **settings) -> bt.Result:
    return bt.backtest(Holds, data=data, capital=100_000, costs=None, slippage=0, **settings)


def test_candles_of_a_special_session_off_the_lattice_are_told_by_the_session():
    with pytest.raises(TimestampError) as raised:
        run(own_bars(("2026-11-08", "14:20")))
    assert str(raised.value) == (
        "NSE:SBIN 15m: 5 bars are not on the NSE bar lattice: the session of 2026-11-08 "
        "(Diwali Laxmi Pujan (Muhurat trading)) opens at 14:20 IST, off the 09:15 bar "
        "lattice, so 15-minute candles of that session cannot be placed. Pass 1-minute "
        "data (the engine builds the session's 15-minute bars from it), or leave that "
        "session's bars out."
    )


def test_the_session_is_named_as_the_calendar_names_it():
    calendar = bt.calendar.default_calendar().copy()
    calendar.add_session("NSE", "2026-11-07", "10:05", "15:30", name="Exchange test day")
    with pytest.raises(TimestampError) as raised:
        run(own_bars(("2026-11-07", "10:05")), calendar=calendar)
    assert "the session of 2026-11-07 (Exchange test day) opens at 10:05 IST" in str(raised.value)


def test_one_minute_data_of_the_same_sessions_runs():
    # The engine builds the 15-minute bars, the session's first from 14:15 to 14:30.
    result = run(own_bars(("2026-11-08", "14:20"), step="1min"), base="1m")
    muhurat = result.equity[result.equity.index.date == pd.Timestamp("2026-11-08").date()]
    assert len(muhurat) == 70


def test_the_same_data_without_that_session_runs():
    # The one buy of the strategy fills: the series loads and the run trades.
    assert len(run(own_bars(None)).fills) == 1


def test_off_lattice_bars_of_a_regular_session_keep_the_lattice_message():
    with pytest.raises(TimestampError, match="does not open at the regular open"):
        run(own_bars(("2026-11-11", "09:20")))
