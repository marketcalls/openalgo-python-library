"""The intraday times in the Assumptions block are the times that applied over the whole
run, not only in its first session (SPEC-32, TIMES-1, `rust/oa_bt/src/engine/session.rs`).

Two idle MIS runs on the built-in calendar, 5-minute bars flat at 100.00 over every
session of the run:

* MCX from 2026-10-29 to 2026-11-03. The session closes at 23:30 IST until United States
  daylight saving ends (Sunday 2026-11-01) and at 23:55 from 2026-11-02; MCX entries end
  30 minutes and the square-off comes 20 minutes before each session's close, so the
  square-off moves from 23:10 to 23:35.
* NSE cash from 2026-07-30 to 2026-08-04. The dated defaults move on 2026-08-03 (decision
  TIMES-1): entry until 15:00 and square-off 15:10 before, 14:50 and 15:00 from then.
"""

import datetime as dt

import pandas as pd

import openalgo.backtest as bt
from openalgo.backtest import calendar
from openalgo.backtest.reference import Instrument

BAR_NS = 5 * 60 * 1_000_000_000
MCX_DAYS = (
    dt.date(2026, 10, 29),
    dt.date(2026, 10, 30),
    dt.date(2026, 11, 2),
    dt.date(2026, 11, 3),
)
NSE_DAYS = (dt.date(2026, 7, 30), dt.date(2026, 7, 31), dt.date(2026, 8, 3), dt.date(2026, 8, 4))
CRUDE = Instrument(
    symbol="CRUDEOIL19NOV26FUT",
    exchange="MCX",
    kind="future",
    lot_size=100,
    tick_size=1.0,
    expiry=dt.date(2026, 11, 19),
    underlying="CRUDEOIL",
)
TIMES = ("entry_from", "entry_until", "square_off")


def flat_bars(exchange: str, days: tuple[dt.date, ...]) -> pd.DataFrame:
    """5-minute bars over each session of `days`, from its open to its close."""
    stamps = []
    for day in days:
        opens, closes = calendar.default_calendar().session(exchange, day)
        stamps += range(opens, closes, BAR_NS)
    index = pd.to_datetime(stamps, utc=True).tz_convert("Asia/Kolkata")
    prices = {name: 100.0 for name in ("open", "high", "low", "close")}
    return pd.DataFrame({**prices, "volume": 1.0}, index=index)


def times(symbol: str, exchange: str, days: tuple[dt.date, ...], **settings) -> dict:
    strategy = type(
        "Idle",
        (bt.Strategy,),
        {"symbol": symbol, "exchange": exchange, "interval": "5m", "product": "MIS"},
    )
    arguments = {"capital": 1_000_000, "costs": None, "slippage": 0, "base": "5m", **settings}
    result = bt.backtest(strategy, data=flat_bars(exchange, days), **arguments)
    return {name: result.assumptions[name] for name in TIMES}


def test_mcx_times_that_follow_the_close_are_stated_as_the_rule():
    assert times("CRUDEOIL19NOV26FUT", "MCX", MCX_DAYS, instrument=CRUDE) == {
        "entry_from": "09:00",
        "entry_until": "30 minutes before each session's close",
        "square_off": "20 minutes before each session's close",
    }


def test_mcx_times_within_one_season_are_one_clock_time():
    assert times("CRUDEOIL19NOV26FUT", "MCX", MCX_DAYS[:2], instrument=CRUDE) == {
        "entry_from": "09:00",
        "entry_until": "23:00",
        "square_off": "23:10",
    }


def test_nse_times_that_change_on_a_date_are_stated_per_period():
    assert times("SBIN", "NSE", NSE_DAYS) == {
        "entry_from": "09:15",
        "entry_until": "15:00 from 2026-07-30 to 2026-07-31; 14:50 from 2026-08-03 to 2026-08-04",
        "square_off": "15:10 from 2026-07-30 to 2026-07-31; 15:00 from 2026-08-03 to 2026-08-04",
    }


def test_a_given_time_applies_to_every_session():
    stated = times("SBIN", "NSE", NSE_DAYS, square_off="14:45")
    assert stated["square_off"] == "14:45"
