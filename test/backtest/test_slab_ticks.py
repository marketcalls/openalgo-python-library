"""Ticks chosen by a price slab, resolved from the run's data (`_slab_ticks`), and the
margin check before any data is loaded (`_margin_check`).

From 2025-04-15 the NIFTY futures tick follows the index close on the last trading day
of the previous month (NSE/FAOP/67135): 0.05 below 15,000, 0.10 from 15,000 to 30,000,
0.20 above 30,000. The runs trade the July 2025 contract NIFTY31JUL25FUT (lot 75 from
the reference tables) on daily bars from 2025-07-01; the previous month's last session
is Monday 2025-06-30.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._margin_check import check_margin
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import Instrument

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
JULY = "NIFTY31JUL25FUT"
AUGUST = "NIFTY28AUG25FUT"


def _daily(rows: dict[dt.date, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([dt.datetime.combine(day, dt.time(9, 15), IST) for day in rows])
    closes = list(rows.values())
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 20.0 for c in closes],
            "low": [c - 20.0 for c in closes],
            "close": closes,
            "volume": 1_000.0,
        },
        index=index,
    )


def _strategy(symbol: str) -> type:
    class Idle(bt.Strategy):
        exchange, interval = "NFO", "D"

    Idle.symbol = symbol
    return Idle


def _run(data: dict, symbol: str = JULY, end: dt.date = dt.date(2025, 7, 2)):
    return bt.backtest(
        _strategy(symbol),
        capital=1_000_000,
        data=data,
        start=dt.date(2025, 7, 1),
        end=end,
        costs=None,
        slippage=0,
    )


JULY_BARS = {
    dt.date(2025, 6, 30): 25_000.0,
    dt.date(2025, 7, 1): 25_050.0,
    dt.date(2025, 7, 2): 25_100.0,
}


def test_without_the_underlying_the_contracts_own_close_chooses_the_slab():
    # The contract closed at 25,000.00 on 2025-06-30: the 15,000 to 30,000 slab, 0.10.
    result = _run({JULY: _daily(JULY_BARS)})
    assert result.manifest["instruments"][0]["tick_size"] == 0.10
    line = result.assumptions["instruments"]
    assert "tick 0.1 by the price slab of NIFTY31JUL25FUT's own close" in line
    assert "2025-06-30 (25000.0)" in line


def test_the_underlying_close_the_circular_names_comes_first():
    # The index closed at 31,000.00 on 2025-06-30 (the slab above 30,000: 0.20), though
    # the future's own close would give 0.10.
    index = _daily({dt.date(2025, 6, 30): 31_000.0})
    result = _run({JULY: _daily(JULY_BARS), "NIFTY": index})
    assert result.manifest["instruments"][0]["tick_size"] == 0.20
    assert "by the price slab of the NIFTY close on 2025-06-30" in result.assumptions["instruments"]


def test_without_a_reference_close_in_the_data_the_run_asks_for_the_tick():
    later = {day: close for day, close in JULY_BARS.items() if day.month == 7}
    with pytest.raises(ConfigError, match="no close of its underlying or of the contract"):
        _run({JULY: _daily(later)})


def test_a_tick_that_would_change_during_the_run_is_refused():
    # June-end 25,000.00 gives 0.10 for July; July-end 31,000.00 gives 0.20 for August.
    rows = {
        dt.date(2025, 6, 30): 25_000.0,
        dt.date(2025, 7, 1): 25_050.0,
        dt.date(2025, 7, 31): 31_000.0,
        dt.date(2025, 8, 1): 31_050.0,
    }
    with pytest.raises(ConfigError, match="changes during the run"):
        _run({AUGUST: _daily(rows)}, AUGUST, dt.date(2025, 8, 1))


def test_the_contracts_of_a_futures_series_take_their_slab_ticks_from_the_data():
    # NIFTY current month from 2025-07-28: July rolls into August on 07-30. Both are
    # traded in July, so both ticks follow the index close of 2025-06-30, 25,000.00.
    class Series(bt.Strategy):
        underlying, exchange, interval = "NIFTY", "NFO", "D"

    days = [dt.date(2025, 7, day) for day in (28, 29, 30, 31)]
    data = {
        JULY: _daily({day: 25_000.0 for day in days}),
        AUGUST: _daily({day: 25_100.0 for day in days[2:]}),
        "NIFTY": _daily({dt.date(2025, 6, 30): 25_000.0}),
    }
    result = bt.backtest(
        Series, capital=1_000_000, data=data, start=days[0], end=days[-1], costs=None
    )
    ticks = {row["symbol"]: row["tick_size"] for row in result.manifest["instruments"]}
    assert ticks == {JULY: 0.10, AUGUST: 0.10}
    assert list(result.rolls["to_symbol"]) == [AUGUST]


# ---------------------------------------------------------------- margin before data


def _future(symbol: str, exchange: str, underlying: str | None, lot: int) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange=exchange,
        kind="future",
        lot_size=lot,
        tick_size=1.0,
        expiry=dt.date(2025, 12, 19),
        underlying=underlying,
    )


def test_an_mcx_future_has_a_margin_row():
    crude = _future("CRUDEOIL19DEC25FUT", "MCX", "CRUDEOIL", 100)
    check_margin([crude], "NRML", (dt.date(2025, 12, 1),))


def test_a_missing_margin_row_stops_the_run_before_any_data_is_loaded():
    # No data and no client: the run would stop at loading the bars, so the margin
    # error shows the check ran first, with the engine's own message.
    unlisted = _future("UNLISTED19DEC25FUT", "MCX", None, 100)

    class Idle(bt.Strategy):
        symbol, exchange, interval = "UNLISTED19DEC25FUT", "MCX", "D"

    with pytest.raises(NotImplementedError, match="futures whose underlying has no row"):
        bt.backtest(
            Idle,
            capital=1_000_000,
            instrument=unlisted,
            start=dt.date(2025, 12, 1),
            end=dt.date(2025, 12, 5),
            costs=None,
        )
