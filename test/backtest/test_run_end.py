"""Where `bt.backtest` ends a run: at the earliest of `end`, the close of the last base
bar any instrument has, and the present (`now=`). Equity, trades and metrics stop
there, a position still open is reported open, and a warning names the end used.

One instrument, 1-minute bars on Tuesday 2025-07-01 from 09:15 IST (NSE session 09:15 to
15:30), a 15-minute MIS strategy that buys 100 at the first strategy-bar close and holds,
no charges and no slippage. The bar opening k minutes after 09:15 opens at 500.00 +
0.10 k and closes 0.05 higher.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
SESSION_MINUTES = 375
# The last of 165 bars opens at 11:59 and closes at 12:00.
UNTIL_NOON = 165


def minute_bars(count: int) -> pd.DataFrame:
    opens = 500.0 + 0.10 * np.arange(count)
    index = pd.DatetimeIndex(
        [SESSION_OPEN + dt.timedelta(minutes=k) for k in range(count)], name="timestamp"
    )
    columns = {
        "open": opens,
        "high": opens + 0.5,
        "low": opens - 0.5,
        "close": opens + 0.05,
        "volume": np.full(count, 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def at(hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime(2025, 7, 1, hour, minute, tzinfo=IST))


class BuyAndHold(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if self.position.is_flat and len(self.close) == 1:
            self.buy(qty=100)


def run(bars: int, **settings):
    arguments = {
        "data": minute_bars(bars),
        "capital": 1_000_000,
        "costs": None,
        "slippage": 0,
        "base": "1m",
        "product": "MIS",
        **settings,
    }
    return bt.backtest(BuyAndHold, **arguments)


def end_warnings(result) -> list[str]:
    return [line for line in result.warnings if line.startswith("The run ends at")]


def test_data_that_stop_at_noon_end_the_equity_series_at_noon():
    assert run(UNTIL_NOON).equity.index[-1] == at(12, 0)


def test_a_position_held_when_the_data_stop_is_reported_open_not_squared_off_eod():
    result = run(UNTIL_NOON)
    assert list(result.fills["reason"]) == ["entry"]
    assert len(result.trades) == 0
    held = result.final_positions.iloc[0]
    assert held["quantity_units"] == 100.0
    # The 11:59 bar (k = 164) closes at 500.00 + 16.40 + 0.05.
    assert held["mark_rupees"] == pytest.approx(516.45)


def test_the_metrics_end_where_the_run_ends():
    assert run(UNTIL_NOON).metrics["end"] == at(12, 0)


def test_a_warning_names_the_end_used_and_why():
    assert end_warnings(run(UNTIL_NOON)) == [
        "The run ends at 2025-07-01 12:00 IST, the close of the last bar in the data, "
        "before the end of its range at 2025-07-01 15:30 IST: equity, trades and metrics "
        "stop there, and positions still open are reported open"
    ]


def test_now_ends_the_run_at_the_last_slice_closed_by_then():
    result = run(SESSION_MINUTES, now=dt.datetime(2025, 7, 1, 11, 0, 30))
    assert result.equity.index[-1] == at(11, 0)
    assert end_warnings(result) == [
        "The run ends at 2025-07-01 11:00 IST, the present (now=), before the end of its "
        "range at 2025-07-01 15:30 IST: equity, trades and metrics stop there, and "
        "positions still open are reported open"
    ]


def test_data_that_reach_the_end_of_the_range_give_no_end_warning():
    result = run(SESSION_MINUTES)
    assert result.equity.index[-1] == at(15, 30)
    assert end_warnings(result) == []


def test_the_manifest_records_no_present_for_a_run_on_the_clock():
    assert run(UNTIL_NOON).manifest["arguments"]["now"] is None


def test_the_manifest_records_the_present_a_given_now_resolved_to():
    result = run(UNTIL_NOON, now=dt.datetime(2025, 7, 1, 11, 0, 30))
    # 11:00:30 IST is 05:30:30 UTC.
    expected = pd.Timestamp("2025-07-01 05:30:30", tz="UTC").value
    assert result.manifest["arguments"]["now"] == expected
