"""The walking skeleton of `bt.backtest`, end to end through the Rust kernel.

One instrument, 1-minute bars on 2025-07-01 from 09:15 IST, a 15-minute strategy, no
charges and no slippage. The bar opening k minutes after 09:15 opens at 500.00 + 0.10 k,
its high is the open plus 0.50, its low the open minus 0.50, its close the open plus
0.05, so every expected number below can be read off the inputs.
"""

import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
CAPITAL = 1_000_000


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


def at(hour: int, minute: int) -> dt.datetime:
    return dt.datetime(2025, 7, 1, hour, minute, tzinfo=IST)


class BuyThenExit(bt.Strategy):
    """Buys 100 at the first 15-minute close and exits at the next."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if self.position.is_flat and len(self.close) == 1:
            self.buy(qty=100)
        elif self.position.is_long:
            self.exit()


def run(strategy=BuyThenExit, bars=45, **settings):
    arguments = {
        "data": minute_bars(bars),
        "capital": CAPITAL,
        "costs": None,
        "slippage": 0,
        "base": "1m",
        "product": "CNC",
        "end": at(9, 15 + bars - 1) if bars <= 45 else None,
        **settings,
    }
    return bt.backtest(strategy, **arguments)


def test_the_entry_and_the_exit_fill_at_the_next_base_bar_open():
    fills = run().fills
    assert list(fills["time"]) == [at(9, 30), at(9, 45)]
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["price_rupees"]) == [501.5, 503.0]
    assert list(fills["reason"]) == ["entry", "signal"]


def test_the_gross_pnl_is_the_difference_of_the_two_opens():
    result = run()
    assert result.gross_pnl["realized"] == pytest.approx(150.0)
    assert result.gross_pnl["unrealized"] == 0.0
    assert result.equity["equity_rupees"].iloc[-1] == pytest.approx(CAPITAL + 150.0)


def test_the_round_trip_is_one_trade():
    trades = run().trades
    assert len(trades) == 1
    assert trades.loc[0, "direction"] == "long"
    assert trades.loc[0, "net_pnl_rupees"] == pytest.approx(150.0)


def test_the_position_history_has_a_row_per_change():
    history = run().positions
    assert list(history["time"]) == [at(9, 30), at(9, 45)]
    assert list(history["quantity_units"]) == [100.0, 0.0]


def test_callbacks_receive_the_events_before_on_bar():
    seen = []

    class Recording(BuyThenExit):
        def on_fill(self, fill):
            seen.append((fill["name"], fill["symbol"], fill["time"]))

    run(Recording)
    assert seen == [("Filled", "SBIN", at(9, 30)), ("Filled", "SBIN", at(9, 45))]


def test_a_class_attribute_wins_over_the_run_argument():
    class EarlySquareOff(BuyThenExit):
        square_off = "09:40"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(qty=100)

    fills = run(EarlySquareOff, product="MIS", square_off="15:10").fills
    assert list(fills["time"]) == [at(9, 30), at(9, 40)]
    assert list(fills["reason"]) == ["entry", "scheduler"]


def test_a_last_bar_signal_is_rejected_as_market_closed():
    handles = []

    class LastBar(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            if bar.time == at(15, 15):
                handles.append(self.buy(qty=10))

    result = run(LastBar, bars=375)
    assert [(order.status, order.reject_code) for order in handles] == [
        ("Rejected", "market_closed")
    ]
    assert result.fills.empty


def test_an_exit_while_flat_places_no_order():
    handles = []

    class ExitFlat(BuyThenExit):
        def on_bar(self, bar):
            handles.append(self.exit())

    run(ExitFlat)
    assert handles and all(handle is None for handle in handles)


@pytest.mark.parametrize(
    "order_arguments",
    [{"stop": bt.points(4)}, {"price": 501.0}, {"validity": "gtt"}],
    ids=["leg stop", "limit order", "gtt"],
)
def test_parts_not_built_yet_say_so(order_arguments):
    class Unbuilt(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            self.buy(qty=10, **order_arguments)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        run(Unbuilt)


def test_live_is_phase_2():
    with pytest.raises(NotImplementedError, match="Phase 2"):
        bt.live(BuyThenExit, None, capital=CAPITAL, max_daily_loss=1_000)


def test_capital_must_be_positive():
    with pytest.raises(ConfigError, match="capital"):
        run(capital=0)


def test_importing_the_package_does_not_load_the_run_layer():
    code = (
        "import sys, openalgo.backtest as bt; "
        "loaded = [m for m in ('openalgo.backtest.run', 'openalgo.backtest.strategy') "
        "if m in sys.modules]; "
        "assert not loaded, loaded; "
        "assert bt.Strategy.__name__ == 'Strategy'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
