"""`Result.trade_list()`: the closed trades as a trader reads them, checked on the
hand-worked round trips of `accounting_runs`.

Intraday (MIS, 1-minute base, no charges, no slippage): 400 shares bought at the 09:30
open at 1,000.00 and sold at the 09:45 open at 1,100.00, so the gross P&L is
400 x 100 = 40,000.00 rupees on an entry value of 400 x 1,000 = 4,00,000.00, a return
of 10 percent, held one bar of the strategy's 15 minutes (15 one-minute base bars),
15 minutes. Daily (CNC): bought at the second session's open and sold at the third
session's, one bar and one calendar day apart.
"""

import datetime as dt

import pandas as pd
import pytest
from accounting_runs import CAPITAL, DAILY_DAYS, at, daily_round_trip, round_trip, session_bars

import openalgo.backtest as bt
from openalgo.backtest._trade_list import TRADE_LIST_COLUMNS

ENTRY, EXIT, UNITS = 1_000.0, 1_100.0, 400.0


@pytest.fixture(scope="module")
def long_trade() -> pd.Series:
    listing = round_trip({"qty": 400}, costs=None, slippage=0).trade_list()
    assert len(listing) == 1
    return listing.iloc[0]


def test_the_list_has_every_column_in_order_indexed_by_trade_number():
    listing = round_trip({"qty": 400}, costs=None, slippage=0).trade_list()
    assert list(listing.columns) == list(TRADE_LIST_COLUMNS)
    assert listing.index.name == "trade" and list(listing.index) == [1]


def test_a_long_trade_reads_its_symbol_direction_product_and_times(long_trade):
    assert (long_trade["symbol"], long_trade["direction"], long_trade["product"]) == (
        "SBIN",
        "long",
        "MIS",
    )
    assert long_trade["entry_time"] == at(9, 30)
    assert long_trade["exit_time"] == at(9, 45)
    assert long_trade["entry_time"].utcoffset() == dt.timedelta(hours=5, minutes=30)


def test_a_long_trade_reads_its_quantity_prices_and_pnl(long_trade):
    assert long_trade["quantity"] == UNITS
    assert (long_trade["entry_price"], long_trade["exit_price"]) == (ENTRY, EXIT)
    assert long_trade["gross_pnl"] == pytest.approx(40_000.0)
    assert long_trade["charges"] == 0.0
    assert long_trade["net_pnl"] == pytest.approx(40_000.0)


def test_the_return_is_the_net_pnl_over_the_entry_value_in_percent(long_trade):
    # 40,000 over 4,00,000 is 10 percent.
    assert long_trade["return_pct"] == pytest.approx(10.0)


def test_an_intraday_trade_is_held_one_strategy_bar_and_fifteen_minutes(long_trade):
    # A 15-minute strategy on 1-minute base bars: one 15-minute bar, not 15 base bars.
    assert long_trade["bars_held"] == 1
    assert long_trade["days_held"] == pytest.approx(15 / (24 * 60))


def five_minute_trade() -> bt.Result:
    """A 5-minute strategy on 1-minute base bars that buys 400 shares at its first bar
    close (09:20, filled at the 09:20 open) and exits at its 13th (10:20, filled at the
    10:20 open): held 60 minutes, 12 five-minute bars, 60 one-minute bars."""

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=400)
        elif len(self.close) == 13:
            self.exit()

    attributes = {"symbol": "SBIN", "exchange": "NSE", "interval": "5m", "on_bar": on_bar}
    strategy = type("FiveMinuteTrip", (bt.Strategy,), attributes)
    arguments = {"data": session_bars(), "capital": CAPITAL, "base": "1m", "product": "MIS"}
    return bt.backtest(strategy, costs=None, slippage=0, **arguments)


def test_each_trade_counts_the_strategy_bars_of_its_own_symbol():
    # Two symbols on 15-minute bars: both bought at 09:30, AAA sold at 09:45 (one bar)
    # and BBB at 10:00 (two bars).
    class TwoSymbols(bt.Strategy):
        symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "15m", "MIS"

        def on_bar(self, bars):
            closes = len(self.data["AAA"].close)
            if closes == 1:
                self.buy(qty=10, symbol="AAA")
                self.buy(qty=10, symbol="BBB")
            elif closes == 2:
                self.exit("AAA")
            elif closes == 3:
                self.exit("BBB")

    data = {"AAA": session_bars(), "BBB": session_bars()}
    result = bt.backtest(TwoSymbols, data=data, capital=CAPITAL, base="1m", costs=None, slippage=0)
    listing = result.trade_list().set_index("symbol")
    assert (listing.loc["AAA", "bars_held"], listing.loc["BBB", "bars_held"]) == (1, 2)


def test_bars_held_counts_the_strategys_bars_not_the_base_bars():
    result = five_minute_trade()
    trade = result.trade_list().iloc[0]
    assert (trade["entry_time"], trade["exit_time"]) == (at(9, 20), at(10, 20))
    assert trade["bars_held"] == 12
    assert list(result.performance.trades["bars_held"]) == [12]


def test_the_exit_reason_and_r_multiple_are_read(long_trade):
    assert long_trade["exit_reason"] == "signal"
    assert pd.isna(long_trade["r_multiple"])  # opened without a stop


def test_a_short_trade_loses_when_the_price_rises():
    def on_bar(self, bar):
        if len(self.close) == 1:
            self.sell(qty=400)
        elif len(self.close) == 2 and not self.position.is_flat:
            self.exit()

    attributes = {"symbol": "SBIN", "exchange": "NSE", "interval": "15m", "on_bar": on_bar}
    strategy = type("ShortTrip", (bt.Strategy,), attributes)
    result = bt.backtest(
        strategy,
        data=session_bars(),
        capital=CAPITAL,
        base="1m",
        product="MIS",
        costs=None,
        slippage=0,
    )
    trade = result.trade_list().iloc[0]
    assert trade["direction"] == "short"
    # Sold 400 at 1,000.00 and bought back at 1,100.00: -40,000 on 4,00,000 sold.
    assert trade["gross_pnl"] == pytest.approx(-40_000.0)
    assert trade["return_pct"] == pytest.approx(-10.0)


def test_charges_are_the_charges_of_the_trades_fills():
    result = round_trip({"qty": 400})
    trade = result.trade_list().iloc[0]
    assert trade["charges"] > 0
    assert trade["charges"] == pytest.approx(result.costs["total"].sum())
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["charges"])


def test_a_daily_delivery_trade_is_held_one_bar_and_one_day():
    trade = daily_round_trip().trade_list().iloc[0]
    assert trade["product"] == "CNC"
    second, third = DAILY_DAYS[1], DAILY_DAYS[2]
    assert trade["entry_time"].date() == second and trade["exit_time"].date() == third
    assert trade["bars_held"] == 1
    assert trade["days_held"] == pytest.approx(1.0)


def test_a_run_without_trades_gives_an_empty_list_with_every_column():
    def on_bar(self, bar):
        return None

    attributes = {"symbol": "SBIN", "exchange": "NSE", "interval": "15m", "on_bar": on_bar}
    strategy = type("Quiet", (bt.Strategy,), attributes)
    result = bt.backtest(strategy, data=session_bars(), capital=CAPITAL, base="1m")
    listing = result.trade_list()
    assert listing.empty and list(listing.columns) == list(TRADE_LIST_COLUMNS)


def test_a_position_open_at_the_end_is_not_listed():
    class BuyAndHold(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(qty=400)

    result = bt.backtest(
        BuyAndHold, data=session_bars(), capital=CAPITAL, base="1m", product="CNC", slippage=0
    )
    assert result.metrics["open_trades"] == 1
    assert result.trade_list().empty


def test_the_tearsheet_lists_the_same_trades():
    result = round_trip({"qty": 400}, costs=None, slippage=0)
    page = result.tearsheet()
    trades = page[page.index("<table class='tbl hover trades'") :]
    trades = trades[: trades.index("</table>")]
    assert trades.count("<tr data-i=") == len(result.trade_list())
    assert "&#8377;40,000.00" in trades and "10.00%" in trades
