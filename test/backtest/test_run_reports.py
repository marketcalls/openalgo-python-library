"""The reports of a backtest: `Result.performance` and `Result.metrics` from the metrics
package over the run's equity and trades, `Result.tearsheet()`, and the quick start of
10-simple-path section 2 on its defaults with no server.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from accounting_runs import CAPITAL, IST, at, round_trip, round_trip_strategy, session_bars

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.costs import COMPONENTS


@pytest.fixture(scope="module")
def charged():
    """The intraday round trip of 400 shares on the default costs and slippage."""
    return round_trip({"qty": 400})


def test_the_metrics_are_computed_from_the_runs_equity_and_trades(charged):
    metrics, trades = charged.metrics, charged.trades
    assert metrics["trades"] == 1 and metrics["open_trades"] == 0
    assert metrics["start_equity"] == CAPITAL
    final = charged.equity["equity_rupees"].iloc[-1]
    assert metrics["net_profit"] == pytest.approx(final - CAPITAL)
    assert metrics["gross_pnl"] == pytest.approx(trades.loc[0, "gross_pnl_rupees"])
    assert metrics["total_costs"] == pytest.approx(trades.loc[0, "costs_rupees"])


def test_the_trades_table_is_in_the_layout_the_metrics_package_reads(charged):
    trade = charged.performance.trades.iloc[0]
    assert (trade["side"], trade["quantity"]) == ("BUY", 400.0)
    assert trade["entry_time"] == at(9, 30) and trade["exit_time"] == at(9, 45)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["costs"])


def test_the_cost_breakdown_is_the_runs_charges(charged):
    breakdown = charged.performance.cost_breakdown
    for name in COMPONENTS:
        assert breakdown[name] == pytest.approx(charged.costs[name].sum())
    assert sum(breakdown.values()) == pytest.approx(charged.trades.loc[0, "costs_rupees"])


def test_a_position_still_open_at_the_end_is_an_open_trade():
    class BuyAndHold(round_trip_strategy(qty=400)):
        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(qty=400)

    result = bt.backtest(
        BuyAndHold, data=session_bars(), capital=CAPITAL, base="1m", product="CNC", slippage=0
    )
    trades = result.performance.trades
    assert result.metrics["open_trades"] == 1 and result.metrics["trades"] == 0
    assert pd.isna(trades.loc[0, "exit_time"]) and trades.loc[0, "entry_time"] == at(9, 30)
    assert trades.loc[0, "costs"] == pytest.approx(result.costs["total"].sum())


def test_the_tearsheet_is_returned_or_written_with_the_assumptions(charged, tmp_path):
    html = charged.tearsheet()
    assert html.startswith("<!DOCTYPE html>") and "RoundTrip" in html
    assert "zerodha (contract-note rounding)" in html
    target = tmp_path / "report.html"
    assert charged.tearsheet(target) == str(target)
    assert target.read_text(encoding="utf-8") == html


def _five_minute_bars(first: dt.date, last: dt.date) -> pd.DataFrame:
    """A seeded random walk at 5 minutes over the NSE sessions from `first` to `last`."""
    calendar = default_calendar()
    days = [d.date() for d in pd.date_range(first, last) if calendar.session("NSE", d.date())]
    rng = np.random.default_rng(11)
    stamps, closes = [], []
    price = 800.0
    for day in days:
        for k in range(75):
            stamps.append(dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST))
            stamps[-1] += dt.timedelta(minutes=5 * k)
            price *= 1 + rng.normal(0, 0.002)
            closes.append(price)
    close = np.round(np.array(closes) / 0.05) * 0.05
    open_ = np.concatenate(([close[0]], close[:-1]))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.5,
            "low": np.minimum(open_, close) - 0.5,
            "close": close,
            "volume": np.full(len(close), 10_000.0),
        },
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )
    return frame


class EmaCross(bt.Strategy):
    """The quick start of 10-simple-path section 2, in the parts built so far: the
    averages come from `openalgo.ta` over the revealed closes (in place of `self.I` and
    `crossed_above`), and orders carry no stop or target (leg rules are another
    workstream). Orders take the engine default size."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    fast, slow = 10, 30

    def on_bar(self, bar):
        closes = self.close.values
        if len(closes) <= self.slow:
            return
        fast, slow = ta.ema(closes, self.fast), ta.ema(closes, self.slow)
        if fast[-2] <= slow[-2] and fast[-1] > slow[-1] and self.position.is_flat:
            self.buy()
        elif fast[-2] >= slow[-2] and fast[-1] < slow[-1] and self.position.is_long:
            self.exit()


def test_the_quick_start_runs_on_its_defaults_with_no_server():
    df = _five_minute_bars(dt.date(2026, 10, 5), dt.date(2026, 10, 30))
    result = bt.backtest(EmaCross, data=df, capital=200_000)
    assert result.assumptions["costs"] == "zerodha (contract-note rounding)"
    assert result.assumptions["slippage"] == "default table"
    assert len(result.trades) > 0 and result.costs["total"].sum() > 0
    assert set(result.contract_notes["note_group"]) == {"NSE_EQ"}
    assert "sharpe" in result.metrics and "max_drawdown" in result.metrics
    assert result.tearsheet().startswith("<!DOCTYPE html>")
