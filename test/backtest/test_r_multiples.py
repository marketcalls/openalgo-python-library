"""R-multiples (12-position-sizing 6, FR-SIZE-09): each closed trade's net P&L over its
initial risk, the loss its opening fill would take at the stop its entry set, and the
summary of their distribution.

Bars: 1-minute SBIN bars on 2025-07-01 whose opens are given below, each with a high and
low 0.50 either side of its open and a close at its open. The strategy decides after
bar n - 1 and fills at the open of bar n:

    bar     0      1      2      3      4      5      6     7 ...
    open  100.00 100.00 104.00 104.00 103.00 103.00 105.00 105.00

Three trades of 10 shares on MIS, no charges, no slippage:

    trade 1  buys at 100.00 with a stop 2.00 below (risk 20.00), sells at 104.00: +40.00, R 2
    trade 2  buys at 104.00 with a stop 2.00 below (risk 20.00), sells at 103.00: -10.00, R -0.5
    trade 3  buys at 103.00 with no stop, sells at 105.00: +20.00, no R
"""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from strategy_samples import CAPITAL, at

import openalgo.backtest as bt

OPENS = [100.0, 100.0, 104.0, 104.0, 103.0, 103.0, 105.0]
COUNT = 30


def bars() -> pd.DataFrame:
    opens = np.array(OPENS + [OPENS[-1]] * (COUNT - len(OPENS)))
    index = pd.DatetimeIndex(
        [at(9, 15) + dt.timedelta(minutes=k) for k in range(COUNT)], name="timestamp"
    )
    columns = {
        "open": opens,
        "high": opens + 0.5,
        "low": opens - 0.5,
        "close": opens,
        "volume": np.full(COUNT, 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


class ThreeTrades(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "MIS"

    def on_bar(self, bar):
        n = len(self.close)
        if n in (1, 3):
            self.buy(qty=10, stop=bt.points(2))
        elif n == 5:
            self.buy(qty=10)
        elif n in (2, 4, 6):
            self.exit()


class Quiet(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "MIS"


def run(strategy: type) -> bt.Result:
    return bt.backtest(strategy, data=bars(), capital=CAPITAL, costs=None, slippage=0)


@pytest.fixture(scope="module")
def result() -> bt.Result:
    return run(ThreeTrades)


def test_each_trade_carries_its_initial_risk(result):
    risk = result.trades["initial_risk_rupees"].tolist()
    assert risk[:2] == [20.0, 20.0]
    assert math.isnan(risk[2])


def test_each_trade_r_is_its_net_pnl_over_its_initial_risk(result):
    r = result.trades["r_multiple"].tolist()
    assert r[:2] == [2.0, -0.5]
    assert math.isnan(r[2])


def test_the_r_summary_describes_the_trades_with_an_initial_risk(result):
    summary = result.r_summary
    assert summary["trades"] == 3
    assert summary["trades_with_risk"] == 2
    assert summary["mean_r"] == 0.75
    assert summary["median_r"] == 0.75
    assert summary["best_r"] == 2.0
    assert summary["worst_r"] == -0.5
    assert summary["total_r"] == 1.5
    assert summary["positive_share"] == 0.5
    # Sample standard deviation of 2 and -0.5: sqrt(2 x 1.25^2 / 1).
    assert summary["std_r"] == pytest.approx(math.sqrt(3.125))
    # One-R bins named by their lower edge: -0.5 is in [-1, 0), 2 in [2, 3).
    assert summary["distribution"] == {-1: 1, 2: 1}


def test_a_run_without_trades_has_an_empty_r_summary():
    summary = run(Quiet).r_summary
    assert (summary["trades"], summary["trades_with_risk"]) == (0, 0)
    assert math.isnan(summary["mean_r"])
    assert summary["distribution"] == {}


def test_the_full_tearsheet_has_an_r_multiples_section(result):
    assert "id='r-multiples'" in result.tearsheet()


def test_the_tearsheet_has_no_r_multiples_section_without_an_initial_risk():
    assert "id='r-multiples'" not in run(Quiet).tearsheet()
