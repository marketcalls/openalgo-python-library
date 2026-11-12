"""Every table of a `Result` has the same columns whether or not it has rows (14-strategy-api
5.2), so code reading a result never has to test for an empty frame first.

Both runs use the rising 1-minute bars of `strategy_samples` on 2025-07-01 (bar k opens
at 500.00 + 0.10 k, high and low 0.50 either side), with the 09:18 bar missing. `Quiet`
never orders. `Busy` buys 10 at its first close with a 1-point target (a risk event, a
closed trade and a stale mark while it holds through the missing bar), and asks for an
order a sizer skips; it pays the default charges, so it has costs and contract notes.
"""

import pytest
from strategy_samples import CAPITAL, minute_bars

import openalgo.backtest as bt

TABLES = (
    "fills",
    "orders",
    "trades",
    "positions",
    "events",
    "risk_events",
    "stale_marks",
    "final_positions",
    "costs",
    "contract_notes",
    "skipped",
)


class Quiet(bt.Strategy):
    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "MIS"


class Busy(Quiet):
    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=10, target=bt.points(1))
            self.buy(size=bt.size.value(1.0))


def run(strategy):
    data = minute_bars(count=30, skip=(3,))
    return bt.backtest(strategy, data=data, capital=CAPITAL, quantity_mode="venue")


@pytest.fixture(scope="module")
def quiet():
    return run(Quiet)


@pytest.fixture(scope="module")
def busy():
    return run(Busy)


@pytest.mark.parametrize("name", TABLES)
def test_a_table_has_the_same_columns_with_and_without_rows(name, quiet, busy):
    empty, full = getattr(quiet, name), getattr(busy, name)
    assert len(empty) == 0 and len(full) > 0
    assert list(empty.columns) == list(full.columns)


def test_an_empty_fills_table_has_an_ist_time_column(quiet, busy):
    assert quiet.fills["time"].dtype == busy.fills["time"].dtype
