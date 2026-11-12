"""Every table of a `Result` has the same columns whether or not it has rows (14-strategy-api
5.2), so code reading a result never has to test for an empty frame first.

Both runs use the rising 1-minute bars of `strategy_samples` on 2025-07-01 (bar k opens
at 500.00 + 0.10 k, high and low 0.50 either side), with the 09:18 bar missing. `Quiet`
never orders. `Busy` buys 10 at its first close with a 1-point target (a risk event, a
closed trade and a stale mark while it holds through the missing bar), and asks for an
order a sizer skips; it pays the default charges, so it has costs and contract notes.

An order without a reject code, a tag or a sizer has NaN in that column of the orders
table, in every run: `ExplicitQuantity` and `DefaultSize` each buy at the first close and
exit at the third, the first with `qty=10` (no order has a sizer), the second with the
engine default size (the entry's sizer is "default", the exit has none).
"""

import math

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


class ExplicitQuantity(Quiet):
    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=10)
        elif len(self.close) == 3:
            self.exit()


class DefaultSize(Quiet):
    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy()
        elif len(self.close) == 3:
            self.exit()


ORDER_TEXT_COLUMNS = ("reject_code", "tag", "sizer")


@pytest.fixture(scope="module")
def explicit_orders():
    return run(ExplicitQuantity).orders


@pytest.fixture(scope="module")
def default_size_orders():
    return run(DefaultSize).orders


def _missing(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def test_with_an_explicit_quantity_no_order_has_a_sizer(explicit_orders):
    sizers = explicit_orders["sizer"].tolist()
    assert len(sizers) == 2
    assert all(_missing(sizer) for sizer in sizers)


def test_with_the_default_size_the_exit_has_no_sizer_marked_the_same_way(default_size_orders):
    sizers = default_size_orders["sizer"].tolist()
    assert sizers[0] == "default"
    assert _missing(sizers[1])


@pytest.mark.parametrize("column", ORDER_TEXT_COLUMNS)
def test_an_optional_text_column_of_the_orders_holds_a_string_or_nan_in_every_run(
    column, explicit_orders, default_size_orders
):
    for orders in (explicit_orders, default_size_orders):
        assert orders[column].dtype == object
        assert all(isinstance(value, str) or _missing(value) for value in orders[column])
