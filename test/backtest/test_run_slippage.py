"""Slippage in a backtest (04-execution-and-market-model 4.3, decision SPEC-27): the
dated default table per class, the forms ticks, bps and percent with a one-tick floor,
user overrides, always against the fill's side and rounded to the 0.05 tick away from
the trader.

The round trip buys at an open of 1,000.00 and sells at an open of 1,100.00 (see
accounting_runs); every expected price below is worked from those two opens.
"""

import datetime as dt
from types import SimpleNamespace

import pytest
from accounting_runs import future_round_trip, round_trip

import openalgo.backtest as bt
from openalgo.backtest._market import market_settings
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import Instrument


def fill_prices(slippage):
    result = round_trip({"qty": 100}, costs=None, slippage=slippage)
    return list(result.fills["price_rupees"])


def test_cash_equity_takes_two_bps_by_default():
    # 2 bps of 1,000.00 is 0.20: 1,000.20. 2 bps of 1,100.00 is 0.22: 1,099.78, rounded
    # down to the tick for a seller, 1,099.75.
    assert fill_prices("default") == [1_000.2, 1_099.75]


def test_a_symbol_named_low_volume_takes_five_bps():
    # 5 bps: 1,000.00 + 0.50 and 1,100.00 - 0.55.
    table = bt.slippage.table(low_volume=("SBIN",))
    assert fill_prices(table) == [1_000.5, 1_099.45]


def test_an_override_replaces_the_row_of_its_class():
    # Three ticks of 0.05 either way.
    table = bt.slippage.table(overrides={"EQ_LARGE_MID": bt.ticks(3)})
    assert fill_prices(table) == [1_000.15, 1_099.85]


def test_a_mapping_is_shorthand_for_overrides():
    assert fill_prices({"EQ_LARGE_MID": bt.ticks(3)}) == [1_000.15, 1_099.85]


def test_percent_of_price_has_a_one_tick_floor():
    # 0.1 percent: 1.00 and 1.10. 0.001 percent is 0.01, below one tick: one tick.
    assert fill_prices(bt.pct(0.1)) == [1_001.0, 1_098.9]
    assert fill_prices(bt.pct(0.001)) == [1_000.05, 1_099.95]


def test_basis_points_have_a_one_tick_floor():
    assert fill_prices(bt.slippage.bps(2)) == [1_000.2, 1_099.75]
    # 0.1 bps of 1,000.00 is 0.01, below one tick.
    assert fill_prices(bt.slippage.bps(0.1)) == [1_000.05, 1_099.95]


def test_zero_slippage_fills_at_the_opens():
    assert fill_prices(0) == [1_000.0, 1_100.0]


def test_an_index_future_takes_one_tick():
    result = future_round_trip({"lots": 1}, costs=None)
    assert list(result.fills["price_rupees"]) == [1_000.05, 1_099.95]
    assert result.assumptions["slippage"] == "default table"


def _classes(*instruments):
    plan = SimpleNamespace(product="NRML", start=dt.date(2026, 11, 4), end=dt.date(2026, 11, 4))
    args = {"costs": None, "slippage": "default"}
    settings = market_settings(plan, list(instruments), args)
    return [row["slippage_class"] for row in settings.config["market"]], settings.warnings


def _derivative(kind, underlying):
    extra = {"strike": 500.0, "option_type": "CE"} if kind == "option" else {}
    return Instrument(
        symbol=f"{underlying}-{kind}",
        exchange="NFO",
        kind=kind,
        lot_size=750,
        tick_size=0.05,
        expiry=dt.date(2026, 11, 24),
        underlying=underlying,
        **extra,
    )


def test_derivatives_take_the_index_or_the_stock_row_by_their_underlying():
    classes, _ = _classes(
        _derivative("future", "BANKNIFTY"),
        _derivative("option", "NIFTY"),
        _derivative("future", "SBIN"),
        _derivative("option", "SBIN"),
    )
    assert classes == ["INDEX_FUT", "INDEX_OPT", "STOCK_FUT", "STOCK_OPT"]


def test_a_stock_underlying_is_named_in_the_warnings():
    _, warnings = _classes(_derivative("future", "SBIN"))
    assert any("SBIN is not a listed index underlying" in text for text in warnings)


def test_the_default_table_lists_every_class_with_its_source_and_status():
    rows = bt.slippage.defaults()
    assert [row["slippage_class"] for row in rows] == list(bt.slippage.CLASSES)
    assert len(rows) == 9
    # No exchange or broker publishes slippage, so every row is an OpenAlgo default.
    assert all(row["source"] and row["status"] == "default" for row in rows)


@pytest.mark.parametrize(
    "slippage",
    ["fast", {"NO_SUCH_CLASS": bt.ticks(1)}, bt.points(1), -1],
    ids=["unknown name", "unknown class", "points", "negative"],
)
def test_an_unknown_slippage_is_a_config_error(slippage):
    with pytest.raises(ConfigError, match="slippage"):
        round_trip({"qty": 100}, costs=None, slippage=slippage)
