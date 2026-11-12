"""Charges in a backtest: every fill charged by the cost engine, contract-note rounding at
the session close, presets, calculator rounding, the DP charge and costs=None.

Expected figures come from `openalgo.backtest.costs` for the same trades (the cost
engine is tested against Zerodha's calculator in test_costs*.py) and from the worked
example of the reference note: an intraday round trip of 400 shares bought at 1,000.00
and sold at 1,100.00 on a 2026 trade date costs 200.62. See accounting_runs for the bars.
"""

import datetime as dt

import pytest
from accounting_runs import (
    DAILY_DAYS,
    DAY,
    ENTRY_PRICE,
    EXIT_PRICE,
    daily_round_trip,
    round_trip,
    session_bars,
)

import openalgo.backtest as bt
from openalgo.backtest import costs
from openalgo.backtest.errors import CostError

INTRADAY = "NSE_EQ_INTRADAY"
QTY = 400


def charged_round_trip(**settings):
    return round_trip({"qty": QTY}, slippage=0, **settings)


def test_each_fill_is_charged_what_the_cost_engine_charges_that_trade():
    result = charged_round_trip()
    expected = [
        costs.trade_cost(side, price, QTY, DAY, segment=INTRADAY).total
        for side, price in (("BUY", ENTRY_PRICE), ("SELL", EXIT_PRICE))
    ]
    assert list(result.costs["total"]) == pytest.approx(expected, abs=1e-9)
    assert list(result.fills["costs_rupees"]) == pytest.approx(expected, abs=1e-9)


def test_a_trades_charges_equal_the_round_trip_cost():
    result = charged_round_trip()
    trip = costs.round_trip_cost(ENTRY_PRICE, EXIT_PRICE, QTY, DAY, segment=INTRADAY)
    assert result.trades.loc[0, "costs_rupees"] == pytest.approx(trip.total, abs=1e-9)
    gross = result.trades.loc[0, "gross_pnl_rupees"]
    assert result.trades.loc[0, "net_pnl_rupees"] == pytest.approx(gross - trip.total)


def test_the_ledger_charges_the_worked_example_contract_note():
    result = charged_round_trip()
    assert result.equity["costs_rupees"].iloc[-1] == pytest.approx(200.62, abs=1e-9)
    notes = result.contract_notes
    assert list(notes["charged_total"]) == [200.62]
    assert notes.loc[0, "posted"] + notes.loc[0, "adjustment"] == pytest.approx(200.62)


def test_contract_notes_equal_the_cost_engines_aggregator():
    result = charged_round_trip()
    fills = result.costs
    trades = costs.trade_costs(
        fills["side"], fills["price_rupees"], fills["quantity_units"], fills["date"], INTRADAY
    )
    expected = costs.contract_note(trades)
    assert list(result.contract_notes["charged_total"]) == list(expected["total"])
    assert list(result.contract_notes["stt"]) == list(expected["stt"])


def test_calculator_rounding_charges_each_fill_as_the_calculator_does():
    model = costs.CostModel(rounding="calculator")
    result = charged_round_trip(costs=model)
    expected = [
        costs.trade_cost(side, price, QTY, DAY, segment=INTRADAY, rounding="calculator").total
        for side, price in (("BUY", ENTRY_PRICE), ("SELL", EXIT_PRICE))
    ]
    assert list(result.costs["total"]) == expected
    assert result.contract_notes.empty
    assert result.costs.attrs["rounding"] == "calculator"


def test_a_brokerage_preset_from_bt_costs_sets_the_brokerage():
    result = charged_round_trip(costs=bt.costs.flat(20))
    assert list(result.costs["brokerage"]) == [20.0, 20.0]
    assert result.assumptions["costs"].startswith("flat(20)")


def test_an_unknown_preset_is_a_cost_error():
    with pytest.raises(CostError, match="preset"):
        charged_round_trip(costs="no_such_broker")


def test_a_delivery_sell_from_holdings_pays_the_dp_charge():
    result = daily_round_trip(costs="zerodha")
    dp = costs.dp_charge(DAILY_DAYS[2]).total
    assert list(result.costs["dp_charge"]) == [0.0, dp]
    assert list(result.costs["date"]) == list(DAILY_DAYS[1:])
    trip = costs.round_trip_cost(
        ENTRY_PRICE,
        EXIT_PRICE,
        QTY,
        DAILY_DAYS[1],
        sell_date=DAILY_DAYS[2],
        segment="NSE_EQ_DELIVERY",
    )
    assert result.trades.loc[0, "costs_rupees"] == pytest.approx(trip.total + dp, abs=1e-9)


def test_a_delivery_buy_and_sell_on_one_day_is_charged_as_intraday_with_no_dp_charge():
    # The worked example traded on CNC: bought and sold the same day, so brokers charge
    # it as intraday (04-execution 2) and the note charges the intraday 200.62.
    result = charged_round_trip(product="CNC")
    trip = costs.round_trip_cost(ENTRY_PRICE, EXIT_PRICE, QTY, DAY, segment=INTRADAY)
    assert result.trades.loc[0, "costs_rupees"] == pytest.approx(trip.total, abs=1e-9)
    assert list(result.contract_notes["charged_total"]) == [200.62]
    assert result.equity["costs_rupees"].iloc[-1] == pytest.approx(200.62, abs=1e-9)
    assert list(result.costs["dp_charge"]) == [0.0, 0.0]
    assert list(result.costs["segment"]) == [INTRADAY, INTRADAY]


def test_a_cost_free_run_warns_on_the_result_and_on_the_tearsheet():
    result = charged_round_trip(costs=None)
    warning = "costs=None: no charges are modelled"
    assert any(text.startswith(warning) for text in result.warnings)
    assert warning in result.tearsheet()
    assert result.costs.empty and result.contract_notes.empty


def test_a_run_before_the_first_cost_schedule_stops_before_it_starts():
    # The first built-in schedule starts on 2020-07-01.
    with pytest.raises(CostError, match="charges are available from 2020-07-01.*pass costs=None"):
        round_trip({"qty": QTY}, data=session_bars(dt.date(2019, 7, 1)))


def test_a_run_in_2022_is_charged_at_the_rates_of_2022():
    # The worked example's trade on 2022-01-03 (schedule IN-2021-05-01): exchange Rs 3.45
    # per lakh of 8,40,000 = 28.98 plus IPFT Rs 0.00084, SEBI Rs 10 per crore = 0.84, GST
    # 18 percent of (40 + 28.98084 + 0.84) = 12.5677512; brokerage 40, STT 110, stamp 12.
    # Total 40 + 110 + 28.98 + 0.84 + 12 + 12.57 = 204.39.
    result = round_trip({"qty": QTY}, slippage=0, data=session_bars(dt.date(2022, 1, 3)))
    assert list(result.contract_notes["charged_total"]) == [204.39]
    assert set(result.costs["schedule"]) == {"IN-2021-05-01"}


def test_fills_on_unverified_rates_are_named_in_the_warnings():
    # The rates of schedule IN-2026-04-01 are verified only up to 2026-11-12.
    day = dt.date(2026, 12, 1)
    with pytest.warns(costs.UnverifiedScheduleWarning):
        result = round_trip({"qty": QTY}, slippage=0, data=session_bars(day), now="2026-12-10")
    named = "IN-2026-04-01 are past verification for 2 fills, from 2026-12-01"
    assert any(named in text for text in result.warnings)
    assert not result.costs["verified"].any()
