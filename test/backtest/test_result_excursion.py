"""MAE, MFE and slippage on a `Result`: the kernel's figures in `trades` and `fills`, in
percent of the capital at work, in `trade_list()` and in the metrics layout.

The intraday round trip of `accounting_runs`: 400 shares bought at the 09:30 open
(1,000.00) and sold at the 09:45 open (1,100.00); every bar before 09:45 has its high
and low 1.00 either side of 1,000.00. With the default slippage the buy fills at
1,000.20 and the sell at 1,099.75. Held through the 09:30 to 09:44 bars, the worst
running P&L is at a low of 999.00: 400 x (999.00 - 1,000.20) = -480.00. The exit's bar
counts its open only: 400 x (1,100.00 - 1,000.20) = 39,920.00, not its high of 1,101.00.
The capital at work is 1,000.20 x 400 = 4,00,080.00. The buy's slippage is 0.20 x 400
= 80.00 and the sell's 0.25 x 400 = 100.00.

A gold future on MCX (multiplier 100: a one-rupee move is Rs 100 a unit), one unit
bought at the 11-03 open and sold at the 11-05 open, no slippage; the bars are below.
"""

import dataclasses
import datetime as dt

import pandas as pd
import pytest
from accounting_runs import CAPITAL, round_trip, session_bars
from test_mcx_future_run import CRUDEOIL, DAYS

import openalgo.backtest as bt

CAPITAL_AT_WORK = 1_000.20 * 400


@pytest.fixture(scope="module")
def slipped() -> bt.Result:
    return round_trip({"qty": 400})


def test_a_trade_carries_its_mae_and_mfe_in_rupees_and_in_percent_of_capital_at_work(slipped):
    trade = slipped.trades.iloc[0]
    assert (trade["mae_rupees"], trade["mfe_rupees"]) == (-480.0, 39_920.0)
    assert trade["mae_pct"] == pytest.approx(-480.0 / CAPITAL_AT_WORK * 100)
    assert trade["mfe_pct"] == pytest.approx(39_920.0 / CAPITAL_AT_WORK * 100)


def test_every_fill_records_what_its_slippage_cost_and_the_trade_their_sum(slipped):
    assert list(slipped.fills["slippage_rupees"]) == [80.0, 100.0]
    assert slipped.trades.loc[0, "slippage_rupees"] == 180.0


def test_slippage_is_in_the_gross_pnl_already_and_is_not_deducted_again(slipped):
    # 400 x (1,099.75 - 1,000.20) = 39,820.00; without slippage the round trip makes
    # 40,000.00, which is that plus the 180.00 of slippage.
    frictionless = round_trip({"qty": 400}, slippage=0).trades.loc[0, "gross_pnl_rupees"]
    trade = slipped.trades.iloc[0]
    assert trade["gross_pnl_rupees"] == pytest.approx(39_820.0)
    assert trade["gross_pnl_rupees"] + trade["slippage_rupees"] == pytest.approx(frictionless)


def test_the_trades_keep_their_columns_and_add_the_new_ones_at_the_end(slipped):
    earlier = [
        "symbol",
        "entry_time",
        "exit_time",
        "trade_id",
        "instrument",
        "product",
        "direction",
        "entry_slice",
        "entry_ts",
        "exit_slice",
        "exit_ts",
        "entry_units",
        "max_units",
        "entry_price_rupees",
        "exit_price_rupees",
        "gross_pnl_rupees",
        "costs_rupees",
        "net_pnl_rupees",
        "exit_reason",
        "fills",
        "initial_risk_rupees",
        "r_multiple",
    ]
    added = ["mae_rupees", "mfe_rupees", "slippage_rupees", "mae_pct", "mfe_pct"]
    assert list(slipped.trades.columns) == earlier + added
    assert list(slipped.fills.columns)[-2:] == ["filled_at_last_close", "slippage_rupees"]


def test_the_trade_list_carries_them_in_its_own_names(slipped):
    row = slipped.trade_list().iloc[0]
    assert (row["mae"], row["mfe"], row["slippage"]) == (-480.0, 39_920.0, 180.0)
    assert row["mae_pct"] == pytest.approx(-480.0 / CAPITAL_AT_WORK * 100)


def test_the_metrics_layout_carries_them_and_an_open_trade_has_none_yet():
    class BuyAndHold(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "15m"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(qty=400)

    result = bt.backtest(
        BuyAndHold, data=session_bars(), capital=CAPITAL, base="1m", product="CNC", slippage=0
    )
    trades = result.performance.trades
    assert trades["exit_time"].isna().all()
    for name in ("mae", "mfe", "mae_pct", "mfe_pct", "slippage"):
        assert trades[name].isna().all(), name


# ----- a contract with a multiplier ---------------------------------------------------

GOLD = dataclasses.replace(
    CRUDEOIL,
    symbol="GOLD05DEC26FUT",
    lot_size=1,
    tick_size=1.0,
    multiplier=100.0,
    expiry=dt.date(2026, 12, 5),
    underlying="GOLD",
)
# (open, high, low, close) of 11-02 to 11-05. Bought at the 11-03 open, 72,010.00: that
# bar counts whole, 100 x (71,800 - 72,010) = -21,000.00 and 100 x (72,300 - 72,010) =
# 29,000.00; 11-04 too (72,400.00: 39,000.00); sold at the 11-05 open, 72,500.00:
# 49,000.00, not its high of 72,900.00.
GOLD_BARS = [
    (72_000.0, 72_100.0, 71_900.0, 72_000.0),
    (72_010.0, 72_300.0, 71_800.0, 72_200.0),
    (72_200.0, 72_400.0, 72_100.0, 72_350.0),
    (72_500.0, 72_900.0, 72_450.0, 72_800.0),
]


class GoldSwing(bt.Strategy):
    """Buys one unit at the first close and exits at the third."""

    symbol, exchange, interval = GOLD.symbol, "MCX", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=1)
        elif len(self.close) == 3:
            self.exit()


@pytest.fixture(scope="module")
def gold() -> bt.Result:
    frame = pd.DataFrame(GOLD_BARS, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in DAYS]).tz_localize("Asia/Kolkata")
    return bt.backtest(
        GoldSwing, data=frame, capital=5_000_000, instrument=GOLD, product="NRML", slippage=0
    )


def test_a_future_counts_its_multiplier_in_the_excursion_and_the_capital_at_work(gold):
    trade = gold.trades.iloc[0]
    assert (trade["mae_rupees"], trade["mfe_rupees"]) == (-21_000.0, 49_000.0)
    assert trade["gross_pnl_rupees"] == 49_000.0
    # The capital at work is 72,010.00 x 1 x 100 = 72,01,000.00.
    assert trade["mae_pct"] == pytest.approx(-21_000 / 7_201_000 * 100)
    assert trade["mfe_pct"] == pytest.approx(49_000 / 7_201_000 * 100)


def test_a_futures_turnover_and_return_count_its_multiplier(gold):
    # (72,010 + 72,500) x 1 x 100; the net return over 72,01,000.00.
    assert gold.metrics["turnover"] == pytest.approx(14_451_000.0)
    row = gold.trade_list().iloc[0]
    assert row["return_pct"] == pytest.approx(row["net_pnl"] / 7_201_000 * 100)
