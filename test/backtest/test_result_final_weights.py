"""What a beginner reads off a run without working it out by hand: the value and weight
of each position held at the end, and whole trade counts in an optimisation table.

The runs use `accounting_runs`' daily bars: one NSE share on 2026-11-02, 03 and 04,
opening (and closing) at 1,000.00, 1,000.00 and 1,100.00; capital 10,00,000.00; no
charges and no slippage, so every figure is worked by hand.
"""

import math

import pytest
from accounting_runs import CAPITAL, NIFTY_FUTURE, daily_bars, session_bars

import openalgo.backtest as bt


class BuyAndHold(bt.Strategy):
    """Buys 100 shares at the first close (filled at the second open, 1,000.00) and
    holds them to the end (marked at the last close, 1,100.00)."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(qty=100)


def held_to_the_end():
    return bt.backtest(
        BuyAndHold, data=daily_bars(), capital=CAPITAL, base="D", costs=None, slippage=0
    )


def test_a_position_held_at_the_end_shows_its_market_value():
    # 100 shares marked at 1,100.00.
    final = held_to_the_end().final_positions
    assert final.loc[final["symbol"] == "SBIN", "value_rupees"].item() == 110_000.0


def test_a_position_held_at_the_end_shows_its_weight_in_percent_of_equity():
    # Equity at the end: 10,00,000.00 + 100 x (1,100.00 - 1,000.00) = 10,10,000.00, so
    # the holding of 1,10,000.00 is 10.891089... percent of it.
    final = held_to_the_end().final_positions
    weight = final.loc[final["symbol"] == "SBIN", "weight_pct"].item()
    assert weight == pytest.approx(110_000.0 / 1_010_000.0 * 100, rel=1e-12)


class HoldsAFuture(bt.Strategy):
    """Buys one lot (75 units) of the NIFTY future at its first bar close (filled at the
    09:30 open, 1,000.00) and holds it to the end (marked at 1,100.00)."""

    symbol, exchange, interval = NIFTY_FUTURE.symbol, "NFO", "15m"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(lots=1)


def test_a_future_held_at_the_end_has_no_value_or_weight():
    # Its contract value, 75 x 1,100.00 = 82,500.00, is not money the account holds: the
    # account holds the margin and the open profit, which equity already counts.
    result = bt.backtest(
        HoldsAFuture,
        data=session_bars(minutes=385),
        instrument=NIFTY_FUTURE,
        capital=CAPITAL,
        base="1m",
        product="NRML",
        costs=None,
        slippage=0,
    )
    held = result.final_positions.iloc[0]
    assert held["quantity_units"] == 75.0
    assert math.isnan(held["value_rupees"]) and math.isnan(held["weight_pct"])


class EveryOtherBar(bt.Strategy):
    """Buys at the first close and exits at the second: one trade per run."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    hold = 1

    def on_bar(self, bar):
        if self.position.is_flat and len(self.close) == 1:
            self.buy(qty=10)
        elif self.position.is_long:
            self.exit()


def test_an_optimisation_table_counts_trades_in_whole_numbers():
    table = bt.optimize(
        EveryOtherBar,
        data=daily_bars(),
        capital=CAPITAL,
        base="D",
        costs=None,
        slippage=0,
        workers=1,
        hold=(1, 2),
    ).table
    assert table["trades"].tolist() == [1, 1]
    assert all(isinstance(count, int) for count in table["trades"].tolist())
