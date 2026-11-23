"""An NSE currency futures trade end to end from the built-in reference tables (item C-4
of 2.4.1): a USDINR future on the user's own 1-minute bars, one lot.

USDINR29DEC26FUT: one lot is USD 1,000 quoted in rupees per dollar, and quantity counts
dollars, as OpenAlgo counts it (its Charges API prices BUY 1000 USDINR29DEC26FUT at 88.5 as
a trade value of 88,500), so the `currency_contracts` table gives lot 1,000 and multiplier
1: a one-rupee move is Rs 1,000 a lot, and qty=1000 is one lot. One CDS session,
2026-11-20 (09:00 to 17:00 IST), NRML, the default costs in contract-note rounding, no
slippage, capital 10,00,000. The strategy buys one lot at the first bar's close and exits
at the third's; each order fills at the next bar's open:

    bars 0 to 2   88.3300   the buy fills at bar 1's open, 88.3300
    bars 3 on     the exit price, at which the sell fills at bar 3's open

Expected charges are worked by hand from the CDS_FUT row of schedule IN-2026-04-01 (the
broker's charges page and calculator, read 2026-11-23): brokerage 0.03 percent or Rs 20
an order (Rs 88,330 is worth Rs 26.50, so Rs 20), exchange 0.00035 percent plus IPFT
0.00005 percent, SEBI Rs 10 a crore, stamp duty 0.0001 percent on the buy, no STT, GST 18
percent of brokerage, exchange charge and SEBI fee.
"""

import datetime as dt
import warnings

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.reference import ReferenceDataWarning

SYMBOL = "USDINR29DEC26FUT"
DAY = dt.date(2026, 11, 20)
CAPITAL = 1_000_000
ENTRY = 88.33
MINUTES = 480  # 09:00 to 17:00
# The day's CDS contract note for a buy at 88.3300 and a sell at 88.3525 (turnover
# 1,76,682.50): brokerage 40.00; exchange 0.0004 percent = 0.70673, 0.71; SEBI 0.17668,
# 0.18; stamp 0.08833, Rs 0; GST 18 percent of 40 + 0.70673 + 0.17668 = 7.35901, 7.36.
NOTE = {"brokerage": 40.0, "stt": 0.0, "exchange_txn": 0.71, "sebi_fee": 0.18}
NOTE |= {"stamp_duty": 0.0, "gst": 7.36, "charged_total": 48.25}


def bars(exit_price: float) -> pd.DataFrame:
    start = pd.Timestamp.combine(DAY, dt.time(9, 0)).tz_localize("Asia/Kolkata")
    index = pd.date_range(start, periods=MINUTES, freq="1min")
    prices = [ENTRY] * 3 + [exit_price] * (MINUTES - 3)
    frame = pd.DataFrame({name: prices for name in ("open", "high", "low", "close")}, index=index)
    frame["volume"] = 100.0
    return frame


class UsdInrTrip(bt.Strategy):
    """Buys one lot at the first close and exits at the third."""

    symbol, exchange, interval, product = SYMBOL, "CDS", "1m", "NRML"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=1)
        elif len(self.close) == 3:
            self.exit()


class UsdInrTripInUnits(UsdInrTrip):
    """The same trip with its quantity in dollars: qty=1000, OpenAlgo's one lot."""

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=1_000)
        elif len(self.close) == 3:
            self.exit()


def run(exit_price: float, strategy: type[bt.Strategy] = UsdInrTrip) -> bt.Result:
    with warnings.catch_warnings():
        # The currency contract terms are unverified until NSE's specification is read.
        warnings.simplefilter("ignore", ReferenceDataWarning)
        return bt.backtest(strategy, data=bars(exit_price), capital=CAPITAL, slippage=0)


@pytest.fixture(scope="module")
def result() -> bt.Result:
    return run(88.3525)


def test_one_lot_is_a_thousand_dollars_bought_and_sold_at_the_next_bar_opens(result):
    fills = result.fills
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["quantity_units"]) == [1_000.0, 1_000.0]
    assert list(fills["price_rupees"]) == [88.33, 88.3525]


def test_the_fills_are_charged_as_nse_currency_futures(result):
    costs = result.costs
    assert list(costs["segment"]) == ["CDS_FUT", "CDS_FUT"]
    assert list(costs["schedule"]) == ["IN-2026-04-01", "IN-2026-04-01"]
    assert costs["verified"].all()


def test_the_days_cds_contract_note_charges_the_schedule_rows(result):
    note = result.contract_notes.iloc[0]
    assert len(result.contract_notes) == 1
    assert note["note_group"] == "CDS"
    for name, expected in NOTE.items():
        assert note[name] == pytest.approx(expected, abs=1e-9), name


def test_the_pnl_of_a_lot_is_the_price_move_times_a_thousand(result):
    # (88.3525 - 88.3300) x 1,000 = 22.50 gross; less the note's 48.25.
    trade = result.trades.iloc[0]
    assert trade["gross_pnl_rupees"] == pytest.approx(22.5, abs=1e-9)
    assert trade["costs_rupees"] == pytest.approx(48.25, abs=1e-9)
    assert trade["net_pnl_rupees"] == pytest.approx(22.5 - 48.25, abs=1e-9)


def test_one_rupee_of_price_move_is_rs_1000_a_lot():
    trade = run(ENTRY + 1.0).trades.iloc[0]
    assert trade["gross_pnl_rupees"] == pytest.approx(1_000.0, abs=1e-9)


def test_a_quantity_of_a_thousand_is_one_lot():
    # qty=1000 buys USD 1,000, one lot: the fills, the gross of 22.50 and the day's note of
    # 48.25 are those of lots=1.
    in_units = run(88.3525, UsdInrTripInUnits)
    assert list(in_units.fills["quantity_units"]) == [1_000.0, 1_000.0]
    trade = in_units.trades.iloc[0]
    assert trade["gross_pnl_rupees"] == pytest.approx(22.5, abs=1e-9)
    assert trade["costs_rupees"] == pytest.approx(48.25, abs=1e-9)
