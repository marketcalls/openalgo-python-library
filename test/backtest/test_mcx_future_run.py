"""An MCX futures trade end to end with `instrument=` given: the fills, every charge line,
the contract notes, the margin blocked and the daily mark to market.

One CRUDEOIL contract (expiry 2026-11-18, lot 100 barrels, tick Re 1, multiplier 1, so
one lot is 100 units and a one-rupee move is Rs 100 a lot), daily bars on the MCX
sessions of 2026-11-02 to 2026-11-05, NRML, the default Zerodha costs in contract-note
rounding, no slippage, capital 10,00,000. Bars (open, high, low, close):

    11-02  5000  5050  4980  5020   buy one lot at the close
    11-03  5030  5080  5010  5060   filled at the open, 5,030.00
    11-04  5060  5100  5040  5090   exit at the close
    11-05  5100  5120  5070  5110   filled at the open, 5,100.00

Every expected charge is worked by hand from the MCX futures (non-agri) rates that the
cost schedule IN-2026-04-01 cites (the Zerodha charges page and calculator, snapshot
2026-11-12; reference note indian-trading-costs.md section 3): brokerage 0.03 percent
capped at Rs 20 an order, CTT 0.01 percent on sells, exchange transaction charge 0.0021
percent, SEBI fee Rs 10 a crore (0.0001 percent), stamp duty 0.002 percent on buys, GST
18 percent of brokerage, exchange charge and SEBI fee. The margin is the dated table's
CRUDEOIL row: 31.25 percent of the contract value.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.reference import Instrument

SYMBOL = "CRUDEOIL18NOV26FUT"
CRUDEOIL = Instrument(
    symbol=SYMBOL,
    exchange="MCX",
    kind="future",
    lot_size=100,
    tick_size=1.0,
    expiry=dt.date(2026, 11, 18),
    underlying="CRUDEOIL",
    settlement="cash",
)
CAPITAL = 1_000_000
DAYS = [dt.date(2026, 11, day) for day in (2, 3, 4, 5)]
BARS = [
    (5000.0, 5050.0, 4980.0, 5020.0),
    (5030.0, 5080.0, 5010.0, 5060.0),
    (5060.0, 5100.0, 5040.0, 5090.0),
    (5100.0, 5120.0, 5070.0, 5110.0),
]
# The MCX session runs 09:00 to 23:55 IST in November (US winter time).
SESSION_OPEN = dt.time(9, 0)
SESSION_CLOSE = dt.time(23, 55)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# Buy of 100 at 5,030.00, value 5,03,000.00.
BUY_LINES = {
    "brokerage": 20.0,  # 0.03% is 150.90, capped at 20.00
    "stt": 0.0,  # CTT is charged on sells only
    "exchange_txn": 10.563,  # 0.0021%
    "sebi_fee": 0.503,  # 0.0001%
    "stamp_duty": 10.06,  # 0.002% on the buy
    "gst": 5.59188,  # 18% of 20.00 + 10.563 + 0.503 = 31.066
    "total": 46.71788,
}
# Sell of 100 at 5,100.00, value 5,10,000.00.
SELL_LINES = {
    "brokerage": 20.0,  # 0.03% is 153.00, capped at 20.00
    "stt": 51.0,  # CTT 0.01% on the sell
    "exchange_txn": 10.71,
    "sebi_fee": 0.51,
    "stamp_duty": 0.0,
    "gst": 5.6196,  # 18% of 20.00 + 10.71 + 0.51 = 31.22
    "total": 87.8396,
}
# One MCX contract note a day: CTT and stamp duty to the rupee, the rest to the paisa.
BUY_NOTE = {"brokerage": 20.0, "stt": 0.0, "exchange_txn": 10.56, "sebi_fee": 0.5}
BUY_NOTE |= {"stamp_duty": 10.0, "gst": 5.59, "charged_total": 46.65}
SELL_NOTE = {"brokerage": 20.0, "stt": 51.0, "exchange_txn": 10.71, "sebi_fee": 0.51}
SELL_NOTE |= {"stamp_duty": 0.0, "gst": 5.62, "charged_total": 87.84}
LEDGER_COSTS = 46.65 + 87.84
GROSS_PNL = (5100.0 - 5030.0) * 100


def bars() -> pd.DataFrame:
    frame = pd.DataFrame(BARS, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in DAYS]).tz_localize("Asia/Kolkata")
    return frame


class CrudeSwing(bt.Strategy):
    """Buys one lot at the first close and exits at the third."""

    symbol, exchange, interval = SYMBOL, "MCX", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=1)
        elif len(self.close) == 3:
            self.exit()


@pytest.fixture(scope="module")
def result() -> bt.Result:
    return bt.backtest(
        CrudeSwing,
        data=bars(),
        capital=CAPITAL,
        instrument=CRUDEOIL,
        product="NRML",
        slippage=0,
    )


def at(day: dt.date, clock: dt.time) -> dt.datetime:
    return dt.datetime.combine(day, clock, tzinfo=IST)


def test_one_lot_is_bought_and_sold_at_the_next_session_opens(result):
    fills = result.fills
    assert list(fills["time"]) == [at(DAYS[1], SESSION_OPEN), at(DAYS[3], SESSION_OPEN)]
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["quantity_units"]) == [100.0, 100.0]
    assert list(fills["price_rupees"]) == [5030.0, 5100.0]


def test_the_fills_are_charged_as_mcx_futures_on_a_verified_schedule(result):
    costs = result.costs
    assert list(costs["segment"]) == ["MCX_FUT", "MCX_FUT"]
    assert list(costs["schedule"]) == ["IN-2026-04-01", "IN-2026-04-01"]
    assert costs["verified"].all()


@pytest.mark.parametrize(("row", "lines"), [(0, BUY_LINES), (1, SELL_LINES)], ids=["buy", "sell"])
def test_every_charge_line_of_a_fill_is_the_published_rate_on_its_value(result, row, lines):
    charged = result.costs.iloc[row]
    for name, expected in lines.items():
        assert charged[name] == pytest.approx(expected, abs=1e-9), name


@pytest.mark.parametrize(("row", "note"), [(0, BUY_NOTE), (1, SELL_NOTE)], ids=["11-03", "11-05"])
def test_each_days_contract_note_rounds_its_lines(result, row, note):
    charged = result.contract_notes.iloc[row]
    assert charged["note_group"] == "MCX"
    for name, expected in note.items():
        assert charged[name] == pytest.approx(expected, abs=1e-9), name


def test_the_ledger_is_charged_the_contract_notes_and_the_trade_its_fills(result):
    assert result.equity["costs_rupees"].iloc[-1] == pytest.approx(LEDGER_COSTS, abs=1e-9)
    trade = result.trades.iloc[0]
    fills_total = BUY_LINES["total"] + SELL_LINES["total"]
    assert trade["costs_rupees"] == pytest.approx(fills_total, abs=1e-9)
    assert trade["gross_pnl_rupees"] == GROSS_PNL
    assert trade["net_pnl_rupees"] == pytest.approx(GROSS_PNL - fills_total, abs=1e-9)


def test_the_margin_is_the_crudeoil_rows_percent_of_the_contract_value_while_held(result):
    # 31.25% of 5,030.00 x 100 = 1,57,187.50, blocked from the entry to the exit.
    margin = result.equity["margin_rupees"]
    assert list(margin) == [0.0, 157_187.5, 157_187.5, 0.0]
    assert any("FUT_CRUDEOIL/1" in text and "unverified" in text for text in result.warnings)


def test_the_daily_mark_to_market_posts_each_sessions_move_and_adds_up_to_the_pnl(result):
    # 11-03: (5,060 - 5,030) x 100; 11-04: (5,090 - 5,060) x 100; 11-05: the exit at
    # 5,100 against the last settlement at 5,090, (5,100 - 5,090) x 100.
    settlements = result.settlements
    assert list(settlements["date"]) == DAYS[1:]
    assert list(settlements["amount_rupees"]) == [3000.0, 3000.0, 1000.0]
    assert settlements["amount_rupees"].sum() == GROSS_PNL


@pytest.mark.parametrize(
    ("day", "equity"),
    [
        (DAYS[0], 1_000_000.0),
        # Marked at 5,060: + 3,000 open P&L, less the buy note of 46.65.
        (DAYS[1], 1_002_953.35),
        # Marked at 5,090: + 6,000 open P&L, less 46.65.
        (DAYS[2], 1_005_953.35),
        # Closed: + 7,000 realized, less both notes, 134.49.
        (DAYS[3], 1_006_865.51),
    ],
)
def test_equity_at_each_session_close_is_marked_to_the_close(result, day, equity):
    row = result.equity.loc[at(day, SESSION_CLOSE)]
    assert row["equity_rupees"] == pytest.approx(equity, abs=1e-6)
