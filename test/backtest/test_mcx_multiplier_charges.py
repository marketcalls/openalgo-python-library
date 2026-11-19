"""MCX futures whose multiplier is not 1 are charged on their traded value through
`bt.backtest` on bars passed with `data=`: every fill's charges, the contract notes, the
trades table and the Charges table of the tearsheet.

Each run buys one lot at the open of the second session and sells it at the open of the
fourth (daily bars on the MCX sessions of 2026-01-05 to 2026-01-08), NRML, the default
Zerodha costs in contract-note rounding, no slippage. The contracts, with the terms of
`mcx_contracts.csv` (lot in units, multiplier: the rupees of a one-rupee move per unit):

    GOLDM05FEB26FUT      100 units, 0.1    100 grams quoted per 10 grams; from the tables
    GOLD05FEB26FUT         1 unit,  100    1 kg quoted per 10 grams; given as instrument=
    ZINC29MAY26FUT         5 units, 1000   5 tonnes quoted per kg; from the tables
    CRUDEOILM19JAN26FUT   10 units, 1      10 barrels quoted per barrel; instrument=

The traded value is price x units x multiplier. Every expected charge is worked by hand
from the MCX futures (non-agri) rates of the cost schedule IN-2024-10-01, in force from
2024-10-01 to 2026-03-31: brokerage 0.03 percent capped at Rs 20 an order, CTT 0.01
percent on sells, exchange transaction charge 0.0021 percent (MCX/F&A/631/2024), SEBI fee
Rs 10 a crore (0.0001 percent), stamp duty 0.002 percent on buys, GST 18 percent of
brokerage, exchange charge and SEBI fee. Each fill is alone on its day's MCX contract
note, which rounds CTT and stamp duty to the rupee and the other lines to the paisa.

The fixed-risk sizer takes the estimated costs of the round trip out of its budget, so
it sizes a GOLDM entry from the same traded value (the last test).
"""

import dataclasses
import datetime as dt
import re

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.reference import Instrument

CAPITAL = 5_000_000
DAYS = [dt.date(2026, 1, day) for day in (5, 6, 7, 8)]
LINES = ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst")
#: The "Total charges" row of the tearsheet's Charges table, its rupees captured.
TOTAL_CHARGES_ROW = re.compile(r"Total charges</td><td class='r nums'>&#8377;([0-9,.]+)<")


@dataclasses.dataclass(frozen=True)
class Case:
    """One contract's run: its symbol, the instrument when given, the buy and sell
    prices (rupees per quoted unit), each fill's charge lines before rounding and its
    contract note's total, and the trade's turnover and gross P&L."""

    symbol: str
    instrument: Instrument | None
    buy: float
    sell: float
    buy_lines: tuple[float, ...]
    sell_lines: tuple[float, ...]
    buy_note: float
    sell_note: float
    turnover: float
    gross_pnl: float

    @property
    def charges(self) -> float:
        return round(self.buy_note + self.sell_note, 2)


def future(symbol: str, lot_size: int, multiplier: float, expiry: dt.date, **terms) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="MCX",
        kind="future",
        lot_size=lot_size,
        tick_size=1.0,
        multiplier=multiplier,
        expiry=expiry,
        **terms,
    )


# The lines in LINES order.
CASES = {
    # Buy 1,20,000.00 x 100 x 0.1 = 12,00,000.00: brokerage 360.00 capped at 20.00,
    # exchange 25.20, SEBI 1.20, stamp 24.00, GST 18% of 46.40 = 8.352 (note 8.35);
    # note 78.75. Sell 1,21,000.00 x 10 = 12,10,000.00: CTT 121.00, exchange 25.41, SEBI
    # 1.21, GST 18% of 46.62 = 8.3916 (note 8.39); note 176.01.
    "GOLDM": Case(
        "GOLDM05FEB26FUT",
        None,
        120_000.0,
        121_000.0,
        (20.0, 0.0, 25.20, 1.20, 24.0, 8.352),
        (20.0, 121.0, 25.41, 1.21, 0.0, 8.3916),
        78.75,
        176.01,
        2_410_000.0,
        10_000.0,
    ),
    # Buy 1,20,000.00 x 1 x 100 = 1,20,00,000.00: exchange 252.00, SEBI 12.00, stamp
    # 240.00, GST 18% of 284.00 = 51.12; note 575.12. Sell 1,21,00,000.00: CTT 1,210.00,
    # exchange 254.10, SEBI 12.10, GST 18% of 286.20 = 51.516 (note 51.52); note 1,547.72.
    "GOLD": Case(
        "GOLD05FEB26FUT",
        future("GOLD05FEB26FUT", 1, 100.0, dt.date(2026, 2, 5), underlying="GOLD"),
        120_000.0,
        121_000.0,
        (20.0, 0.0, 252.0, 12.0, 240.0, 51.12),
        (20.0, 1_210.0, 254.10, 12.10, 0.0, 51.516),
        575.12,
        1_547.72,
        24_100_000.0,
        100_000.0,
    ),
    # Buy 280.00 x 5 x 1,000 = 14,00,000.00: exchange 29.40, SEBI 1.40, stamp 28.00, GST
    # 18% of 50.80 = 9.144 (note 9.14); note 87.94. Sell 282.50 x 5,000 = 14,12,500.00:
    # CTT 141.25 (note 141), exchange 29.6625 (29.66), SEBI 1.4125 (1.41), GST 18% of
    # 51.075 = 9.1935 (9.19); note 201.26.
    "ZINC": Case(
        "ZINC29MAY26FUT",
        None,
        280.0,
        282.5,
        (20.0, 0.0, 29.40, 1.40, 28.0, 9.144),
        (20.0, 141.25, 29.6625, 1.4125, 0.0, 9.1935),
        87.94,
        201.26,
        2_812_500.0,
        12_500.0,
    ),
    # Multiplier 1, charged as before. Buy 5,400.00 x 10 = 54,000.00: brokerage 0.03% =
    # 16.20, exchange 1.134 (1.13), SEBI 0.054 (0.05), stamp 1.08 (1), GST 18% of 17.388 =
    # 3.12984 (3.13); note 21.51. Sell 54,500.00: brokerage 16.35, CTT 5.45 (5), exchange
    # 1.1445 (1.14), SEBI 0.0545 (0.05), GST 18% of 17.549 = 3.15882 (3.16); note 25.70.
    "CRUDEOILM": Case(
        "CRUDEOILM19JAN26FUT",
        future("CRUDEOILM19JAN26FUT", 10, 1.0, dt.date(2026, 1, 19), underlying="CRUDEOILM"),
        5_400.0,
        5_450.0,
        (16.20, 0.0, 1.134, 0.054, 1.08, 3.12984),
        (16.35, 5.45, 1.1445, 0.0545, 0.0, 3.15882),
        21.51,
        25.70,
        108_500.0,
        500.0,
    ),
}


def bars(case: Case) -> pd.DataFrame:
    """Day 1 closes at the buy price, day 2 opens there; day 4 opens at the sell price."""
    buy, sell = case.buy, case.sell
    rows = [(buy, buy), (buy, sell), (sell, sell), (sell, sell)]
    frame = pd.DataFrame(
        [(o, max(o, c), min(o, c), c) for o, c in rows], columns=["open", "high", "low", "close"]
    )
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in DAYS]).tz_localize("Asia/Kolkata")
    return frame


class OneLot(bt.Strategy):
    """Buys one lot at the first close and exits at the third."""

    exchange, interval = "MCX", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=1)
        elif len(self.close) == 3:
            self.exit()


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def run(request) -> tuple[Case, bt.Result]:
    case = CASES[request.param]
    strategy = type(f"OneLot{request.param}", (OneLot,), {"symbol": case.symbol})
    result = bt.backtest(
        strategy,
        data=bars(case),
        capital=CAPITAL,
        instrument=case.instrument,
        product="NRML",
        slippage=0,
    )
    return case, result


def test_one_lot_is_bought_and_sold_at_the_given_prices(run):
    case, result = run
    fills = result.fills
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["price_rupees"]) == [case.buy, case.sell]


def test_every_charge_line_of_a_fill_is_the_rate_on_its_traded_value(run):
    case, result = run
    costs = result.costs
    assert list(costs["segment"]) == ["MCX_FUT", "MCX_FUT"]
    for row, lines in enumerate((case.buy_lines, case.sell_lines)):
        charged = costs.iloc[row]
        for name, expected in zip(LINES, lines, strict=True):
            assert charged[name] == pytest.approx(expected, abs=1e-9), (row, name)
        assert charged["total"] == pytest.approx(sum(lines), abs=1e-9), row


def test_each_days_contract_note_rounds_the_lines_of_its_fill(run):
    case, result = run
    notes = result.contract_notes
    assert list(notes["note_group"]) == ["MCX", "MCX"]
    assert list(notes["charged_total"]) == pytest.approx([case.buy_note, case.sell_note], abs=1e-9)
    assert list(result.costs["charged"]) == pytest.approx([case.buy_note, case.sell_note], abs=1e-9)


def test_the_trade_and_the_ledger_pay_the_contract_notes(run):
    case, result = run
    trade = result.trades.iloc[0]
    assert trade["gross_pnl_rupees"] == pytest.approx(case.gross_pnl, abs=1e-9)
    assert trade["costs_rupees"] == pytest.approx(case.charges, abs=1e-9)
    assert trade["net_pnl_rupees"] == pytest.approx(case.gross_pnl - case.charges, abs=1e-9)
    assert result.equity["costs_rupees"].iloc[-1] == pytest.approx(case.charges, abs=1e-9)


def test_the_reports_turnover_and_charges_are_the_trades(run):
    case, result = run
    performance = result.performance
    assert performance.metrics["turnover"] == pytest.approx(case.turnover)
    assert performance.metrics["total_costs"] == pytest.approx(case.charges, abs=1e-9)
    assert sum(performance.cost_breakdown.values()) == pytest.approx(case.charges, abs=1e-9)


def test_the_tearsheets_charges_table_shows_the_charges(run):
    case, result = run
    shown = TOTAL_CHARGES_ROW.search(result.tearsheet())
    assert shown is not None
    assert float(shown.group(1).replace(",", "")) == case.charges


class RiskedGoldm(bt.Strategy):
    """Buys GOLDM at the first close, sized to risk 1 percent of the capital to a stop
    2 percent away."""

    symbol, exchange, interval = "GOLDM05FEB26FUT", "MCX", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(size=bt.size.fixed_risk(1), stop=bt.pct(2))


def test_fixed_risk_sizes_goldm_net_of_the_round_trip_on_its_traded_value():
    # The budget is 1 percent of 50,00,000.00, 50,000.00, to a stop 2 percent below the
    # decision close of 1,20,000.00: 2,400.00 a quoted unit, 240.00 a unit (multiplier
    # 0.1). Before costs that is 208.33 units, 200 in whole lots of 100. Their round trip
    # at 1,20,000.00 is worth 24,00,000.00 a side: the buy 20.00 + exchange 50.40 + SEBI
    # 2.40 + stamp 48.00 + GST 18% of 72.80 = 13.104, 133.904; the sale 20.00 + CTT 240.00
    # + 50.40 + 2.40 + 13.104, 325.904; 459.808 in all. (50,000.00 - 459.808) / 240.00 is
    # 206.42 units, 200 in whole lots. On the units as the value (2,40,00,000.00 a side)
    # the round trip would be 1,126.64 + 3,046.64 = 4,173.28, leaving 190.94 units: 100.
    result = bt.backtest(
        RiskedGoldm,
        data=bars(CASES["GOLDM"]),
        capital=CAPITAL,
        product="NRML",
        slippage=0,
    )
    assert result.fills.loc[0, "quantity_units"] == 200.0
