"""A live GOLDM round trip on the fake OpenAlgo is charged on its traded value, units x
multiplier x price (`bt.live`'s day summary and its strategy's `bt.Result`; the kernel's
charge book through the live venue).

The market is `session_samples`' day, Monday 2026-11-16, with the session started at
14:40:05 IST, but the instrument is GOLDM05DEC26FUT on MCX (whose session runs 09:00 to
23:55): 100 units a lot, 100 grams quoted per 10 grams, so a multiplier of 0.1
(`mcx_contracts.csv`), tick Re 1. The broker serves flat candles at 1,20,000.00 over the
MCX sessions of the two days before and of today up to the start, and GOLDM trades 100
units every 10 seconds at 1,20,000.00 from then on. `OneGoldmLot` buys one lot at its
first decision and sells it three bars later; the session is stopped at 14:50, flat.

Both fills are at 1,20,000.00, each worth 1,20,000.00 x 100 x 0.1 = 12,00,000.00. By the
MCX futures (non-agri) rates of the cost schedule IN-2026-04-01 (brokerage 0.03 percent
capped at Rs 20 an order, CTT 0.01 percent on sells, exchange transaction charge 0.0021
percent, SEBI fee Rs 10 a crore, stamp duty 0.002 percent on buys, GST 18 percent of
brokerage, exchange charge and SEBI fee), worked by hand:

- the buy: brokerage 20.00, exchange 25.20, SEBI 1.20, stamp 24.00, GST 18 percent of 46.40
  = 8.352; 78.752, posted 78.75;
- the sale: brokerage 20.00, CTT 120.00, exchange 25.20, SEBI 1.20, GST 8.352; 174.752,
  posted 174.75;

so the day summary, each fill rounded to the paisa, charges 253.50. The day's one MCX
contract note has the lines brokerage 40.00, CTT 120, exchange 50.40, SEBI 2.40, stamp 24
and GST 16.704 rounded to 16.70: 253.50 too. Charged on its units as the value
(1,20,00,000.00 a fill) the day would cost 2,110.24.
"""

from __future__ import annotations

import pytest
from fake_openalgo import LIVE_CASH
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from plan_samples import DAY, HISTORY_DAYS, MINUTE_NS
from session_samples import NS, TRADE_EVERY_S, ist, rig

import openalgo.backtest as bt
from openalgo.backtest.calendar import default_calendar

GOLDM = "GOLDM05DEC26FUT"
LOT_UNITS = 100
PRICE = 120_000.0
#: A lot of GOLDM is worth 1,20,000.00 x 100 x 0.1 = 12,00,000.00 and blocks 20 percent
#: of that, 2,40,000.00, as margin: more than the samples' capital, so the session is
#: given the fake broker's whole cash, 2,50,000.00.
CAPITAL = LIVE_CASH
#: The day's charges, rupees (module docstring).
DAY_CHARGES = 253.50


def flat_candles(start: tuple[int, int]) -> list[tuple[int, float, float, float, float, float]]:
    """1-minute candles at `PRICE` over the MCX sessions of the two days before the live
    day and of the live day up to the IST (hour, minute) `start`."""
    calendar = default_calendar()
    rows = []
    for day in (*HISTORY_DAYS, DAY):
        opens, closes = calendar.session("MCX", day)
        end = ist(*start) if day == DAY else closes
        rows += [
            (minute, PRICE, PRICE, PRICE, PRICE, 1_000.0) for minute in range(opens, end, MINUTE_NS)
        ]
    return rows


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path):
    """The rig with GOLDM listed, its candles served and its trades scripted (module
    docstring); SBIN does not trade."""
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=None)
    fake = made.fake
    fake.add_instrument(
        GOLDM,
        "MCX",
        lot_size=LOT_UNITS,
        tick_size=1.0,
        freeze_qty=10_000,
        instrument_type="FUT",
        expiry="05-DEC-26",
    )
    fake.seed_candles(GOLDM, "MCX", flat_candles((14, 40)))
    fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS)
    for instant in range(ist(14, 40, 15), ist(15, 0), TRADE_EVERY_S * NS):
        made.script.at(instant, lambda: fake.trade(GOLDM, "MCX", PRICE, volume=LOT_UNITS))
    made.stop_at(ist(14, 50))
    seam(made, monkeypatch)
    return made


class OneGoldmLot(bt.Strategy):
    """Buys one lot of GOLDM at its first decision and sells it three bars later."""

    symbol, exchange, interval = GOLDM, "MCX", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(LOT_UNITS)
        elif self.calls == 4:
            self.exit()


@pytest.fixture
def day(world) -> bt.SessionResult:
    """`OneGoldmLot`'s day through `bt.live`."""
    return live(world, OneGoldmLot, capital=CAPITAL)


def test_one_lot_is_bought_and_sold_at_the_markets_price(day, world):
    orders = world.orders()
    assert [(order.action, order.quantity) for order in orders] == [
        ("BUY", LOT_UNITS),
        ("SELL", LOT_UNITS),
    ]
    assert list(day.fills["price"]) == [PRICE, PRICE]
    assert day.summary.flat


def test_the_day_summary_charges_each_fill_on_its_traded_value(day):
    assert day.summary.charges == pytest.approx(DAY_CHARGES, abs=1e-9)


def test_the_strategys_result_charges_the_fills_and_the_days_contract_note(day):
    result = day.results["OneGoldmLot"]
    assert list(result.costs["total"]) == pytest.approx([78.752, 174.752], abs=1e-9)
    assert list(result.contract_notes["charged_total"]) == pytest.approx([DAY_CHARGES], abs=1e-9)
    assert result.trades["costs_rupees"].sum() == pytest.approx(DAY_CHARGES, abs=1e-9)
