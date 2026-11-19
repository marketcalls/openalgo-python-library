"""A live GOLDM day on the fake OpenAlgo is charged on its traded value, units x
multiplier x price (`bt.live`'s day summary and its strategy's `bt.Result`; the kernel's
charge book through the live venue): a round trip of one lot filled whole, and one of two
lots whose buy the broker fills in two parts.

The market is `session_samples`' day, Monday 2026-11-16, with the session started at
14:40:05 IST, but the instrument is GOLDM05DEC26FUT on MCX (whose session runs 09:00 to
23:55): 100 units a lot, 100 grams quoted per 10 grams, so a multiplier of 0.1
(`mcx_contracts.csv`), tick Re 1. The broker serves flat candles at 1,20,000.00 over the
MCX sessions of the two days before and of today up to the start, and GOLDM trades 100
units every 10 seconds from then on. Each strategy buys at its first decision, at the
close of the 14:41 bar, and sells three bars later; the session is stopped at 14:50, flat.

By the MCX futures (non-agri) rates of the cost schedule IN-2026-04-01 (brokerage 0.03
percent capped at Rs 20 an order, CTT 0.01 percent on sells, exchange transaction charge
0.0021 percent, SEBI fee Rs 10 a crore, stamp duty 0.002 percent on buys, GST 18 percent
of brokerage, exchange charge and SEBI fee), worked by hand:

One lot filled whole. GOLDM trades at 1,20,000.00 all day, so both fills are at
1,20,000.00, each worth 1,20,000.00 x 100 x 0.1 = 12,00,000.00:

- the buy: brokerage 20.00, exchange 25.20, SEBI 1.20, stamp 24.00, GST 18 percent of 46.40
  = 8.352; 78.752, posted 78.75;
- the sale: brokerage 20.00, CTT 120.00, exchange 25.20, SEBI 1.20, GST 8.352; 174.752,
  posted 174.75;

so the day summary, each fill rounded to the paisa, charges 253.50. The day's one MCX
contract note has the lines brokerage 40.00, CTT 120, exchange 50.40, SEBI 2.40, stamp 24
and GST 16.704 rounded to 16.70: 253.50 too. Charged on its units as the value
(1,20,00,000.00 a fill) the day would cost 2,110.24.

Two lots, the buy filled in parts. GOLDM trades at 1,20,000.00 up to 14:41:55 and at
1,20,010.00 from 14:42:05, and the broker fills the first 100 units of the next order at
the price it arrives at and the other 100 at the next trade: the buy of 200 units fills
100 at 1,20,000.00 and 100 at 1,20,010.00, and the sale fills 200 at 1,20,010.00. The two
lots block 20 percent of 24,00,000.00, 4,80,000.00, as margin, so the broker holds
10,00,000.00 of cash and the session is given all of it.

- the buy is one order worth 12,00,000.00 + 12,00,100.00 = 24,00,100.00, costed once on
  its total: brokerage 20.00 (one order), exchange 50.4021, SEBI 2.4001, stamp 48.002, GST
  18 percent of 72.8022 = 13.104396; 133.908596, posted 133.91. Its first part is charged
  as the one-lot buy above, 78.752 posted 78.75, and the second the rest, 55.156596 posted
  55.16;
- the sale is worth 1,20,010.00 x 200 x 0.1 = 24,00,200.00: brokerage 20.00, CTT 240.02,
  exchange 50.4042, SEBI 2.4002, GST 18 percent of 72.8044 = 13.104792; 325.929192, posted
  325.93;

so the day summary charges 78.75 + 55.16 + 325.93 = 459.84. The day's contract note has
the lines brokerage 40.00, CTT 240.02 rounded to 240, exchange 100.8063 to 100.81, SEBI
4.8003 to 4.80, stamp 48.002 to 48 and GST 26.209188 to 26.21: 459.82, which the trade
pays. The summary's 459.84 is the fills' rounding; the note's adjustment is the -0.02
between them.
"""

from __future__ import annotations

import functools

import pytest
from fake_openalgo import LIVE_CASH
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from plan_samples import DAY, HISTORY_DAYS, MINUTE_NS
from session_samples import NS, TRADE_EVERY_S, Rig, ist, rig

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
#: The price GOLDM trades at from `RAISED_FROM` in the two-lot day, rupees.
RAISED_PRICE = 120_010.0
RAISED_FROM = ist(14, 42, 5)
#: The broker's cash and the session's capital in the two-lot day, rupees: the two lots
#: block 4,80,000.00 of margin (module docstring).
PARTS_CAPITAL = 1_000_000.0
#: The two-lot day's charges, rupees: the summary's, each fill rounded to the paisa, and
#: the contract note's (module docstring).
PARTS_DAY_CHARGES = 459.84
PARTS_NOTE_CHARGES = 459.82


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


def goldm_rig(make_fake_openalgo, monkeypatch, tmp_path, *, raised: bool) -> Rig:
    """The rig with GOLDM listed, its candles served and its trades scripted (module
    docstring), at 1,20,000.00 throughout or, when `raised`, at `RAISED_PRICE` from
    `RAISED_FROM`; SBIN does not trade."""
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
        price = RAISED_PRICE if raised and instant >= RAISED_FROM else PRICE
        trade = functools.partial(fake.trade, GOLDM, "MCX", price, volume=LOT_UNITS)
        made.script.at(instant, trade)
    made.stop_at(ist(14, 50))
    seam(made, monkeypatch)
    return made


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path):
    """The rig of the one-lot day."""
    return goldm_rig(make_fake_openalgo, monkeypatch, tmp_path, raised=False)


@pytest.fixture
def parts_world(make_fake_openalgo, monkeypatch, tmp_path):
    """The rig of the two-lot day: the broker holds `PARTS_CAPITAL` and fills the next
    order's first lot alone."""
    make = functools.partial(make_fake_openalgo, live_cash=PARTS_CAPITAL)
    made = goldm_rig(make, monkeypatch, tmp_path, raised=True)
    made.fake.split_next_fill(LOT_UNITS)
    return made


class OneGoldmLot(bt.Strategy):
    """Buys one lot of GOLDM at its first decision and sells it three bars later."""

    symbol, exchange, interval = GOLDM, "MCX", "1m"
    lot_count = 1

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(self.lot_count * LOT_UNITS)
        elif self.calls == 4:
            self.exit()


class TwoGoldmLots(OneGoldmLot):
    """Buys two lots of GOLDM at its first decision and sells them three bars later."""

    lot_count = 2


@pytest.fixture
def day(world) -> bt.SessionResult:
    """`OneGoldmLot`'s day through `bt.live`."""
    return live(world, OneGoldmLot, capital=CAPITAL)


@pytest.fixture
def parts_day(parts_world) -> bt.SessionResult:
    """`TwoGoldmLots`' day through `bt.live`."""
    return live(parts_world, TwoGoldmLots, capital=PARTS_CAPITAL)


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


def test_the_two_lot_buy_is_one_order_filled_in_two_parts(parts_day, parts_world):
    orders = parts_world.orders()
    assert [(order.action, order.quantity) for order in orders] == [
        ("BUY", 2 * LOT_UNITS),
        ("SELL", 2 * LOT_UNITS),
    ]
    fills = parts_day.fills
    assert list(zip(fills["action"], fills["units"], fills["price"], strict=True)) == [
        ("BUY", LOT_UNITS, PRICE),
        ("BUY", LOT_UNITS, RAISED_PRICE),
        ("SELL", 2 * LOT_UNITS, RAISED_PRICE),
    ]
    assert parts_day.summary.flat


def test_the_parts_of_an_order_are_charged_on_its_whole_traded_value(parts_day):
    result = parts_day.results["TwoGoldmLots"]
    charged = [78.752, 55.156596, 325.929192]
    assert list(result.costs["total"]) == pytest.approx(charged, abs=1e-9)


def test_the_two_lot_day_summary_charges_each_part_rounded_to_the_paisa(parts_day):
    assert parts_day.summary.charges == pytest.approx(PARTS_DAY_CHARGES, abs=1e-9)


def test_the_two_lot_trade_pays_the_days_contract_note(parts_day):
    result = parts_day.results["TwoGoldmLots"]
    notes = result.contract_notes
    assert list(notes["charged_total"]) == pytest.approx([PARTS_NOTE_CHARGES], abs=1e-9)
    assert result.trades["costs_rupees"].sum() == pytest.approx(PARTS_NOTE_CHARGES, abs=1e-9)
