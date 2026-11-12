"""Calculator mode against independent statements of the calculator's arithmetic.

The oracles below restate how the Zerodha brokerage calculator computes a round trip
(its published script, https://zerodha.com/static/js/brokerage.js, as of
2026-11-12), with the rates of the reference note `indian-trading-costs.md`:

- the same IEEE double operations in the same order as the calculator;
- two-decimal rounding done on the exact binary value, half up (`Decimal`), which is
  what the calculator's number formatting does;
- tax and stamp duty rounded to the paisa and then to the rupee, half up (commodity
  CTT stays in paise).

They share no code with the engine.

Equity and equity derivatives: the engine takes the same inputs as the calculator
(prices and a quantity), so it must agree to the paisa on every trade.

MCX futures: the calculator takes lots and a contract multiplier and multiplies
price x multiplier x lots (and CTT as rate x price x lots x multiplier), while the
engine takes units (lots x multiplier). A product can then differ in its last binary
digit and round the other way, so the engine is required to be within one paisa of
the calculator, not equal to it. `mcx_calculator` keeps the calculator's own order
of operations so that this difference is visible to the test.

The four screenshot-verified trades of the reference note are in `test_costs.py`.
"""

import math
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import pytest

from openalgo.backtest import costs

SNAPSHOT = "2026-11-12"
TRADES_PER_SEGMENT = 250
PAISA = Decimal("0.01")
ONE_PAISA = 0.01 + 1e-9

# Zerodha brokerage: 0.03 percent of a leg's value, at most Rs 20 a leg; Rs 20 flat
# for an options leg.
BROKERAGE_RATE = 0.0003
RUPEES_PER_ORDER = 20.0
GST_RATE = 0.18


def to_paise(rupees: float) -> float:
    """Round to two decimals, half up, on the exact binary value."""
    return float(Decimal(rupees).quantize(PAISA, rounding=ROUND_HALF_UP))


def to_rupee(rupees: float) -> float:
    """Round to the rupee, half up."""
    return float(math.floor(rupees + 0.5))


@dataclass(frozen=True)
class CalculatorTab:
    """One equity or F&O tab of the calculator: its rates as fractions."""

    brokerage: str  # "none", "rate" or "flat"
    tax_rate: float  # STT
    exchange_rate: float  # transaction charge, without IPFT
    stamp_rate: float  # stamp duty on the buy value
    ipft_rate: float = 0.0
    sebi_rate: float = 0.000001
    tax_on_turnover: bool = False  # delivery: STT on buys and sells

    def leg_brokerage(self, value: float) -> float:
        if self.brokerage == "none" or value == 0:
            return 0.0
        if self.brokerage == "flat":
            return RUPEES_PER_ORDER
        uncapped = value * BROKERAGE_RATE
        return RUPEES_PER_ORDER if uncapped > RUPEES_PER_ORDER else to_paise(uncapped)

    def round_trip(self, buy: float, sell: float, quantity: float) -> dict[str, float]:
        buy_value, sell_value = buy * quantity, sell * quantity
        turnover = to_paise((buy + sell) * quantity)
        brokerage = to_paise(self.leg_brokerage(buy_value) + self.leg_brokerage(sell_value))
        taxed = turnover if self.tax_on_turnover else sell_value
        stt = to_rupee(to_paise(taxed * self.tax_rate))
        exchange = to_paise(
            to_paise(turnover * self.exchange_rate) + to_paise(turnover * self.ipft_rate)
        )
        sebi = to_paise(turnover * self.sebi_rate)
        gst = to_paise(GST_RATE * (brokerage + exchange + sebi))
        stamp = to_rupee(to_paise(buy_value * self.stamp_rate))
        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_txn": exchange,
            "sebi_fee": sebi,
            "stamp_duty": stamp,
            "gst": gst,
            "total": to_paise(brokerage + stt + exchange + gst + sebi + stamp),
        }


# Reference note section 1. segment: (tab, price range, quantity range in units)
TABS = {
    "NSE_EQ_INTRADAY": (
        CalculatorTab("rate", 0.00025, 0.0000297, 0.00003, ipft_rate=0.000001),
        (5, 4000),
        (1, 900),
    ),
    "NSE_EQ_DELIVERY": (
        CalculatorTab("none", 0.001, 0.0000297, 0.00015, ipft_rate=0.000001, tax_on_turnover=True),
        (5, 4000),
        (1, 900),
    ),
    "BSE_EQ_INTRADAY": (CalculatorTab("rate", 0.00025, 0.0000375, 0.00003), (5, 4000), (1, 900)),
    "BSE_EQ_DELIVERY": (
        CalculatorTab("none", 0.001, 0.0000375, 0.00015, tax_on_turnover=True),
        (5, 4000),
        (1, 900),
    ),
    "NFO_FUT": (
        CalculatorTab("rate", 0.0005, 0.0000173, 0.00002, ipft_rate=0.000001),
        (100, 26000),
        (15, 1500),
    ),
    "BFO_FUT": (CalculatorTab("rate", 0.0005, 0.0, 0.00002), (100, 85000), (10, 600)),
    "NFO_OPT": (
        CalculatorTab("flat", 0.0015, 0.0003503, 0.00003, ipft_rate=0.000005),
        (0.5, 600),
        (25, 3000),
    ),
    "BFO_OPT": (CalculatorTab("flat", 0.0015, 0.000325, 0.00003), (0.5, 900), (10, 1200)),
}


def random_trades(segment: str):
    """Seeded (buy, sell, quantity) trades with prices on a five-paise tick.

    Every tenth trade is buy only and every eleventh sell only, as the calculator
    allows a price to be left at zero.
    """
    _, (low, high), (fewest, most) = TABS[segment]
    rng = random.Random(f"costs-parity-{segment}")
    for index in range(TRADES_PER_SEGMENT):
        buy = round(round(rng.uniform(low, high) / 0.05) * 0.05, 2)
        sell = round(round(buy * rng.uniform(0.9, 1.1) / 0.05) * 0.05, 2)
        quantity = float(rng.randint(fewest, most))
        if index % 10 == 9:
            sell = 0.0
        elif index % 11 == 10:
            buy = 0.0
        yield buy, sell, quantity


def in_paise(amounts: dict[str, float]) -> dict[str, int]:
    return {name: round(amount * 100) for name, amount in amounts.items()}


@pytest.mark.parametrize("segment", TABS)
def test_calculator_mode_is_paisa_exact_for_equity_and_equity_derivatives(segment):
    tab = TABS[segment][0]
    model = costs.CostModel(rounding="calculator")
    for buy, sell, quantity in random_trades(segment):
        actual = model.round_trip(buy, sell, quantity, SNAPSHOT, segment).as_dict()
        expected = tab.round_trip(buy, sell, quantity)
        assert in_paise(actual) == in_paise(expected), (segment, buy, sell, quantity)


@pytest.mark.parametrize("segment", TABS)
def test_a_one_sided_calculator_trade_matches_a_round_trip_with_a_zero_price(segment):
    tab = TABS[segment][0]
    model = costs.CostModel(rounding="calculator")
    for buy, sell, quantity in list(random_trades(segment))[:40]:
        bought = model.trade("BUY", buy, quantity, SNAPSHOT, segment).as_dict()
        sold = model.trade("SELL", sell, quantity, SNAPSHOT, segment).as_dict()
        assert in_paise(bought) == in_paise(tab.round_trip(buy, 0.0, quantity)), (buy, quantity)
        assert in_paise(sold) == in_paise(tab.round_trip(0.0, sell, quantity)), (sell, quantity)


def test_the_equity_oracle_reproduces_the_screenshot_verified_example():
    # Reference note section 6, to check the oracle itself: totals of the four tabs.
    totals = {
        "NSE_EQ_INTRADAY": (1000, 1100, 200.62),
        "NSE_EQ_DELIVERY": (1000, 1100, 931.42),
        "NFO_FUT": (1000, 1100, 294.33),
        "NFO_OPT": (100, 110, 149.52),
    }
    for segment, (buy, sell, total) in totals.items():
        assert TABS[segment][0].round_trip(buy, sell, 400)["total"] == total, segment


# Reference note section 3: (segment, contract multiplier, agri, exchange rate, price range).
# The calculator charges no CTT on any agri contract, but CTT exempts only the commodities
# of rule 3 of the CTT Rules 2013, so MENTHAOIL and CPO (MCX_FUT_AGRI_CTT) pay CTT here and
# are not compared; COTTONCNDY (cotton is in rule 3) is.
MCX_CONTRACTS = {
    "CRUDEOIL": ("MCX_FUT", 100, False, 0.000021, (4000, 7000)),
    "CRUDEOILM": ("MCX_FUT", 10, False, 0.000021, (4000, 7000)),
    "NATURALGAS": ("MCX_FUT", 1250, False, 0.000021, (150, 400)),
    "GOLDM": ("MCX_FUT", 10, False, 0.000021, (60000, 130000)),
    "SILVERM": ("MCX_FUT", 5, False, 0.000021, (70000, 150000)),
    "COPPER": ("MCX_FUT", 2500, False, 0.000021, (700, 1100)),
    "ZINC": ("MCX_FUT", 5000, False, 0.000021, (200, 320)),
    "COTTONCNDY": ("MCX_FUT_AGRI", 48, True, 0.000021, (50000, 62000)),
}


def mcx_leg_brokerage(value: float) -> float:
    # The calculator charges a flat Rs 20 above Rs 2 lakh and otherwise the capped
    # 0.03 percent; both give Rs 20 from Rs 66,667 up.
    if value > 200000 or value * BROKERAGE_RATE > RUPEES_PER_ORDER:
        return RUPEES_PER_ORDER
    return to_paise(value * BROKERAGE_RATE)


def mcx_calculator(commodity: str, buy: float, sell: float, lots: float) -> dict[str, float]:
    """The commodity futures tab, with its own order of operations: lots and the
    contract multiplier are separate factors, as the calculator has them."""
    _, multiplier, agri, exchange_rate, _ = MCX_CONTRACTS[commodity]
    turnover = to_paise((buy + sell) * multiplier * lots)
    brokerage = mcx_leg_brokerage(buy * multiplier * lots) + mcx_leg_brokerage(
        sell * multiplier * lots
    )
    if agri:
        sebi, ctt = to_paise(turnover * 0.0000001), 0.0
    else:
        sebi, ctt = to_paise(turnover * 0.000001), to_paise(0.0001 * sell * lots * multiplier)
    exchange = to_paise(exchange_rate * turnover)
    gst = to_paise(GST_RATE * (brokerage + exchange + sebi))
    stamp = to_rupee(to_paise(buy * lots * multiplier * 0.00002))
    return {
        "brokerage": brokerage,
        "stt": ctt,
        "exchange_txn": exchange,
        "sebi_fee": sebi,
        "stamp_duty": stamp,
        "gst": gst,
        "total": to_paise(brokerage + ctt + exchange + gst + sebi + stamp),
    }


def mcx_trades(commodity: str):
    """Seeded (buy, sell, lots) trades on a five-paise tick, 1 to 20 lots."""
    low, high = MCX_CONTRACTS[commodity][4]
    rng = random.Random(f"costs-parity-mcx-{commodity}")
    for _ in range(TRADES_PER_SEGMENT):
        buy = round(round(rng.uniform(low, high) / 0.05) * 0.05, 2)
        sell = round(round(buy * rng.uniform(0.97, 1.03) / 0.05) * 0.05, 2)
        yield buy, sell, float(rng.randint(1, 20))


def test_the_mcx_oracle_reproduces_a_known_calculator_result():
    # CRUDEOILM, 5 lots, buy 5403 and sell 5485: the calculator shows CTT 27.42 and
    # total 93.74 (it forms CTT as 0.0001 x 5485 x 5 x 10).
    result = mcx_calculator("CRUDEOILM", 5403.0, 5485.0, 5.0)
    assert (result["stt"], result["total"]) == (27.42, 93.74)


@pytest.mark.parametrize("commodity", MCX_CONTRACTS)
def test_calculator_mode_is_within_one_paisa_of_the_calculator_for_mcx(commodity):
    segment, multiplier = MCX_CONTRACTS[commodity][:2]
    model = costs.CostModel(rounding="calculator")
    for buy, sell, lots in mcx_trades(commodity):
        actual = model.round_trip(buy, sell, lots * multiplier, SNAPSHOT, segment).as_dict()
        expected = mcx_calculator(commodity, buy, sell, lots)
        gaps = {name: abs(actual[name] - expected[name]) for name in expected}
        assert max(gaps.values()) <= ONE_PAISA, (commodity, buy, sell, lots, gaps)
