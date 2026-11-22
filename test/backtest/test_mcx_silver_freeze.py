"""A backtest of silver micro, SILVERMIC30NOV26FUT, on bars passed with `data=`: its
maximum order size comes from the built-in `mcx_contracts` table, so the run needs no
symbol-master snapshot and no user file.

MCX/TRD/275/2025 (2025-05-30), the specification of the Silver Micro contracts expiring
from February to November 2026: trading unit 1 kg, quoted per kg, tick Re 1, maximum
order size 600 kg. OpenAlgo counts MCX quantity in the trading unit's number, so a lot is
1 unit and the freeze quantity is 600 units. An order of exactly the freeze quantity
passes; one above it is rejected (`reject_code` "freeze_qty").

The bars are four MCX sessions, Monday 2026-11-16 to Thursday 2026-11-19, flat at Rs
1,00,000.00 a kg. 600 kg are worth 6,00,00,000.00 and block 20 percent of that,
1,20,00,000.00, as futures margin, inside the 2,00,00,000.00 of capital.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt

SYMBOL = "SILVERMIC30NOV26FUT"
DAYS = [dt.date(2026, 11, day) for day in (16, 17, 18, 19)]
PRICE = 100_000.0
CAPITAL = 20_000_000
#: MCX/TRD/275/2025: the maximum order size of Silver Micro, 600 kg, in units of 1 kg.
FREEZE_UNITS = 600


def flat_bars() -> pd.DataFrame:
    """Four daily bars at `PRICE`, IST session dates."""
    frame = pd.DataFrame(
        {"open": PRICE, "high": PRICE, "low": PRICE, "close": PRICE, "volume": 1_000.0},
        index=pd.DatetimeIndex([pd.Timestamp(day) for day in DAYS]).tz_localize("Asia/Kolkata"),
    )
    return frame


class SilverMicro(bt.Strategy):
    """Buys `units` at the first close and exits at the third."""

    symbol, exchange, interval = SYMBOL, "MCX", "D"
    units = FREEZE_UNITS

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=self.units)
        elif len(self.close) == 3:
            self.exit()


def run(units: int) -> bt.Result:
    strategy = type(f"SilverMicro{units}", (SilverMicro,), {"units": units})
    return bt.backtest(strategy, data=flat_bars(), capital=CAPITAL, product="NRML", slippage=0)


@pytest.fixture(scope="module")
def at_the_limit() -> bt.Result:
    return run(FREEZE_UNITS)


def test_the_instrument_takes_its_maximum_order_size_from_the_mcx_circular(at_the_limit):
    (instrument,) = at_the_limit.manifest["instruments"]
    assert instrument["freeze_qty"] == FREEZE_UNITS
    sources = {term: (source, status) for term, source, status in instrument["provenance"]}
    assert sources["freeze_qty"] == ("MCX/TRD/275/2025 (2025-05-30)", "verified")


def test_an_order_of_the_maximum_order_size_fills(at_the_limit):
    fills = at_the_limit.fills
    assert list(fills["side"]) == ["BUY", "SELL"]
    assert list(fills["quantity_units"]) == [FREEZE_UNITS, FREEZE_UNITS]


def test_an_order_above_the_maximum_order_size_is_rejected():
    result = run(FREEZE_UNITS + 1)
    assert list(result.orders["reject_code"]) == ["freeze_qty"]
    assert result.fills.empty
