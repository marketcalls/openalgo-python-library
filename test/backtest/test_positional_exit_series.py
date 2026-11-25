"""A futures series on an MCX delivery contract rolls before its positional exit (unit N4,
item N4-2, review round 2): with `roll=N` the hand-over is counted back N sessions from the
contract's positional exit day, the trading day before its tender period, not from its last
trading day, so the position rolls to the next contract instead of being exited flat by
the expiry exit while the roll is still to come.

Gold Mini, February and March 2026 contracts (last trading days Thu 5 February and Thu 5
March), lot 100, tick Re 1, multiplier 0.1. mcx_contracts.csv gives the February contract a
tender period of 3 trading days, the expiry day included (MCX/TRD/456/2025): Tue 3, Wed 4
and Thu 5 February, so its exit day is Mon 2 February. The MCX sessions before it are Sun 1
February (the Union Budget session of the calendar), Fri 30 and Thu 29 January; Mon 26
January is closed (Republic Day).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest import _futures
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.reference import Instrument

FEBRUARY = "GOLDM05FEB26FUT"
MARCH = "GOLDM05MAR26FUT"


def gold_mini(symbol: str, expiry: dt.date) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="MCX",
        kind="future",
        lot_size=100,
        tick_size=1.0,
        multiplier=0.1,
        expiry=expiry,
        underlying="GOLDM",
        settlement="physical",
    )


CONTRACTS = {
    FEBRUARY: gold_mini(FEBRUARY, dt.date(2026, 2, 5)),
    MARCH: gold_mini(MARCH, dt.date(2026, 3, 5)),
}
# The MCX sessions of the run, Tue 27 January to Tue 10 February 2026.
SESSIONS = [dt.date(2026, 1, day) for day in (27, 28, 29, 30)]
SESSIONS += [dt.date(2026, 2, day) for day in (1, 2, 3, 4, 5, 6, 9, 10)]


class GoldMiniSeries(bt.Strategy):
    """Buys one lot of the current contract at the first close and holds it."""

    underlying, exchange, interval = "GOLDM", "MCX", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(lots=1)


class _Plan:
    exchange = "MCX"
    start = SESSIONS[0]
    end = SESSIONS[-1]


def test_the_sessions_of_the_run_are_those_of_the_calendar():
    calendar = default_calendar()
    days = [SESSIONS[0] + dt.timedelta(days=n) for n in range(15)]
    assert [day for day in days if calendar.is_open("MCX", day)] == SESSIONS


@pytest.mark.parametrize(
    ("roll", "hand_over"),
    [
        # On the exit day itself, at its open, before the square-off.
        (0, dt.date(2026, 2, 2)),
        # The session before the exit day.
        (1, dt.date(2026, 2, 1)),
        # Three sessions before it: 1 February, 30 January, 29 January.
        (3, dt.date(2026, 1, 29)),
    ],
    ids=["roll 0", "roll 1", "roll 3"],
)
def test_the_roll_is_counted_back_from_the_exit_day_before_the_tender_period(roll, hand_over):
    args = {"instrument": CONTRACTS, "roll": roll}
    plan = _futures.resolve_futures(GoldMiniSeries, _Plan, args, default_calendar())
    assert plan.series[0].hand_over == (hand_over, None)


def test_held_to_expiry_a_delivery_series_warns_it_is_exited_not_settled():
    # roll=None: the February contract is exited at its exit day's square-off, not settled
    # in cash at its last close, and the warning says so.
    args = {"instrument": CONTRACTS, "roll": None}
    plan = _futures.resolve_futures(GoldMiniSeries, _Plan, args, default_calendar())
    (warning,) = plan.warnings
    assert warning.startswith(f"{FEBRUARY}, {MARCH}: the exchange settles these futures by")
    assert "exit reason expiry_exit, instead of being settled" in warning


def daily_bars(days: list[dt.date], first_price: float) -> pd.DataFrame:
    """Daily bars rising 10.00 a session from `first_price`, each opening at its close."""
    prices = [first_price + 10.0 * n for n in range(len(days))]
    frame = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in days]).tz_localize("Asia/Kolkata")
    return frame


def series_run(roll: int) -> bt.Result:
    data = {
        FEBRUARY: daily_bars(SESSIONS[:9], 78_000.0),
        MARCH: daily_bars(SESSIONS, 78_500.0),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bt.backtest(
            GoldMiniSeries,
            data=data,
            capital=10_000_000,
            instrument=CONTRACTS,
            product="NRML",
            start=SESSIONS[0],
            end=SESSIONS[-1],
            slippage=0,
            costs=None,
            roll=roll,
        )


@pytest.mark.parametrize(
    ("roll", "february_sale", "march_buy"),
    [
        # Rolled at the Mon 2 February open (the sixth session), before its square-off.
        (0, 78_050.0, 78_550.0),
        # Rolled at the Sun 1 February open (the fifth session).
        (1, 78_040.0, 78_540.0),
    ],
    ids=["roll 0", "roll 1"],
)
def test_a_gold_mini_series_rolls_to_march_instead_of_being_exited_flat(
    roll, february_sale, march_buy
):
    # Bought at the 27 January close, filled at the 28 January open, 78,010.00; rolled at
    # the open of its hand-over session, and March held to the end of the run.
    result = series_run(roll)
    fills = result.fills[["symbol", "side", "price_rupees", "reason"]].values.tolist()
    assert fills == [
        [FEBRUARY, "BUY", 78_010.0, "entry"],
        [FEBRUARY, "SELL", february_sale, "roll"],
        [MARCH, "BUY", march_buy, "roll"],
    ]
    held = result.final_positions.set_index("symbol")["quantity_units"]
    assert held.get(MARCH) == 100.0
