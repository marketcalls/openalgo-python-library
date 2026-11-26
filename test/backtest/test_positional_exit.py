"""The positional exit day of a derivatives contract (`_positional_exit`; design
20-positional-live section 9, unit N4 item N4-2): worked out from the reference tables and
passed to the kernel, which exits every position still open on the contract at the
square-off time of that day, exit reason `expiry_exit`.

The exit days below are worked by hand from the exchange calendar: the last trading day of
an option and of a stock future; for an MCX delivery contract the trading day before the
last `tender_days` trading days of the contract (mcx_contracts.csv: Gold Mini from the
October 2025 expiry, 3 days, expiry day included, MCX/TRD/456/2025).
"""

from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._positional_exit import exit_day, positional_exits
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.reference import Instrument, ReferenceDataWarning

# Thu 2026-02-05, the last trading day of the February 2026 MCX gold contracts.
GOLD_EXPIRY = dt.date(2026, 2, 5)


def contract(symbol: str, exchange: str, kind: str, settlement: str, **terms) -> Instrument:
    defaults = {"lot_size": 1, "tick_size": 1.0, "expiry": GOLD_EXPIRY}
    return Instrument(
        symbol=symbol,
        exchange=exchange,
        kind=kind,
        settlement=settlement,
        **(defaults | terms),
    )


GOLDM = contract(
    "GOLDM05FEB26FUT", "MCX", "future", "physical", lot_size=100, multiplier=0.1, underlying="GOLDM"
)
GOLD = contract("GOLD05FEB26FUT", "MCX", "future", "physical", multiplier=100.0, underlying="GOLD")
CRUDE = contract(
    "CRUDEOIL19FEB26FUT", "MCX", "future", "cash", lot_size=100, expiry=dt.date(2026, 2, 19)
)
SBIN_FUTURE = contract(
    "SBIN26FEB26FUT",
    "NFO",
    "future",
    "physical",
    lot_size=750,
    tick_size=0.05,
    expiry=dt.date(2026, 2, 26),
    underlying="SBIN",
)
NIFTY_FUTURE = contract(
    "NIFTY26FEB26FUT",
    "NFO",
    "future",
    "cash",
    lot_size=65,
    tick_size=0.1,
    expiry=dt.date(2026, 2, 26),
    underlying="NIFTY",
)
NIFTY_CALL = contract(
    "NIFTY26FEB2626000CE",
    "NFO",
    "option",
    "cash",
    lot_size=65,
    tick_size=0.05,
    expiry=dt.date(2026, 2, 26),
    strike=26000.0,
    option_type="CE",
    underlying="NIFTY",
)


def test_the_calendar_days_the_mcx_figure_is_worked_from():
    calendar = default_calendar()
    open_days = [day for day in range(1, 6) if calendar.is_open("MCX", dt.date(2026, 2, day))]
    # Sun 1 February 2026 is a special session of the calendar (the Union Budget).
    assert open_days == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        # An option exits on its last trading day, whatever settles it.
        (NIFTY_CALL, dt.date(2026, 2, 26)),
        # A stock future, settled by delivery: on its last trading day.
        (SBIN_FUTURE, dt.date(2026, 2, 26)),
        # An index future, settled in cash: settles at expiry, no exit day.
        (NIFTY_FUTURE, None),
        # An MCX cash contract: settles at expiry.
        (CRUDE, None),
        # Gold Mini: the tender period is Tue 3, Wed 4 and Thu 5 February; the exit day is
        # the trading day before it, Mon 2 February.
        (GOLDM, dt.date(2026, 2, 2)),
    ],
    ids=["index option", "stock future", "index future", "mcx cash", "mcx gold mini"],
)
def test_each_contract_takes_its_positional_exit_day(instrument, expected):
    assert exit_day(instrument, default_calendar()) == (expected, None)


def test_an_mcx_delivery_contract_the_table_gives_no_tender_period_for_is_named_in_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        day, warning = exit_day(GOLD, default_calendar())
    assert day is None
    # No run takes reference tables of the user's, so the advice is the one a user can follow
    # (the lead's ruling 9a of review round 4 of unit N5a).
    assert warning == (
        "GOLD05FEB26FUT is settled by delivery, and the reference table mcx_contracts gives no "
        "tender period for it: a position held into its tender period is settled at expiry in "
        "cash, where MCX would mark it for delivery. Exit it before its tender period"
    )


def test_the_exit_days_and_warnings_of_a_run_are_in_registration_order():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReferenceDataWarning)
        exits = positional_exits([NIFTY_FUTURE, GOLD, GOLDM], default_calendar())
    assert exits.days == (None, None, dt.date(2026, 2, 2))
    assert len(exits.warnings) == 1 and exits.warnings[0].startswith("GOLD05FEB26FUT")


# A run end to end: daily bars, one lot bought at the first close, held.

IST = "Asia/Kolkata"


def daily_bars(days: list[dt.date], closes: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"open": closes, "high": [c + 5 for c in closes], "low": [c - 5 for c in closes]}
    )
    frame["close"] = closes
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in days]).tz_localize(IST)
    return frame


def holder(symbol: str, exchange: str) -> type[bt.Strategy]:
    """Buys one lot at the first close and holds it."""

    class Holder(bt.Strategy):
        interval = "D"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(lots=1)

    Holder.symbol, Holder.exchange = symbol, exchange
    return Holder


def held_run(instrument: Instrument, days: list[dt.date], closes: list[float]) -> bt.Result:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bt.backtest(
            holder(instrument.symbol, instrument.exchange),
            data=daily_bars(days, closes),
            capital=10_000_000,
            instrument=instrument,
            product="NRML",
            slippage=0,
            costs=None,
        )


def test_a_gold_mini_held_into_its_tender_period_exits_the_trading_day_before_it():
    # Bought at the 28 January close, filled at the 29 January open (78,000.00); daily bars
    # hold no bar at or after the 23:10 square-off, so it is sold at the close of Mon 2
    # February, 78,300.00, flagged.
    days = [dt.date(2026, 1, 28), dt.date(2026, 1, 29), dt.date(2026, 1, 30)]
    days += [dt.date(2026, 2, 2), dt.date(2026, 2, 3)]
    closes = [78_000.0, 78_000.0, 78_100.0, 78_300.0, 78_500.0]
    result = held_run(GOLDM, days, closes)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "expiry_exit"
    assert trade["exit_price_rupees"] == 78_300.0
    assert pd.Timestamp(trade["exit_ts"]).date() == dt.date(2026, 2, 2)


def test_a_gold_contract_with_no_tender_period_settles_at_expiry_and_warns():
    days = [dt.date(2026, 2, day) for day in (2, 3, 4, 5)]
    result = held_run(GOLD, days, [150_000.0, 150_100.0, 150_200.0, 150_300.0])
    assert result.trades.iloc[0]["exit_reason"] == "expiry"
    assert any(note.startswith("GOLD05FEB26FUT is settled by delivery") for note in result.warnings)


def test_a_gold_mini_bought_at_the_close_of_its_exit_day_is_refused():
    # The buy decided at the close of Mon 2 February would fill at the Tue 3 February open,
    # inside the tender period: it is refused, and nothing is held into it or settled.
    days = [dt.date(2026, 2, day) for day in (2, 3, 4, 5)]
    result = held_run(GOLDM, days, [78_000.0, 78_100.0, 78_200.0, 78_300.0])
    assert result.trades.empty
    assert list(result.orders["reject_code"]) == ["contract_expiring"]


GOLDM_OCTOBER = contract(
    "GOLDM03OCT25FUT",
    "MCX",
    "future",
    "physical",
    lot_size=100,
    multiplier=0.1,
    underlying="GOLDM",
    expiry=dt.date(2025, 10, 3),
)


def mcx_sessions(first: dt.date, last: dt.date) -> list[dt.date]:
    calendar = default_calendar()
    count = (last - first).days + 1
    days = [first + dt.timedelta(days=n) for n in range(count)]
    return [day for day in days if calendar.is_open("MCX", day)]


@pytest.mark.parametrize(
    "start",
    [dt.date(2025, 9, 1), dt.date(2025, 9, 25)],
    ids=["from 1 September", "from 25 September"],
)
def test_the_tender_period_is_read_for_the_contract_whatever_day_the_run_starts(start):
    # Gold Mini of the October 2025 expiry (Fri 3 October): the verified row of
    # MCX/TRD/456/2025 covers it from the 22 September trade date and gives 3 tender days,
    # Tue 30 September, Wed 1 and Fri 3 October (Thu 2 October is an MCX holiday), so its
    # exit day is Mon 29 September whether the run starts before that row's first trade
    # date or after it.
    assert mcx_sessions(dt.date(2025, 9, 29), dt.date(2025, 10, 3)) == [
        dt.date(2025, 9, 29),
        dt.date(2025, 9, 30),
        dt.date(2025, 10, 1),
        dt.date(2025, 10, 3),
    ]
    days = mcx_sessions(start, dt.date(2025, 10, 3))
    result = held_run(GOLDM_OCTOBER, days, [78_000.0 + 10.0 * n for n in range(len(days))])
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "expiry_exit"
    assert pd.Timestamp(trade["exit_ts"]).date() == dt.date(2025, 9, 29)
    assert not any(note.startswith("GOLDM03OCT25FUT") for note in result.warnings)


def test_a_stock_future_held_to_its_last_day_exits_there_and_an_index_future_settles():
    days = [dt.date(2026, 2, day) for day in (24, 25, 26)]
    stock = held_run(SBIN_FUTURE, days, [1_000.0, 1_010.0, 1_020.0])
    assert stock.trades.iloc[0]["exit_reason"] == "expiry_exit"
    assert stock.trades.iloc[0]["exit_price_rupees"] == 1_020.0
    index = held_run(NIFTY_FUTURE, days, [26_000.0, 26_010.0, 26_020.0])
    assert index.trades.iloc[0]["exit_reason"] == "expiry"
