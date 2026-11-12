"""A run charged at rates that were never verified says so once, briefly: one short
UnverifiedScheduleWarning per process for each cost schedule and segment, with the
details (the dates, the schedule's notes) in `result.warnings`. A sweep reports it from
the parent process only, never once per worker.

The runs hold SBIN on delivery (CNC, daily bars) in July 2025, when cost schedule
IN-2024-10-01 is in force; its NSE_EQ_DELIVERY rates were never verified against their
source.
"""

import datetime as dt
import warnings

import pandas as pd
from benchmark_runs import CAPITAL, Hold

import openalgo.backtest as bt
from openalgo.backtest.costs import UnverifiedScheduleWarning

DAYS = [dt.date(2025, 7, day) for day in (1, 2, 3, 4)]
SHORT = (
    "the NSE_EQ_DELIVERY rates of cost schedule IN-2024-10-01 were never verified against "
    "their source, so the charges may be off (details in result.warnings)"
)


def july_bars() -> pd.DataFrame:
    """Daily SBIN bars at 1,000.00, stamped at IST midnight of each session."""
    index = pd.DatetimeIndex([pd.Timestamp(day).tz_localize("Asia/Kolkata") for day in DAYS])
    columns = {"open": 1_000.0, "high": 1_010.0, "low": 990.0, "close": 1_000.0}
    frame = pd.DataFrame(columns, index=index)
    frame["volume"] = 1_000.0
    return frame


def july_run() -> bt.Result:
    return bt.backtest(Hold, data=july_bars(), capital=CAPITAL, slippage=0, now="2025-07-10")


def unverified(caught: list) -> list[str]:
    return [str(item.message) for item in caught if item.category is UnverifiedScheduleWarning]


def test_runs_in_one_process_warn_once_in_short_form():
    with warnings.catch_warnings(record=True) as caught:
        # Python's default action: show a warning once for each place it is issued.
        warnings.simplefilter("default")
        july_run()
        july_run()
    assert unverified(caught) == [SHORT]


def test_the_details_are_in_the_result_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedScheduleWarning)
        result = july_run()
    detail = (
        "charges: the NSE_EQ_DELIVERY rates of cost schedule IN-2024-10-01 used for "
        "2025-07-01 were never verified against their source, so the charges may be off."
    )
    assert any(text.startswith(detail) for text in result.warnings)


def test_a_sweep_warns_from_the_parent_and_not_from_its_workers(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        bt.optimize(
            Hold,
            data=july_bars(),
            capital=CAPITAL,
            slippage=0,
            now="2025-07-10",
            qty=range(10, 60, 10),
            workers=2,
            progress=lambda line: None,
        )
    assert unverified(caught) == [SHORT]
    assert "UnverifiedScheduleWarning" not in capfd.readouterr().err
