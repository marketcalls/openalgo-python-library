"""A run charged at rates that are not verified says so once, briefly: one short
UnverifiedScheduleWarning per process for each cost schedule and segment, with the
details (the dates, the schedule's notes) in `result.warnings`. A sweep reports it from
the parent process only, never once per worker.

The runs hold SBIN on delivery (CNC, daily bars) in December 2026, when cost schedule
IN-2026-04-01 is in force; its NSE_EQ_DELIVERY rates are verified only up to 2026-11-12.
"""

import datetime as dt
import warnings

import pandas as pd
from benchmark_runs import CAPITAL, Hold

import openalgo.backtest as bt
from openalgo.backtest.costs import UnverifiedScheduleWarning

DAYS = [dt.date(2026, 12, day) for day in (1, 2, 3, 4)]
SHORT = (
    "the NSE_EQ_DELIVERY rates of cost schedule IN-2026-04-01 are verified only up to an "
    "earlier date, so the charges may be off (details in result.warnings)"
)


def december_bars() -> pd.DataFrame:
    """Daily SBIN bars at 1,000.00, stamped at IST midnight of each session."""
    index = pd.DatetimeIndex([pd.Timestamp(day).tz_localize("Asia/Kolkata") for day in DAYS])
    columns = {"open": 1_000.0, "high": 1_010.0, "low": 990.0, "close": 1_000.0}
    frame = pd.DataFrame(columns, index=index)
    frame["volume"] = 1_000.0
    return frame


def december_run() -> bt.Result:
    return bt.backtest(Hold, data=december_bars(), capital=CAPITAL, slippage=0, now="2026-12-10")


def unverified(caught: list) -> list[str]:
    return [str(item.message) for item in caught if item.category is UnverifiedScheduleWarning]


def test_runs_in_one_process_warn_once_in_short_form():
    with warnings.catch_warnings(record=True) as caught:
        # Python's default action: show a warning once for each place it is issued.
        warnings.simplefilter("default")
        december_run()
        december_run()
    assert unverified(caught) == [SHORT]


def test_the_details_are_in_the_result_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedScheduleWarning)
        result = december_run()
    detail = (
        "charges: the NSE_EQ_DELIVERY rates of cost schedule IN-2026-04-01 used for "
        "2026-12-01 are verified only up to an earlier date, so the charges may be off."
    )
    assert any(text.startswith(detail) for text in result.warnings)


def test_a_sweep_warns_from_the_parent_and_not_from_its_workers(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        bt.optimize(
            Hold,
            data=december_bars(),
            capital=CAPITAL,
            slippage=0,
            now="2026-12-10",
            qty=range(10, 60, 10),
            workers=2,
            progress=lambda line: None,
        )
    assert unverified(caught) == [SHORT]
    assert "UnverifiedScheduleWarning" not in capfd.readouterr().err
