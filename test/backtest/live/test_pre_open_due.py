"""The bar a daily or longer strategy may decide before today's open (`_pre_open_due`;
design 20-positional-live section 8; review finding 1 of unit N5b): only the bar that closed
at the exchange's last session before today, as a backtest decides it for today's open.

Each row gives a strategy's interval, today, the closes of its bars of its interval and of
its daily candles (by the IST day of the session they close at, the NSE calendar's close),
and the last bar its journal decided; it expects the last decided bar the kernel is told and
the warning, if any. The sessions are NSE's of 2026: Monday 2026-11-30 ends November and
Wednesday 2026-09-30 the third quarter.
"""

from __future__ import annotations

import datetime as dt

import pytest

from openalgo.backtest._live._pre_open_due import (
    DECIDED_TODAY,
    DUE_EARLIER,
    MISSING_CANDLE,
    StrategyBars,
    due_bar,
)
from openalgo.backtest.calendar import default_calendar


def day(month: int, date: int) -> dt.date:
    return dt.date(2026, month, date)


def close(session: dt.date) -> int:
    """NSE's close of `session`, UTC nanoseconds."""
    return default_calendar().session("NSE", session)[1]


def open_of(session: dt.date) -> int:
    return default_calendar().session("NSE", session)[0]


def due(interval, today, closes, daily, last):
    bars = StrategyBars(
        name="S",
        interval=interval,
        exchange="NSE",
        closes=[close(c) for c in closes],
        daily_closes=[close(d) for d in daily],
        open_ns=open_of(today),
    )
    return due_bar(bars, today, None if last is None else close(last), default_calendar())


# (interval, today, bar closes, daily candle closes, journal's last) -> (told, warning)
ROWS = [
    pytest.param(
        "M",
        day(12, 1),
        [day(10, 30), day(11, 30)],
        [day(11, 27), day(11, 30)],
        day(10, 30),
        (day(10, 30), None),
        id="a monthly bar is due the morning after the month's last session",
    ),
    pytest.param(
        "M",
        day(12, 1),
        [day(10, 30), day(11, 30)],
        [day(11, 27)],
        day(10, 30),
        (
            day(11, 30),
            MISSING_CANDLE.format(name="S", interval="M", day="2026-11-30"),
        ),
        id="a monthly bar whose last daily candle is missing is not decided",
    ),
    pytest.param(
        "M",
        day(11, 16),
        [day(10, 30)],
        [day(11, 13)],
        None,
        (
            day(10, 30),
            DUE_EARLIER.format(name="S", interval="M", closed="2026-10-30", due="2026-11-02"),
        ),
        id="a first start in the middle of a month decides nothing",
    ),
    pytest.param(
        "M",
        day(11, 16),
        [day(10, 30)],
        [day(11, 13)],
        day(10, 30),
        (day(10, 30), None),
        id="in the middle of a month a decided bar is quiet",
    ),
    pytest.param(
        "Q",
        day(10, 1),
        [day(6, 30), day(9, 30)],
        [day(9, 29), day(9, 30)],
        day(6, 30),
        (day(6, 30), None),
        id="a quarterly bar is due the morning after the quarter's last session",
    ),
    pytest.param(
        "D",
        day(11, 16),
        [day(11, 12), day(11, 13)],
        [day(11, 12), day(11, 13)],
        day(11, 13),
        (day(11, 13), DECIDED_TODAY.format(name="S", interval="D", day="2026-11-13")),
        id="a daily bar an earlier start of today decided is not decided again",
    ),
]


@pytest.mark.parametrize(("interval", "today", "closes", "daily", "last", "expected"), ROWS)
def test_the_bar_due_today_is_the_one_that_closed_at_the_last_session(
    interval, today, closes, daily, last, expected
):
    told, warning = expected

    answer = due(interval, today, closes, daily, last)

    assert (answer.last_decided_ns, answer.note) == (close(told), warning)
