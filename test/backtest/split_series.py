"""Series shared by the split and bonus tests.

Daily NSE bars of one share from Monday 2024-07-01 to Friday 2024-07-12 (ten sessions,
NSE hours of the fake calendar). `gap_at_ex_date` gives the reference shape: 2,000.00
(open and close) up to Friday 2024-07-05, then a gap at the open of Monday 2024-07-08.
"""

import datetime as dt

import numpy as np
import pandas as pd
from data_fixtures.fakes import FakeCalendar, days_between

CAL = FakeCalendar()
SESSIONS = [d for d in days_between(dt.date(2024, 7, 1), dt.date(2024, 7, 12)) if d.weekday() < 5]
EX_DATE = dt.date(2024, 7, 8)
BEFORE = 2_000.0


def frame(ts_open, step_ns, opens, closes, exchange, interval):
    """Canonical bars; the high and low are 1.00 beyond the open and close."""
    opens = np.asarray(opens, dtype="float64")
    closes = np.asarray(closes, dtype="float64")
    bars = pd.DataFrame(
        {
            "ts_open": np.asarray(ts_open, dtype="int64"),
            "ts_close": np.asarray(ts_open, dtype="int64") + step_ns,
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": np.full(len(opens), 10_000.0),
            "oi": np.full(len(opens), np.nan),
        }
    )
    bars.attrs.update(symbol="SHARE", exchange=exchange, interval=interval)
    return bars


def daily(prices, exchange="NSE", interval="D", days=SESSIONS):
    """Daily bars from (open, close) pairs, one per session in `days` (NSE hours)."""
    opens = [CAL.session("NSE", day)[0] for day in days]
    session_ns = CAL.session("NSE", days[0])[1] - opens[0]
    return frame(opens, session_ns, *zip(*prices, strict=True), exchange, interval)


def gap_at_ex_date(ex_open, ex_close, after=None, before=BEFORE):
    """The reference series: `before` up to 2024-07-05, then the ex-date opening at
    `ex_open` and closing at `ex_close`, then `after` (default `ex_close`)."""
    after = ex_close if after is None else after
    return [(before, before)] * 5 + [(ex_open, ex_close)] + [(after, after)] * 4


def issue(report, check):
    """The one issue of `check` in a quality report."""
    found = [item for item in report.issues if item.check == check]
    assert len(found) == 1, f"expected one {check} issue, got {[i.check for i in report.issues]}"
    return found[0]
