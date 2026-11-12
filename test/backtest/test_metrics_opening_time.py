"""The opening time: the instant a run opened with its capital, before its first point.

With `opening_time` the run starts there: elapsed time and CAGR are counted from it,
the first bar runs from it to the first point (so exposure counts the first slice), and
a drawdown from the opening is timed from it. Without it, everything is as before.
2 June 2025 is a Monday; closes are at 15:30 IST.
"""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"
NANOS_PER_DAY = 86_400 * 10**9


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def closes(*values: float) -> pd.Series:
    days = pd.bdate_range("2025-06-02", periods=len(values), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


OPEN = at("2025-06-02", "09:15")


def analyse(equity: pd.Series, **kwargs) -> dict:
    return metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0, **kwargs).metrics


def test_the_run_starts_at_the_opening_time():
    m = analyse(closes(101.0, 102.0, 103.0), opening_time=OPEN)
    assert m["start"] == OPEN
    # 09:15 Monday to 15:30 Wednesday: two days and six and a quarter hours.
    assert m["elapsed_days"] == pytest.approx(2 + 6.25 / 24)


def test_cagr_is_annualised_from_the_opening_time():
    equity = closes(*[100.0 + k for k in range(1, 31)])
    m = analyse(equity, opening_time=OPEN)
    elapsed = (equity.index[-1] - OPEN).value / NANOS_PER_DAY
    assert m["cagr"] == pytest.approx(1.30 ** (365.25 / elapsed) - 1)
    without = analyse(equity)
    first = (equity.index[-1] - equity.index[0]).value / NANOS_PER_DAY
    assert without["cagr"] == pytest.approx(1.30 ** (365.25 / first) - 1)


def held_in_the_first_slice() -> pd.DataFrame:
    """One trade from the 09:15 open to 15:00 on Monday, inside the first slice."""
    return pd.DataFrame(
        {
            "entry_time": [OPEN],
            "exit_time": [at("2025-06-02", "15:00")],
            "entry_price": [100.0],
            "exit_price": [101.0],
            "quantity": [1.0],
            "side": ["BUY"],
            "gross_pnl": [1.0],
            "net_pnl": [1.0],
            "costs": [0.0],
        }
    )


def test_exposure_counts_the_first_slice_from_the_opening_time():
    equity = closes(101.0, 101.0, 101.0)
    with_time = analyse(equity, trades=held_in_the_first_slice(), opening_time=OPEN)
    # Bars: 09:15 to the Monday close, then Monday to Tuesday and Tuesday to Wednesday.
    assert with_time["exposure"] == pytest.approx(1 / 3)
    # Without it, time before the first point is not a bar: the trade overlaps none.
    assert analyse(equity, trades=held_in_the_first_slice())["exposure"] == 0.0


def test_a_drawdown_from_the_opening_is_timed_from_the_opening_time():
    m = analyse(closes(95.0, 101.0), opening_time=OPEN)
    assert m["max_drawdown_peak"] == OPEN
    assert m["max_drawdown_recovery"] == at("2025-06-03")
    assert m["max_drawdown_days"] == pytest.approx(1 + 6.25 / 24)


def test_the_benchmark_base_is_read_at_the_opening_time():
    equity = closes(102.0, 104.04)
    bench = pd.Series([1000.0, 1010.0, 1020.0], index=[OPEN, *equity.index])
    m = analyse(equity, opening_time=OPEN, benchmark=bench)
    assert m["benchmark_sessions"] == 2
    assert m["benchmark_total_return"] == pytest.approx(0.02)


def test_an_opening_time_as_utc_nanoseconds_is_accepted():
    m = analyse(closes(101.0, 102.0), opening_time=int(OPEN.value))
    assert m["start"] == OPEN


@pytest.mark.parametrize(
    "value",
    [at("2025-06-02"), at("2025-06-03"), pd.Timestamp("2025-06-02 09:15"), "09:15"],
)
def test_an_opening_time_that_is_not_an_aware_time_before_the_first_point_is_refused(value):
    with pytest.raises(ConfigError):
        analyse(closes(101.0, 102.0), opening_time=value)


def test_an_opening_time_needs_an_opening_wealth():
    with pytest.raises(ConfigError, match="opening_wealth"):
        metrics.analyse(closes(101.0, 102.0), exchanges="NSE", opening_time=OPEN)


def test_without_an_opening_time_the_start_is_the_first_point():
    m = analyse(closes(101.0, 102.0))
    assert m["start"] == at("2025-06-02")
    assert math.isnan(m["cagr"])
