"""The warm-up of an anchored indicator (decision D25, `_warm_up_anchor`): a VWAP that
restarts at each anchor period needs the bars since the start of the period that holds
`start`, is loaded from there, and is waited for when the data begin inside it.

The bars are the seeded walks of `warm_up_bars` (volume 1,000 units a bar, so a VWAP is
the plain average of the bars' (high + low + close) / 3 since its anchor). The run
starts on Friday 2026-03-20; its anchor periods begin on Monday 2026-03-16 (Week),
2026-03-01 (Month), 2026-01-01 (Quarter and Year). Expected values are those of
`ta.vwap` over all the bars given.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from warm_up_bars import IST, SETTINGS, intraday, sessions, walk

import openalgo.backtest as bt
from openalgo import ta

DAILY = walk("2020-01-01", "2026-06-30", seed=5)
START = dt.date(2026, 3, 20)
PERIOD_STARTS = {
    "Week": dt.date(2026, 3, 16),
    "Month": dt.date(2026, 3, 1),
    "Quarter": dt.date(2026, 1, 1),
    "Year": dt.date(2026, 1, 1),
}


def anchored(anchor, interval: str = "D", product: str = "CNC") -> type[bt.Strategy]:
    """A strategy that registers the VWAP anchored on `anchor` and records its first
    decision: (bar open, the VWAP)."""

    class Anchored(bt.Strategy):
        symbol, exchange = "SBIN", "NSE"
        seen: list = []

        def init(self):
            self.vwap = self.I(ta.vwap, self.high, self.low, self.close, self.volume, anchor=anchor)

        def on_bar(self, bar):
            if not self.seen:
                self.seen.append((bar.time, self.vwap[-1]))

    Anchored.interval, Anchored.product, Anchored.seen = interval, product, []
    return Anchored


def full_history(bars: pd.DataFrame, anchor: str, at: dt.datetime) -> float:
    values = ta.vwap(bars.high, bars.low, bars.close, bars.volume, anchor=anchor)
    return float(np.asarray(values)[bars.index.get_loc(pd.Timestamp(at))])


@pytest.mark.parametrize("anchor", list(PERIOD_STARTS))
def test_an_anchored_vwap_is_computed_from_the_start_of_its_period(anchor):
    strategy = anchored(anchor)
    bt.backtest(strategy, data=DAILY, start=START, **SETTINGS)
    ((at, value),) = strategy.seen
    assert at.date() == START
    assert value == pytest.approx(full_history(DAILY, anchor, at), rel=1e-12)


@pytest.mark.parametrize("anchor", list(PERIOD_STARTS))
def test_the_history_of_an_anchored_vwap_is_asked_from_the_start_of_its_period(anchor):
    result = bt.backtest(anchored(anchor), data=DAILY, start=START, **SETTINGS)
    text = result.assumptions["warm_up"]
    assert f"needs the D bars of SBIN since the start of its {anchor}" in text
    assert f"(asked from {PERIOD_STARTS[anchor]})" in text
    assert text.endswith("ready at start")


def test_the_manifest_records_the_anchor_period_of_the_need():
    result = bt.backtest(anchored("Month"), data=DAILY, start=START, **SETTINGS)
    (need,) = result.manifest["arguments"]["warm_up"]["needs"]
    assert (need["bars"], need["since"]) == (None, "Month")


def test_a_session_vwap_after_a_datetime_start_holds_the_morning_before_it():
    bars = intraday("2026-03-02", "2026-03-13", minutes=5)
    start = dt.datetime(2026, 3, 10, 13, 0, tzinfo=IST)
    strategy = anchored("Session", "5m", "MIS")
    bt.backtest(strategy, data=bars, start=start, end="2026-03-13", base="5m", **SETTINGS)
    ((at, value),) = strategy.seen
    assert value == pytest.approx(full_history(bars, "Session", at), rel=1e-12)


def test_data_that_begin_inside_the_anchor_period_wait_for_the_next_period():
    # The bars begin on 2026-03-10, inside March: the monthly VWAP of March is partial,
    # so the strategy decides from the close of the first session of April.
    data = DAILY[DAILY.index.date >= dt.date(2026, 3, 10)]
    strategy = anchored("Month")
    result = bt.backtest(strategy, data=data, start=START, **SETTINGS)
    april = sessions("2026-04-01", "2026-04-30")[0]
    assert result.manifest["arguments"]["warm_up"]["first_decision"] == f"{april} 15:30 IST"
    (warning,) = [note for note in result.warnings if "waited" in note]
    assert warning.endswith(
        "needs the D bars of SBIN since the start of its Month and 8 bars closed before "
        f"{START}: the bars given for SBIN begin on 2026-03-10; give earlier ones to decide "
        f"from {START}."
    )


def test_a_vwap_without_an_anchor_runs_over_all_its_history_and_is_not_waited_for():
    result = bt.backtest(anchored(None), data=DAILY, start=START, **SETTINGS)
    (warning,) = [note for note in result.warnings if "did not wait" in note]
    assert "does not settle on the engine's reference bars" in warning
