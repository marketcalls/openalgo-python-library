"""`bt.backtest(positional_stops=...)` (design 20-positional-live section 6, decisions P5
and 11.4; unit N4 item N4-1): "gtt" (the default) holds a CNC or NRML position's stop as
one SINGLE GTT; "rearm_daily" holds it as an MIS position's DAY stop, placed again at each
session's open, and the run says what that assumes.

SBIN on NSE daily bars, CNC, no charges, slippage one tick (0.05). A LIMIT buy of 100 at
1,000.00 is decided at the close of Mon 2025-06-30 and fills at the 07-01 open of 1,000.00
(a resting limit takes no slippage); its stop 50 points below, 950.00.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

DAYS = [dt.date(2025, 6, 30), dt.date(2025, 7, 1), dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
FLAT = (1000.0, 1005.0, 995.0, 1000.0)
# Day 2 opens at 940.00, beyond the 950.00 stop, and trades back up to 945.00.
GAP = (940.0, 945.0, 930.0, 942.0)


def bars(day_two: tuple[float, float, float, float]) -> pd.DataFrame:
    rows = [FLAT, FLAT, day_two, FLAT]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([pd.Timestamp(day) for day in DAYS]).tz_localize("Asia/Kolkata")
    return frame


class DipHolder(bt.Strategy):
    """Rests a CNC buy of 100 at 1,000.00 with a 50-point stop at the first close."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    product = "CNC"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=100, price=1000.0, stop=bt.points(50))


class OtherHolder(DipHolder):
    """The same, a second strategy of a portfolio run."""


def run(positional_stops, strategy=DipHolder, **settings) -> bt.Result | bt.PortfolioResult:
    return bt.backtest(
        strategy,
        data=bars(GAP),
        capital=1_000_000,
        costs=None,
        slippage=bt.ticks(1),
        positional_stops=positional_stops,
        **settings,
    )


def exit_fill(result: bt.Result) -> pd.Series:
    fills = result.fills
    return fills.iloc[-1]


def test_rearm_daily_exits_a_gap_beyond_the_stop_at_the_open_less_slippage_flagged():
    # No stop is placed at day 2's open: sold at 940.00 less a tick, 939.95.
    result = run("rearm_daily")
    trade = result.trades.iloc[0]
    assert (trade["exit_reason"], trade["exit_price_rupees"]) == ("leg_sl", 939.95)
    assert "stop_crossed" in exit_fill(result)["flags"]


def test_the_default_gtt_gives_the_gtts_child_fill_on_the_same_gap():
    # The GTT fires on the gap; its child sells 1 percent below the 950.00 trigger, at
    # 940.50, reached when the bar trades back up to 945.00.
    result = run("gtt")
    trade = result.trades.iloc[0]
    assert (trade["exit_reason"], trade["exit_price_rupees"]) == ("leg_sl", 940.5)
    assert "stop_crossed" not in exit_fill(result)["flags"]


def test_a_rearm_daily_run_warns_what_it_assumes_and_states_the_setting():
    result = run("rearm_daily")
    notes = [note for note in result.warnings if note.startswith("positional_stops=")]
    assert len(notes) == 1
    assert "started before the open on every trading day a position was held" in notes[0]
    assert result.assumptions["positional_stops"] == "rearm_daily"


def test_a_gtt_run_has_no_such_warning_and_states_nothing_new():
    result = run("gtt")
    assert not any(note.startswith("positional_stops=") for note in result.warnings)
    assert "positional_stops" not in result.assumptions


@pytest.mark.parametrize("value", ["overnight", "GTT", None, 1])
def test_any_other_value_is_refused(value):
    with pytest.raises(ConfigError, match="positional_stops takes one of"):
        run(value)


def test_a_portfolio_run_takes_it_by_strategy_name():
    portfolio = run(
        {"DipHolder": "rearm_daily", "OtherHolder": "gtt"}, strategy=[DipHolder, OtherHolder]
    )
    rearmed = portfolio.strategy("DipHolder").trades.iloc[0]["exit_price_rupees"]
    held = portfolio.strategy("OtherHolder").trades.iloc[0]["exit_price_rupees"]
    assert (rearmed, held) == (939.95, 940.5)


def test_a_portfolio_dict_that_leaves_a_strategy_out_is_refused():
    with pytest.raises(ConfigError, match="positional_stops"):
        run({"DipHolder": "rearm_daily"}, strategy=[DipHolder, OtherHolder])
