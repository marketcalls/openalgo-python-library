"""What `bt.backtest` does with a split or bonus in its data (FR-MKT-10; PRD 04 section 7,
PRD 06 section 5): a suspected one stops the run, a declared one is a warning, and
holding across a declared one the bars are not adjusted for is not built (FR-MKT-11,
Phase 5).

The data: daily bars of one NSE share on the six sessions 2026-10-05 (Monday) to
2026-10-12, opening and closing at 2,000.00 up to 2026-10-07 and at 1,000.00 from the
ex-date, Thursday 2026-10-08 (high and low 10.00 either side). A daily strategy decides
at a session's close and fills at the next session's open, so the order decided at the
close of bar n (1 for 2026-10-05) fills at the open of session n + 1. No charges, no
slippage, CNC.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from accounting_runs import IST, future_round_trip

import openalgo.backtest as bt
from openalgo.backtest.data import CorporateAction
from openalgo.backtest.errors import ConfigError, DataQualityError

SESSIONS = [dt.date(2026, 10, d) for d in (5, 6, 7, 8, 9, 12)]
EX_DATE = "2026-10-08"
BONUS = [CorporateAction.bonus(EX_DATE, 1, 1)]
QTY = 10


def halved_bars() -> pd.DataFrame:
    """The daily bars of the module docstring."""
    prices = np.array([2_000.0] * 3 + [1_000.0] * 3)
    index = pd.DatetimeIndex(
        [dt.datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST) for d in SESSIONS],
        name="timestamp",
    )
    columns = {
        "open": prices,
        "high": prices + 10.0,
        "low": prices - 10.0,
        "close": prices,
        "volume": np.full(len(prices), 50_000.0),
    }
    return pd.DataFrame(columns, index=index)


def holding(entry_bar: int, exit_bar: int | None, **entry) -> type[bt.Strategy]:
    """Buys QTY units at the close of bar `entry_bar` and exits at the close of
    `exit_bar` (never when None)."""

    def on_bar(self, bar):
        if len(self.close) == entry_bar:
            self.buy(qty=QTY, **entry)
        elif len(self.close) == exit_bar and not self.position.is_flat:
            self.exit()

    attributes = {"symbol": "SBIN", "exchange": "NSE", "interval": "D", "on_bar": on_bar}
    return type("Holding", (bt.Strategy,), attributes)


def run(strategy, **settings):
    arguments = {"data": halved_bars(), "capital": 1_000_000, "costs": None, "slippage": 0}
    return bt.backtest(strategy, **{**arguments, **settings})


# --- not declared ---------------------------------------------------------------------


def test_a_suspected_unadjusted_split_stops_the_run_and_names_the_way_on():
    with pytest.raises(DataQualityError) as caught:
        run(holding(1, 2))
    message = str(caught.value)
    assert "1 suspected unadjusted split or bonus (2026-10-08: the previous close 2000.00" in (
        message
    )
    assert "declare what happened with corporate_actions=[...]" in message


def minute_bars() -> pd.DataFrame:
    """1-minute bars of 2026-10-07 at 2,000.00 and 2026-10-08 at 1,000.00 (09:15 to 15:30)."""
    stamps, prices = [], []
    for day, price in ((SESSIONS[2], 2_000.0), (SESSIONS[3], 1_000.0)):
        opens = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
        stamps += [opens + dt.timedelta(minutes=k) for k in range(375)]
        prices += [price] * 375
    prices = np.array(prices)
    columns = {"open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices}
    columns["volume"] = np.full(len(prices), 100.0)
    return pd.DataFrame(columns, index=pd.DatetimeIndex(stamps, name="timestamp"))


def test_an_intraday_run_stops_on_the_split_instead_of_trying_another_base_interval():
    attributes = {"symbol": "SBIN", "exchange": "NSE", "interval": "15m"}
    strategy = type("Flat", (bt.Strategy,), {**attributes, "on_bar": lambda self, bar: None})
    with pytest.raises(DataQualityError, match="1 suspected unadjusted split or bonus"):
        run(strategy, data=minute_bars())


# --- declared, nothing held across ----------------------------------------------------


def test_a_declared_split_crossed_while_flat_runs_and_warns():
    result = run(holding(1, 2), corporate_actions=BONUS)
    assert result.trades["gross_pnl_rupees"].tolist() == [0.0]  # 2,000.00 in and out
    assert any(
        text.startswith(
            "SBIN: the bars are not adjusted for 1 declared corporate actions (2026-10-08: "
            "a bonus (2 shares for every 1); the previous close 2000.00 against the open "
            "1000.00 is a ratio of 2.000)"
        )
        for text in result.warnings
    )


def test_an_entry_that_fills_at_the_ex_date_open_is_not_held_across_it():
    result = run(holding(3, 4), corporate_actions=BONUS)
    trade = result.trades.iloc[0]
    assert (trade["entry_price_rupees"], trade["exit_price_rupees"]) == (1_000.0, 1_000.0)


def test_the_manifest_records_the_declarations():
    arguments = run(holding(1, 2), corporate_actions=BONUS).manifest["arguments"]
    assert arguments["corporate_actions"] == [
        {"ex_date": "2026-10-08", "kind": "bonus", "factor": 2.0}
    ]


# --- declared, held across: not built -------------------------------------------------


def test_a_position_held_across_a_declared_split_is_not_built_yet():
    # Bought at the 2026-10-06 open, still held at the 2026-10-07 close.
    with pytest.raises(NotImplementedError) as caught:
        run(holding(1, 4), corporate_actions=BONUS)
    assert str(caught.value).startswith(
        "SBIN: a position is held across 2026-10-08, the ex-date of a bonus (2 shares for "
        "every 1) that the bars are not adjusted for. Adjusting the units, average price, "
        "levels and GTTs of a holding at a corporate action is FR-MKT-11, planned for "
        "Phase 5."
    )


def test_an_exit_that_fills_at_the_ex_date_open_still_holds_across_it():
    # Decided at the 2026-10-07 close: the position is held into the ex-date's open.
    with pytest.raises(NotImplementedError, match="a position is held across 2026-10-08"):
        run(holding(1, 3), corporate_actions=BONUS)


def test_a_gtt_resting_across_a_declared_split_is_not_built_yet():
    # A buy GTT triggered at 2,500.00, placed at the 2026-10-05 close, never fires.
    strategy = holding(1, None, price=2_500.0, validity="gtt")
    with pytest.raises(NotImplementedError, match="SBIN: a GTT rests across 2026-10-08"):
        run(strategy, corporate_actions=BONUS)


# --- declared as a genuine move -------------------------------------------------------


def test_a_position_held_across_a_genuine_move_keeps_its_units_and_takes_the_loss():
    result = run(holding(1, 4), corporate_actions=[CorporateAction.none(EX_DATE)])
    trade = result.trades.iloc[0]
    assert (trade["entry_price_rupees"], trade["exit_price_rupees"]) == (2_000.0, 1_000.0)
    assert trade["gross_pnl_rupees"] == QTY * (1_000.0 - 2_000.0)
    assert any("read as genuine price moves" in text for text in result.warnings)


# --- arguments ------------------------------------------------------------------------


def test_declarations_for_a_symbol_the_run_does_not_trade_are_a_config_error():
    with pytest.raises(ConfigError, match=r"corporate_actions names \['INFY'\]"):
        run(holding(1, 2), corporate_actions={"INFY": BONUS})


def test_declarations_by_symbol_reach_their_symbol():
    result = run(holding(1, 2), corporate_actions={"SBIN": BONUS})
    assert any(text.startswith("SBIN: the bars are not adjusted") for text in result.warnings)


def test_declarations_for_a_futures_contract_are_a_config_error():
    with pytest.raises(ConfigError, match="corporate_actions apply to shares on NSE and BSE"):
        future_round_trip(corporate_actions=BONUS)
