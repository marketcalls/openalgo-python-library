"""The excursion and trading-cost metrics of `metrics.analyse` on a hand-built trades
table, and NaN with no error for a table without their columns.

Three closed trades, each with Rs 1,000 at work (entry price x quantity), and one open:

    #  side  entry  exit  qty  gross  costs  net   mae  mfe  mae_pct  mfe_pct  slippage
    1  BUY   100    110   10   100    10     90    -20  120  -2.0     12.0     1.00
    2  SELL  200    210    5   -50    10     -60   -60   15  -6.0      1.5     2.00
    3  BUY    50     52   20    40     5     35    -10   50  -1.0      5.0     0.50
    4  BUY   100    open  10     open, no excursion yet

Average MAE (-2 - 6 - 1) / 3 = -3 percent; average MFE (12 + 1.5 + 5) / 3 = 6.1667
percent; worst MAE -60, best MFE 120; edge ratio 6.1667 / 3 = 2.0556; the winners (net
above 0) are 1 and 3, gross 140 over MFE 170 kept 0.8235. Slippage 3.50; charges 25.00;
trading cost 28.50; gross P&L 90.00, so 31.67 percent of it; the closed trades traded
2,100 + 2,050 + 2,040 = 6,190.00 (their turnover; the open trade's entry adds 1,000.00 to
the turnover of every trade), so 28.50 is 46.04 basis points of it.
"""

import math

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.metrics import DEFINITIONS, GROUPS

IST = "Asia/Kolkata"
EXCURSION_KEYS = (
    "average_mae",
    "average_mfe",
    "worst_mae",
    "best_mfe",
    "edge_ratio",
    "profit_kept",
)
#: The closed trades' turnover: known from prices and quantities, with or without the
#: optional columns.
CLOSED_TURNOVER = "closed_turnover"
COST_KEYS = (
    "total_slippage",
    "trading_cost",
    "trading_cost_share_of_gross_pnl",
    "trading_cost_bps",
)
OPTIONAL = ("mae", "mfe", "mae_pct", "mfe_pct", "slippage")


def at(day: int, clock: str = "10:00") -> pd.Timestamp:
    return pd.Timestamp(f"2025-07-{day:02d} {clock}", tz=IST)


def trades_table() -> pd.DataFrame:
    rows = [
        (1, 2, 100.0, 110.0, 10.0, "BUY", 100.0, 10.0, -20.0, 120.0, -2.0, 12.0, 1.0),
        (2, 3, 200.0, 210.0, 5.0, "SELL", -50.0, 10.0, -60.0, 15.0, -6.0, 1.5, 2.0),
        (3, 4, 50.0, 52.0, 20.0, "BUY", 40.0, 5.0, -10.0, 50.0, -1.0, 5.0, 0.5),
    ]
    frame = pd.DataFrame(
        [
            {
                "entry_time": at(entry),
                "exit_time": at(exit_day, "14:00"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "side": side,
                "gross_pnl": gross,
                "costs": costs,
                "net_pnl": gross - costs,
                **dict(zip(OPTIONAL, figures, strict=True)),
            }
            for (
                entry,
                exit_day,
                entry_price,
                exit_price,
                quantity,
                side,
                gross,
                costs,
                *figures,
            ) in rows
        ]
    )
    still_open = {
        "entry_time": at(4, "14:30"),
        "exit_time": pd.NaT,
        "entry_price": 100.0,
        "exit_price": np.nan,
        "quantity": 10.0,
        "side": "BUY",
        "gross_pnl": 0.0,
        "costs": 0.0,
        "net_pnl": 0.0,
        **dict.fromkeys(OPTIONAL, np.nan),
    }
    table = pd.concat([frame, pd.DataFrame([still_open])], ignore_index=True)
    table["exit_time"] = pd.to_datetime(table["exit_time"], utc=True).dt.tz_convert(IST)
    return table


def analysed(trades: pd.DataFrame) -> metrics.Performance:
    days = [at(day, "15:30") for day in range(1, 8)]
    equity = pd.Series(np.linspace(1_00_000.0, 1_00_065.0, len(days)), index=pd.DatetimeIndex(days))
    return metrics.analyse(equity, exchanges="NSE", trades=trades, opening_wealth=1_00_000.0)


@pytest.fixture(scope="module")
def figures():
    return analysed(trades_table()).metrics


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("average_mae", -0.03),
        ("average_mfe", 18.5 / 3 / 100),
        ("worst_mae", -60.0),
        ("best_mfe", 120.0),
        ("edge_ratio", (18.5 / 3) / 3.0),
        ("profit_kept", 140.0 / 170.0),
    ],
)
def test_the_excursion_figures_of_the_closed_trades(figures, key, expected):
    assert figures[key] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("total_slippage", 3.5),
        ("trading_cost", 28.5),
        ("trading_cost_share_of_gross_pnl", 28.5 / 90.0),
        ("trading_cost_bps", 28.5 / 6_190.0 * 10_000),
    ],
)
def test_the_trading_cost_of_the_closed_trades(figures, key, expected):
    assert figures[key] == pytest.approx(expected)


def test_the_trading_cost_is_measured_against_the_closed_trades_turnover(figures):
    # 6,190.00 of the closed trades; the turnover of every trade adds the open entry.
    assert figures[CLOSED_TURNOVER] == pytest.approx(6_190.0)
    assert figures["turnover"] == pytest.approx(7_190.0)
    expected = figures["trading_cost"] / figures[CLOSED_TURNOVER] * 10_000
    assert figures["trading_cost_bps"] == pytest.approx(expected)


def test_the_existing_figures_are_untouched_by_the_new_columns(figures):
    # The same table without the new columns gives every earlier figure unchanged.
    plain = analysed(trades_table().drop(columns=list(OPTIONAL))).metrics
    for key in plain:
        if key in EXCURSION_KEYS + COST_KEYS:
            continue
        same = plain[key] == figures[key] or (
            isinstance(plain[key], float) and math.isnan(plain[key]) and math.isnan(figures[key])
        )
        assert same, key


def test_a_table_without_the_columns_gives_nan_figures_and_no_error():
    plain = analysed(trades_table().drop(columns=list(OPTIONAL))).metrics
    for key in EXCURSION_KEYS + COST_KEYS:
        assert math.isnan(plain[key]), key
    assert plain["total_costs"] == 25.0


def test_a_fill_without_modelled_slippage_makes_the_slippage_unknown_not_zero():
    trades = trades_table()
    trades.loc[1, "slippage"] = np.nan
    figures = analysed(trades).metrics
    assert math.isnan(figures["total_slippage"]) and math.isnan(figures["trading_cost"])
    assert figures["average_mae"] == pytest.approx(-0.03)


def test_no_closed_trade_has_no_excursion_and_no_slippage():
    figures = analysed(trades_table().iloc[[3]]).metrics
    for key in EXCURSION_KEYS:
        assert math.isnan(figures[key]), key
    assert (figures["total_slippage"], figures["trading_cost"]) == (0.0, 0.0)
    assert figures[CLOSED_TURNOVER] == 0.0
    assert math.isnan(figures["trading_cost_share_of_gross_pnl"])
    assert math.isnan(figures["trading_cost_bps"])


def test_trades_that_never_went_against_have_an_infinite_edge_ratio():
    trades = trades_table()
    trades[["mae", "mae_pct"]] = 0.0
    assert analysed(trades).metrics["edge_ratio"] == math.inf


def test_every_new_figure_is_defined_in_its_own_group():
    group = GROUPS["Excursions and trading cost"]
    assert group == EXCURSION_KEYS + (CLOSED_TURNOVER,) + COST_KEYS
    units = [DEFINITIONS[key].unit for key in group]
    assert units == ["fraction", "fraction", "rupees", "rupees", "ratio", "fraction"] + [
        "rupees",
        "rupees",
        "rupees",
        "fraction",
        "bps",
    ]
