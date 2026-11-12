"""Trade statistics, exposure, turnover and costs, counted per closed trade.

The equity curve is the start at 09:15 on Monday 2 June 2025 and one close per weekday;
trade figures are worked out by hand from the tables below.
"""

import math

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def equity(points: int = 5, value: float = 1000.0) -> pd.Series:
    closes = pd.bdate_range("2025-06-02", periods=points - 1, tz=IST)
    index = [at("2025-06-02", "09:15"), *(closes + pd.Timedelta(hours=15, minutes=30))]
    return pd.Series(value, index=index)


def trades(rows: list[dict]) -> pd.DataFrame:
    """A trades table; each row gives entry, exit and net P&L, the rest defaults."""
    frame = []
    for row in rows:
        net = row["net"]
        costs = row.get("costs", 0.0)
        frame.append(
            {
                "entry_time": row.get("entry", at("2025-06-02", "10:00")),
                "exit_time": row["exit"],
                "entry_price": row.get("entry_price", 100.0),
                "exit_price": row.get("exit_price", 100.0 + (net + costs) / 10),
                "quantity": row.get("quantity", 10.0),
                "side": row.get("side", "BUY"),
                "gross_pnl": net + costs,
                "net_pnl": net,
                "costs": costs,
                **row.get("extra", {}),
            }
        )
    return pd.DataFrame(frame)


def stats(rows: list[dict], **kwargs) -> dict:
    return dict(metrics.analyse(equity(), exchanges="NSE", trades=trades(rows), **kwargs).metrics)


def exits(*hours: int) -> list[pd.Timestamp]:
    return [at("2025-06-02", "10:00") + pd.Timedelta(hours=h) for h in hours]


# Given out of exit order; by exit time the net P&L reads +100, -50, 0, +200, -25, -25.
SIX = [
    {"exit": e, "net": n}
    for e, n in zip(exits(4, 1, 6, 2, 5, 3), [200.0, 100.0, -25.0, -50.0, -25.0, 0.0], strict=True)
]


def test_counts_rates_and_averages():
    m = stats(SIX)
    assert (m["trades"], m["winning_trades"], m["losing_trades"]) == (6, 2, 3)
    assert m["win_rate"] == pytest.approx(2 / 6)
    assert m["average_win"] == pytest.approx(150.0)
    assert m["average_loss"] == pytest.approx(-100 / 3)
    assert m["expectancy"] == pytest.approx(200 / 6)
    assert (m["largest_win"], m["largest_loss"]) == (200.0, -50.0)


def test_profit_factor_and_payoff_ratio():
    m = stats(SIX)
    assert m["profit_factor"] == pytest.approx(300 / 100)
    assert m["payoff_ratio"] == pytest.approx(150 / (100 / 3))


def test_consecutive_runs_follow_exit_order_and_breakeven_breaks_them():
    m = stats(SIX)
    assert (m["max_consecutive_wins"], m["max_consecutive_losses"]) == (1, 2)


def test_no_trades_leave_trade_figures_undefined_and_sums_zero():
    m = dict(metrics.analyse(equity(), exchanges="NSE").metrics)
    assert (m["trades"], m["open_trades"], m["exposure"], m["turnover"]) == (0, 0, 0.0, 0.0)
    assert (m["gross_pnl"], m["total_costs"], m["net_pnl"]) == (0.0, 0.0, 0.0)
    for key in (
        "win_rate",
        "profit_factor",
        "expectancy",
        "payoff_ratio",
        "average_holding_seconds",
    ):
        assert math.isnan(m[key]), key


def test_one_winning_trade():
    m = stats([{"exit": at("2025-06-03"), "net": 40.0}])
    assert (m["trades"], m["win_rate"], m["profit_factor"]) == (1, 1.0, math.inf)
    assert math.isnan(m["payoff_ratio"]) and math.isnan(m["average_loss"])


def test_average_holding_time_is_the_mean_of_exit_minus_entry():
    rows = [
        {"entry": at("2025-06-03", "10:00"), "exit": at("2025-06-03", "11:00"), "net": 1.0},
        {"entry": at("2025-06-03", "10:00"), "exit": at("2025-06-03", "13:00"), "net": 1.0},
    ]
    assert stats(rows)["average_holding_seconds"] == pytest.approx(2 * 3600)


def test_exposure_is_the_share_of_bars_overlapping_a_position():
    # Bars end at the closes of 2, 3, 4 and 5 June; held from 3 June 09:15 to 4 June
    # 09:15, which overlaps the bars ending on 3 June and 4 June.
    rows = [{"entry": at("2025-06-03", "09:15"), "exit": at("2025-06-04", "09:15"), "net": 5.0}]
    assert stats(rows)["exposure"] == pytest.approx(2 / 4)


def test_an_open_trade_counts_for_exposure_and_turnover_only():
    rows = [
        {"entry": at("2025-06-03", "09:15"), "exit": at("2025-06-03", "14:00"), "net": 10.0},
        {
            "entry": at("2025-06-04", "09:15"),
            "exit": pd.NaT,
            "net": math.nan,
            "exit_price": math.nan,
        },
    ]
    frame = trades(rows)
    frame.loc[1, ["gross_pnl", "costs"]] = math.nan
    m = metrics.analyse(equity(), exchanges="NSE", trades=frame).metrics
    assert (m["trades"], m["open_trades"], m["net_pnl"]) == (1, 1, 10.0)
    # Bars ending 3 June (first trade), 4 and 5 June (held to the end): 3 of 4.
    assert m["exposure"] == pytest.approx(3 / 4)
    # Entries 100 x 10 twice, one exit at 101 x 10.
    assert m["turnover"] == pytest.approx(1000 + 1000 + 1010)


def test_turnover_counts_both_legs_and_its_ratio_uses_mean_equity():
    rows = [{"exit": at("2025-06-03"), "net": 100.0, "entry_price": 100.0, "exit_price": 110.0}]
    m = stats(rows)
    assert m["turnover"] == pytest.approx(100 * 10 + 110 * 10)
    assert m["turnover_ratio"] == pytest.approx(2100 / 1000)


def test_a_short_trade_with_a_negative_price_counts_its_magnitude_in_turnover():
    rows = [
        {
            "exit": at("2025-06-03"),
            "net": 50.0,
            "side": "SELL",
            "entry_price": -5.0,
            "exit_price": -10.0,
        }
    ]
    assert stats(rows)["turnover"] == pytest.approx(50 + 100)


def test_costs_and_their_share_of_gross_pnl():
    rows = [
        {
            "exit": at("2025-06-03"),
            "net": 100.0,
            "costs": 20.0,
            "extra": {"brokerage": 12.0, "stt": 8.0},
        },
        {
            "exit": at("2025-06-04"),
            "net": -50.0,
            "costs": 10.0,
            "extra": {"brokerage": 6.0, "stt": 4.0},
        },
    ]
    result = metrics.analyse(equity(), exchanges="NSE", trades=trades(rows))
    m = result.metrics
    assert (m["gross_pnl"], m["total_costs"], m["net_pnl"]) == (80.0, 30.0, 50.0)
    assert m["cost_share_of_gross_pnl"] == pytest.approx(30 / 80)
    assert dict(result.cost_breakdown) == {"brokerage": 18.0, "stt": 12.0}


def test_the_cost_share_is_undefined_for_a_losing_gross_pnl():
    m = stats([{"exit": at("2025-06-03"), "net": -60.0, "costs": 10.0}])
    assert math.isnan(m["cost_share_of_gross_pnl"])


def test_net_pnl_must_be_gross_pnl_minus_costs():
    frame = trades([{"exit": at("2025-06-03"), "net": 100.0, "costs": 10.0}])
    frame.loc[0, "costs"] = 9.0
    with pytest.raises(ConfigError, match="trade 0"):
        metrics.analyse(equity(), exchanges="NSE", trades=frame)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"side": "LONG"}, "side"),
        ({"quantity": 0.0}, "quantity"),
        ({"quantity": -10.0}, "quantity"),
        ({"entry_price": math.nan}, "entry_price"),
        ({"entry_time": pd.NaT}, "entry_time"),
    ],
)
def test_unusable_trade_values_are_refused(change, message):
    frame = trades([{"exit": at("2025-06-03"), "net": 1.0}])
    for column, value in change.items():
        frame[column] = pd.Series(
            [value], dtype=frame[column].dtype if column != "side" else object
        )
    with pytest.raises(ConfigError, match=message):
        metrics.analyse(equity(), exchanges="NSE", trades=frame)


def test_a_missing_column_is_named():
    frame = trades([{"exit": at("2025-06-03"), "net": 1.0}]).drop(columns=["costs"])
    with pytest.raises(ConfigError, match="costs"):
        metrics.analyse(equity(), exchanges="NSE", trades=frame)


def test_a_trade_exiting_before_it_enters_is_refused():
    rows = [{"entry": at("2025-06-04"), "exit": at("2025-06-03"), "net": 1.0}]
    with pytest.raises(ConfigError, match="exits before it enters"):
        stats(rows)


def test_naive_trade_times_are_refused():
    frame = trades([{"exit": at("2025-06-03"), "net": 1.0}])
    frame["exit_time"] = frame["exit_time"].dt.tz_localize(None)
    with pytest.raises(ConfigError, match="time zone"):
        metrics.analyse(equity(), exchanges="NSE", trades=frame)


def test_the_trades_table_is_kept_as_given():
    frame = trades(SIX)
    frame["symbol"] = "SBIN"
    result = metrics.analyse(equity(), exchanges="NSE", trades=frame)
    assert result.trades is frame
    assert np.array_equal(result.trades["net_pnl"].to_numpy(), frame["net_pnl"].to_numpy())
