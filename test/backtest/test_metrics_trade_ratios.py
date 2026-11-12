"""Ratios of closed trades: win/loss count ratio, Kelly criterion, risk of ruin, CPC
index and the common sense ratio, and the losing streaks of trades.

Trades close one per session from Monday 2 June 2025, in the order given, with no
costs. Expected values are the published formulas applied to the literal P&L.
"""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics

IST = "Asia/Kolkata"


def at(day: pd.Timestamp, clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {clock}", tz=IST)


def run(net: list[float], capital: float = 10_000.0) -> metrics.Performance:
    """One trade a session with net P&L `net`, on an equity that books them."""
    days = pd.bdate_range("2025-06-02", periods=max(len(net), 1), tz=IST)
    closes = [at(day, "15:30") for day in days]
    equity = pd.Series(capital + pd.Series(net or [0.0]).cumsum().to_numpy(), index=closes)
    trades = pd.DataFrame(
        {
            "entry_time": [at(day, "10:00") for day in days[: len(net)]],
            "exit_time": [at(day, "15:00") for day in days[: len(net)]],
            "entry_price": 100.0,
            "exit_price": [100.0 + p for p in net],
            "quantity": 1.0,
            "side": "BUY",
            "gross_pnl": net,
            "net_pnl": net,
            "costs": 0.0,
        }
    )
    return metrics.analyse(equity, exchanges="NSE", trades=trades, opening_wealth=capital)


def test_win_loss_ratio_kelly_and_cpc_index():
    m = run([100.0, -50.0, 200.0, -50.0, 0.0]).metrics
    # Two wins, two losses; the breakeven trade is neither.
    assert m["win_loss_ratio"] == pytest.approx(1.0)
    # p = 2 / 4 among decided trades, payoff 150 / 50 = 3: 0.5 - 0.5 / 3.
    assert m["kelly_criterion"] == pytest.approx(0.5 - 0.5 / 3)
    # Profit factor 300 / 100 x win rate 2 / 5 x payoff 3.
    assert m["cpc_index"] == pytest.approx(3 * 0.4 * 3)


def test_the_common_sense_ratio_is_profit_factor_times_tail_ratio():
    m = run([100.0, -50.0, 200.0, -50.0, 0.0]).metrics
    assert m["common_sense_ratio"] == pytest.approx(m["profit_factor"] * m["tail_ratio"])


def test_the_risk_of_ruin_is_the_lundberg_bound():
    # +2 and -1 rupees, equally likely, from 10 rupees: the adjustment coefficient is
    # the log of the golden ratio, so the bound is golden ratio ^ -10.
    m = run([2.0, -1.0], capital=10.0).metrics
    golden = (1 + math.sqrt(5)) / 2
    assert m["risk_of_ruin"] == pytest.approx(golden**-10)


def test_a_mean_loss_makes_ruin_certain():
    assert run([100.0, -150.0]).metrics["risk_of_ruin"] == 1.0


def test_losing_trade_streaks_are_counted_by_length():
    p = run([-10.0, -10.0, 5.0, -10.0, 5.0, -10.0, -10.0])
    streaks = p.losing_streaks
    assert list(streaks.index) == [1, 2]
    assert streaks.loc[1, "trades"] == 1 and streaks.loc[2, "trades"] == 2


def test_all_winning_trades():
    m = run([10.0, 20.0, 30.0]).metrics
    assert m["win_loss_ratio"] == math.inf
    assert m["risk_of_ruin"] == 0.0
    assert math.isnan(m["kelly_criterion"])


def test_all_losing_trades():
    m = run([-10.0, -20.0]).metrics
    assert m["win_loss_ratio"] == 0.0
    assert m["risk_of_ruin"] == 1.0
    assert math.isnan(m["kelly_criterion"])


def test_no_trades_have_no_trade_ratio():
    m = run([]).metrics
    for key in ("win_loss_ratio", "kelly_criterion", "risk_of_ruin", "cpc_index"):
        assert math.isnan(m[key]), key
