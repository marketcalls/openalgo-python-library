"""The contract multiplier in the traded value: a trades table may carry each trade's
`multiplier` (rupees per one-rupee move per unit of quantity, 1 when the column is
absent), and the turnover and the trade list's return count it.

One gold future quoted per 10 grams and traded in 1 kg units (multiplier 100): bought at
72,000.00 and sold at 72,100.00, one unit, gross P&L 1 x 100 x 100 = 10,000.00, charges
40.00. Its contract value at entry is 72,000 x 1 x 100 = 72,00,000.00.
"""

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest._trade_list import trade_list
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def gold_trade(**columns) -> pd.DataFrame:
    row = {
        "entry_time": pd.Timestamp("2025-07-01 10:00", tz=IST),
        "exit_time": pd.Timestamp("2025-07-01 14:00", tz=IST),
        "entry_price": 72_000.0,
        "exit_price": 72_100.0,
        "quantity": 1.0,
        "side": "BUY",
        "gross_pnl": 10_000.0,
        "net_pnl": 9_960.0,
        "costs": 40.0,
        **columns,
    }
    return pd.DataFrame([row])


def analysed(trades: pd.DataFrame) -> metrics.Performance:
    equity = pd.Series(
        [10_000_000.0, 10_009_960.0],
        index=pd.DatetimeIndex(["2025-07-01 15:30", "2025-07-02 15:30"]).tz_localize(IST),
    )
    return metrics.analyse(equity, exchanges="MCX", trades=trades, opening_wealth=10_000_000.0)


@pytest.mark.parametrize(
    ("columns", "turnover"),
    [
        # 72,000 x 1 x 100 + 72,100 x 1 x 100.
        ({"multiplier": 100.0}, 14_410_000.0),
        # Without the column every multiplier is 1: 72,000 + 72,100.
        ({}, 144_100.0),
    ],
)
def test_the_turnover_counts_each_trades_multiplier(columns, turnover):
    assert analysed(gold_trade(**columns)).metrics["turnover"] == pytest.approx(turnover)


def test_the_trade_lists_return_is_over_the_contract_value():
    # 9,960 over 72,00,000 is 0.138333 percent.
    listing = trade_list(gold_trade(multiplier=100.0))
    assert listing["return_pct"].iloc[0] == pytest.approx(9_960 / 7_200_000 * 100)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_a_multiplier_that_is_not_a_positive_number_is_refused(bad):
    with pytest.raises(ConfigError, match="multiplier"):
        analysed(gold_trade(multiplier=bad))
