"""An open short on cash equity is marked correctly in the equity curve at every bar
while it is open (01-requirements 6.4: short positions mis-marked in cash mode is one of
the pitfalls the engine must not repeat).

One NSE share on 2026-11-04, a 1-minute MIS strategy on 1-minute bars, no charges and no
slippage, capital 10,00,000. The strategy sells 10 shares at the close of bar 0 (filled
at the open of bar 1, 1,000.00) and covers at the close of bar 4 (filled at the open of
bar 5, 985.00). Bars (open, high, low, close):

    bar 0  09:15  1000  1001   999  1000
    bar 1  09:16  1000  1003   999  1002
    bar 2  09:17  1002  1006  1001  1005
    bar 3  09:18  1005  1005   990   991
    bar 4  09:19   991   993   984   985
    bar 5  09:20   985   987   983   986   and the same bar to 09:24

Cash is the capital plus what the sale brought in less what the cover paid; the short's
value is minus 10 times the close; equity is their sum. The rows below are worked by
hand from those two lines, at the close of each bar (the equity table's index).
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2026, 11, 4, 9, 15, tzinfo=IST)
CAPITAL = 1_000_000
UNITS = 10
BARS = [
    (1000.0, 1001.0, 999.0, 1000.0),
    (1000.0, 1003.0, 999.0, 1002.0),
    (1002.0, 1006.0, 1001.0, 1005.0),
    (1005.0, 1005.0, 990.0, 991.0),
    (991.0, 993.0, 984.0, 985.0),
    *[(985.0, 987.0, 983.0, 986.0)] * 5,
]

# (bar, cash, value of the short, equity, unrealized P&L, gross exposure), rupees.
HAND_ROWS = [
    (0, 1_000_000.0, 0.0, 1_000_000.0, 0.0, 0.0),
    # Sold 10 at 1,000.00: cash 10,00,000 + 10,000 = 10,10,000.
    (1, 1_010_000.0, -10_020.0, 999_980.0, -20.0, 10_020.0),
    (2, 1_010_000.0, -10_050.0, 999_950.0, -50.0, 10_050.0),
    (3, 1_010_000.0, -9_910.0, 1_000_090.0, 90.0, 9_910.0),
    (4, 1_010_000.0, -9_850.0, 1_000_150.0, 150.0, 9_850.0),
    # Covered 10 at 985.00: cash 10,10,000 - 9,850 = 10,00,150.
    (5, 1_000_150.0, 0.0, 1_000_150.0, 0.0, 0.0),
]


def close_of(bar: int) -> dt.datetime:
    """The close of bar `bar`: the stamp of its equity row."""
    return SESSION_OPEN + dt.timedelta(minutes=bar + 1)


def bars() -> pd.DataFrame:
    frame = pd.DataFrame(BARS, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    stamps = [SESSION_OPEN + dt.timedelta(minutes=k) for k in range(len(BARS))]
    frame.index = pd.DatetimeIndex(stamps, name="timestamp")
    return frame


class ShortThenCover(bt.Strategy):
    """Sells short at the close of bar 0 and covers at the close of bar 4."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.sell(qty=UNITS)
        elif len(self.close) == 5:
            self.exit()


@pytest.fixture(scope="module")
def result() -> bt.Result:
    return bt.backtest(
        ShortThenCover,
        data=bars(),
        capital=CAPITAL,
        costs=None,
        slippage=0,
        base="1m",
        product="MIS",
        end=SESSION_OPEN + dt.timedelta(minutes=len(BARS) - 1),
    )


def test_the_short_opens_and_covers_at_the_next_opens(result):
    fills = result.fills
    assert list(fills["side"]) == ["SELL", "BUY"]
    assert list(fills["price_rupees"]) == [1000.0, 985.0]
    assert list(fills["quantity_units"]) == [UNITS, UNITS]


@pytest.mark.parametrize("row", HAND_ROWS, ids=[f"bar {row[0]}" for row in HAND_ROWS])
def test_equity_is_cash_plus_the_short_marked_at_the_close(result, row):
    bar, cash, value, equity, _, _ = row
    assert cash + value == equity
    assert result.equity.loc[close_of(bar), "equity_rupees"] == equity


@pytest.mark.parametrize("row", HAND_ROWS, ids=[f"bar {row[0]}" for row in HAND_ROWS])
def test_the_open_short_gains_when_the_price_falls_and_loses_when_it_rises(result, row):
    bar, _, _, _, unrealized, _ = row
    assert result.equity.loc[close_of(bar), "unrealized_rupees"] == unrealized


@pytest.mark.parametrize("row", HAND_ROWS, ids=[f"bar {row[0]}" for row in HAND_ROWS])
def test_the_exposure_of_a_short_is_its_gross_notional(result, row):
    bar, _, _, _, _, exposure = row
    assert result.equity.loc[close_of(bar), "exposure_rupees"] == exposure


def test_the_covered_short_is_one_short_trade_with_its_profit(result):
    # (1,000.00 - 985.00) x 10 = 150.00.
    trade = result.trades.iloc[0]
    assert (trade["direction"], trade["gross_pnl_rupees"]) == ("short", 150.0)
    assert result.equity["equity_rupees"].iloc[-1] == CAPITAL + 150.0
