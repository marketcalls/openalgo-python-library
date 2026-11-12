"""Quick start for openalgo.backtest: an EMA crossover backtest on your own bars.

Runs anywhere, with no OpenAlgo server: the daily bars are synthetic. Replace
`daily_bars()` with your own DataFrame of open, high, low, close and volume (indexed by
IST time) to test real prices, or pass an OpenAlgo client instead of `data=` to use
history from OpenAlgo (Historify when it holds the whole range, else your broker):

    result = bt.backtest(EmaCross, client, start=START, end=END, capital=CAPITAL)

Run it from the repository root:

    python examples/backtest_quickstart.py
"""

import numpy as np
import pandas as pd

import openalgo.backtest as bt
from openalgo import ta

# IST session dates of the run, inclusive.
START, END = "2026-04-01", "2026-11-06"
# Rupees allocated to the strategy.
CAPITAL = 100_000


def daily_bars(start: str = START, end: str = END) -> pd.DataFrame:
    """Synthetic daily bars, one per NSE session from `start` to `end`: a slow rise with
    two waves on it, prices in rupees, indexed by the IST session date."""
    sessions = bt.calendar.default_calendar().sessions("NSE", start, end)
    k = np.arange(len(sessions))
    close = (1000 + k + 60 * np.sin(k / 6) + 10 * np.sin(k / 1.3)).round(1)
    open_ = (close - 4 * np.sin(1.3 * k)).round(1)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 6,
            "low": np.minimum(open_, close) - 6,
            "close": close,
            "volume": 50_000,
        },
        index=pd.DatetimeIndex(sessions["date"]).tz_localize("Asia/Kolkata"),
    )


class EmaCross(bt.Strategy):
    """Long while the 5-day EMA is above the 20-day EMA, on SBIN delivery (CNC)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    fast = 5
    slow = 20

    def init(self):
        self.fast_ema = self.I(ta.ema, self.close, self.fast)
        self.slow_ema = self.I(ta.ema, self.close, self.slow)

    def on_bar(self, bar):
        if len(self.close) <= self.slow:  # let the slow EMA warm up first
            return
        if self.position.is_flat and self.fast_ema.crossed_above(self.slow_ema):
            self.buy()  # as many shares as the capital allows after estimated charges
        elif self.position.is_long and self.fast_ema.crossed_below(self.slow_ema):
            self.exit()


def main() -> None:
    """Run the backtest and print the result summary and the headline figures."""
    result = bt.backtest(EmaCross, data=daily_bars(), capital=CAPITAL)
    print(result)
    metrics = result.metrics
    print(f"Net profit   : Rs {metrics['net_profit']:,.2f}")
    print(f"Total return : {metrics['total_return']:.2%}")
    print(f"Max drawdown : {metrics['max_drawdown']:.2%}")
    print(f"Trades       : {metrics['trades']} ({metrics['win_rate']:.0%} won)")
    print(f"Charges      : Rs {metrics['total_costs']:,.2f}")


if __name__ == "__main__":
    main()
