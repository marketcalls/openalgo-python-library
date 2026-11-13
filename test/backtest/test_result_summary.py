"""`Result.summary()`: the few figures a beginner prints after a run, as plain text: net
profit in rupees with Indian digit grouping, the return, the maximum drawdown, the closed
trades and their win rate ("no closed trades" when there are none, where the win rate is
undefined), and the benchmark's return for a run with one.

The runs, with no charges and no slippage:

- the daily round trip of `accounting_runs`: 2,000 shares bought at 1,000.00 on
  2026-11-03 and sold at 1,100.00 on 2026-11-04, from 50,00,000 of capital. Net profit
  2,000 x 100.00 = 2,00,000, a return of 2,00,000 / 50,00,000 = 4.00%; the equity never
  falls below a peak, so the maximum drawdown is 0.00%; one winning trade.
- `Hold` of `benchmark_runs`: 100 shares bought at 1,000.00 and held while the share
  closes at 1,010, 1,030, 1,020 and 1,040, from 1,00,000. The equity ends at
  1,00,000 + 100 x 40 = 1,04,000: a net profit of 4,000 and a return of 4.00%. Its
  deepest fall is from 1,03,000 to 1,02,000, 1,02,000 / 1,03,000 - 1 = -0.97%. The
  position is still open: no trade has closed. NIFTY moves by 1%, 2%, -1%, 0.5% and 1%
  on the five sessions: 1.01 x 1.02 x 0.99 x 1.005 x 1.01 - 1 = 3.52%.
- `SellsToOpen`: a daily stock strategy whose three delivery shorts are all rejected, so
  nothing ever trades.

The wording rules are checked on `summary_text` with figures given by hand: a win rate
never reads 0% or 100% unless it is exactly so, and an amount that rounds to zero has no
minus sign.
"""

import pytest
from accounting_runs import CAPITAL, daily_bars, daily_round_trip
from benchmark_runs import index_bars, nifty_closes, own_run, share_bars

import openalgo.backtest as bt
from openalgo.backtest._summary import summary_text


class SellsToOpen(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.sell(qty=10)


def test_a_run_with_trades_is_summarised_in_four_lines():
    result = daily_round_trip({"qty": 2_000}, capital=5_000_000, costs=None)
    assert result.summary() == (
        "Net profit   : Rs 2,00,000.00\n"
        "Return       : 4.00%\n"
        "Max drawdown : 0.00%\n"
        "Trades       : 1 (100% won)"
    )


def test_a_run_that_never_traded_says_no_closed_trades_instead_of_a_win_rate():
    result = bt.backtest(SellsToOpen, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)
    assert result.summary() == (
        "Net profit   : Rs 0.00\n"
        "Return       : 0.00%\n"
        "Max drawdown : 0.00%\n"
        "Trades       : no closed trades"
    )


def test_a_position_held_to_the_end_is_counted_as_still_open():
    assert own_run().summary() == (
        "Net profit   : Rs 4,000.00\n"
        "Return       : 4.00%\n"
        "Max drawdown : -0.97%\n"
        "Trades       : no closed trades (1 still open)"
    )


def test_a_run_with_a_benchmark_index_adds_its_return():
    result = own_run(data={"SBIN": share_bars(), "NIFTY": index_bars()}, benchmark="NIFTY")
    assert result.summary().splitlines()[-1] == "Benchmark    : 3.52% return (NIFTY)"


def test_a_benchmark_given_as_closes_is_named_as_given():
    result = own_run(benchmark=nifty_closes())
    assert result.summary().splitlines()[-1] == "Benchmark    : 3.52% return (given closes)"


def test_a_run_without_a_benchmark_has_no_benchmark_line():
    assert "Benchmark" not in own_run().summary()


def figures(**changed) -> dict:
    """The metrics `summary_text` reads, for a run of 250 closed trades and none open
    that made nothing; `changed` replaces some."""
    base = {
        "net_profit": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "trades": 250,
        "open_trades": 0,
        "win_rate": 0.5,
        "benchmark_total_return": float("nan"),
    }
    return base | changed


def trades_line(**changed) -> str:
    return summary_text(figures(**changed), None).splitlines()[3]


@pytest.mark.parametrize(
    ("trades", "winners", "shown"),
    [
        (250, 1, "0.4%"),  # whole percent would read 0%
        (250, 249, "99.6%"),  # whole percent would read 100%
        (3_000, 1, "0.03%"),  # one decimal would still read 0.0%
        (250, 0, "0%"),  # no winner at all
        (250, 250, "100%"),  # every trade won
        (250, 90, "36%"),  # whole percent where it does not mislead
    ],
)
def test_the_win_rate_shows_zero_or_a_hundred_percent_only_when_exact(trades, winners, shown):
    line = trades_line(trades=trades, win_rate=winners / trades)
    assert line == f"Trades       : {trades} ({shown} won)"


def test_a_loss_below_half_a_paisa_reads_as_zero_rupees():
    lines = summary_text(figures(net_profit=-0.004), None).splitlines()
    assert lines[0] == "Net profit   : Rs 0.00"
