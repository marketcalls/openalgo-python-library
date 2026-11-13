"""Rejected orders are told on screen: when any order of a run was rejected, `bt.backtest`
ends with one `RejectedOrdersWarning` (a Python warning, so it can be filtered by its
class) that says how many orders were rejected, the most common reason in plain words and
where the details are. A sweep's combinations never issue it.

The runs use the bars of `accounting_runs`: one NSE session on 2026-11-04 in 1-minute
bars (a 15-minute strategy decides at 09:30, 09:45, 10:00 and 10:15, and an entry decided
after an entry cut-off of 09:40 is rejected), and the daily bars of 2026-11-02 to
2026-11-04. Capital is 10,00,000 and every price is near 1,000.00, so a buy of 5,000
shares (50,00,000) is rejected for margin.
"""

import warnings

from accounting_runs import CAPITAL, daily_bars, round_trip, session_bars
from benchmark_runs import QTY_RANGE, Hold, share_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import RejectedOrdersWarning


class SellsToOpen(bt.Strategy):
    """A daily stock strategy that sells 10 shares at every close while flat: on
    delivery (CNC) each sale would open a short, so all three are rejected."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.sell(qty=10)


class BuysTooMuchThenLate(bt.Strategy):
    """Buys 5,000 shares at 09:30 (margin), then 10 shares at 09:45, 10:00 and 10:15."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=5_000)
        elif len(self.close) <= 4:
            self.buy(qty=10)


class BuysTwice(bt.Strategy):
    """Buys 5,000 shares at 09:30 (margin) and 10 at 09:45 (after the cut-off)."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=5_000)
        elif len(self.close) == 2:
            self.buy(qty=10)


class BuysFourTimes(bt.Strategy):
    """Buys 10 shares at each of its first four bar closes (09:30 to 10:15)."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) <= 4:
            self.buy(qty=10)


ENTRY_WINDOW = (
    "orders after the MIS entry cut-off (entry_until) are rejected, and so are entries "
    "before entry_from"
)


def notices(run, *args, **settings) -> list[warnings.WarningMessage]:
    """The RejectedOrdersWarnings `run(*args, **settings)` issues, every one recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(*args, **settings)
    return [item for item in caught if item.category is RejectedOrdersWarning]


def daily(strategy: type) -> bt.Result:
    return bt.backtest(strategy, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)


def intraday(strategy: type, **settings) -> bt.Result:
    arguments = {"data": session_bars(), "capital": CAPITAL, "base": "1m", "product": "MIS"}
    return bt.backtest(strategy, costs=None, slippage=0, **arguments, **settings)


def test_a_run_whose_orders_were_all_rejected_for_one_reason_says_why_once():
    (notice,) = notices(daily, SellsToOpen)
    assert str(notice.message) == (
        "3 of 3 orders were rejected because a delivery (CNC) short is not allowed in "
        "Indian cash markets (reject_code 'product_rule'). See result.warnings."
    )


def test_the_notice_counts_the_rejected_orders_among_all_the_orders():
    # 09:30 and 09:45 are before the 09:50 cut-off; 10:00 and 10:15 are after it.
    (notice,) = notices(intraday, BuysFourTimes, entry_until="09:50")
    assert str(notice.message) == (
        f"2 of 4 orders were rejected because {ENTRY_WINDOW} (reject_code 'entry_window'). "
        "See result.warnings."
    )


def test_several_reasons_are_told_by_the_most_common_one():
    # 09:30: margin; 09:45, 10:00 and 10:15: after the 09:40 cut-off.
    (notice,) = notices(intraday, BuysTooMuchThenLate, entry_until="09:40")
    assert str(notice.message) == (
        f"4 of 4 orders were rejected, 3 of them because {ENTRY_WINDOW} (reject_code "
        "'entry_window', the most common reason). See result.warnings."
    )


def test_reasons_with_as_many_orders_are_told_by_the_one_that_occurred_first():
    (notice,) = notices(intraday, BuysTwice, entry_until="09:40")
    assert str(notice.message) == (
        "2 of 2 orders were rejected, 1 of them because the margin the order needed was "
        "more than the free cash (reject_code 'margin', the most common reason). See "
        "result.warnings."
    )


def test_the_notice_points_at_the_line_that_ran_the_backtest():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt.backtest(SellsToOpen, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)
    (notice,) = [item for item in caught if item.category is RejectedOrdersWarning]
    assert notice.filename == __file__


def test_the_notice_is_a_user_warning_that_its_class_silences():
    assert issubclass(RejectedOrdersWarning, UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", RejectedOrdersWarning)
        daily(SellsToOpen)
    assert [item for item in caught if item.category is RejectedOrdersWarning] == []


def test_a_run_without_rejected_orders_gives_no_notice():
    assert notices(round_trip, {"qty": 10}, costs=None, slippage=0) == []


def test_a_sweep_gives_no_notice_from_this_process_or_its_workers(capfd):
    # 25, 50 or 100 shares at 1,000.00 need 25,000 or more against 10,000 of capital,
    # so every combination's only order is rejected for margin.
    sweep = {"data": share_bars(), "capital": 10_000, "costs": None, "slippage": 0}
    assert len(notices(bt.backtest, Hold, **sweep)) == 1
    found = notices(
        bt.optimize, Hold, **sweep, qty=QTY_RANGE, workers=2, progress=lambda line: None
    )
    assert found == []
    assert "RejectedOrdersWarning" not in capfd.readouterr().err
