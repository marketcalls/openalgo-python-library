"""Orders that were not placed are told on screen: when orders of a run were rejected or
skipped for a reason the run did not ask for (margin, the product rule, a size that
rounds down to zero, ...), `bt.backtest` ends with one `OrdersNotPlacedWarning` (a Python
warning, so it can be filtered by its class) that says how many, the most common reason
in plain words and where the details are. Orders a rule of the run refused as it intends
(the MIS entry window, a size cap) are left to `result.warnings`. A sweep's combinations
never issue it.

The runs use the bars of `accounting_runs`: one NSE session on 2026-11-04 in 1-minute
bars (a 15-minute strategy decides at 09:30, 09:45, 10:00 and 10:15, and an entry decided
after an entry cut-off of 09:40 is rejected), and the daily bars of 2026-11-02 to
2026-11-04 (a daily strategy decides at each 15:30 close). Every price is near 1,000.00:
from 10,00,000 of capital a buy of 5,000 shares (50,00,000) is rejected for margin, and
from 500 an order with no quantity (500 less the 2 percent cash buffer, 490 at 1,000.00:
0.49 shares) is skipped as rounded to zero.
"""

import warnings

from accounting_runs import CAPITAL, daily_bars, round_trip, session_bars
from benchmark_runs import QTY_RANGE, Hold, share_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import OrdersNotPlacedWarning

MARGIN = "the margin the order needed was more than the free cash"
ROUNDED = "the sized quantity was less than one share or, for a derivative, one lot"


class SellsToOpen(bt.Strategy):
    """A daily stock strategy that sells 10 shares at every close while flat: on
    delivery (CNC) each sale would open a short, so all three are rejected."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.sell(qty=10)


class BuysWhileFlat(bt.Strategy):
    """A daily stock strategy that buys with the engine default at every close while
    flat."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy()


class TooMuchThenDefault(bt.Strategy):
    """Buys 5,000 shares at the first close (margin), then with the engine default at
    the second and third (rounded to zero from 500 of capital)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=5_000)
        else:
            self.buy()


class TooMuchThenDefaultOnce(bt.Strategy):
    """Buys 5,000 shares at the first close (margin) and with the engine default at the
    second (rounded to zero from 500 of capital)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=5_000)
        elif len(self.close) == 2:
            self.buy()


class DefaultThenTooMuch(bt.Strategy):
    """Buys with the engine default at the first close (rounded to zero from 500 of
    capital) and 5,000 shares at the second (margin)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy()
        elif len(self.close) == 2:
            self.buy(qty=5_000)


class BuysTooMuchThenLate(bt.Strategy):
    """Buys 5,000 shares at 09:30 (margin), then 10 shares at 09:45, 10:00 and 10:15."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(qty=5_000)
        elif len(self.close) <= 4:
            self.buy(qty=10)


class BuysFourTimes(bt.Strategy):
    """Buys 10 shares at each of its first four bar closes (09:30 to 10:15)."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) <= 4:
            self.buy(qty=10)


class CappedWithoutStop(bt.Strategy):
    """Buys half its equity at 09:30 with no stop under a risk cap: skipped ("cap")."""

    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"
    size_caps = bt.size.caps(risk=bt.pct(0.5))

    def on_bar(self, bar):
        if len(self.close) == 1:
            self.buy(size=bt.size.percent(50))


def notices(run, *args, **settings) -> list[warnings.WarningMessage]:
    """The OrdersNotPlacedWarnings `run(*args, **settings)` issues, every one recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(*args, **settings)
    return [item for item in caught if item.category is OrdersNotPlacedWarning]


def daily(strategy: type, capital: float = CAPITAL) -> bt.Result:
    return bt.backtest(strategy, data=daily_bars(), capital=capital, costs=None, slippage=0)


def intraday(strategy: type, **settings) -> bt.Result:
    arguments = {"data": session_bars(), "capital": CAPITAL, "base": "1m", "product": "MIS"}
    return bt.backtest(strategy, costs=None, slippage=0, **arguments, **settings)


def test_a_run_whose_orders_were_all_rejected_for_one_reason_says_why_once():
    (notice,) = notices(daily, SellsToOpen)
    assert str(notice.message) == (
        "3 of 3 orders were rejected because a delivery (CNC) short is not allowed in "
        "Indian cash markets (reject_code 'product_rule'). See result.warnings."
    )


def test_a_run_whose_orders_were_all_skipped_says_why_once():
    (notice,) = notices(daily, BuysWhileFlat, capital=500)
    assert str(notice.message) == (
        f"3 orders were skipped because {ROUNDED} (skip reason 'rounded_to_zero'). See "
        "result.warnings."
    )


def test_rejected_and_skipped_orders_are_counted_together_by_the_most_common_reason():
    (notice,) = notices(daily, TooMuchThenDefault, capital=500)
    assert str(notice.message) == (
        f"1 of 1 order was rejected and 2 more were skipped, 2 of them because {ROUNDED} "
        "(skip reason 'rounded_to_zero', the most common reason). See result.warnings."
    )


def test_reasons_with_as_many_orders_are_told_by_the_one_decided_first():
    # The rejection (first close) is decided before the skip (second close).
    (notice,) = notices(daily, TooMuchThenDefaultOnce, capital=500)
    assert str(notice.message) == (
        f"1 of 1 order was rejected and 1 more was skipped, 1 of them because {MARGIN} "
        "(reject_code 'margin', the most common reason). See result.warnings."
    )


def test_a_skip_decided_first_is_told_before_a_rejection_as_common():
    # The skip (first close) is decided before the rejection (second close).
    (notice,) = notices(daily, DefaultThenTooMuch, capital=500)
    assert str(notice.message) == (
        f"1 of 1 order was rejected and 1 more was skipped, 1 of them because {ROUNDED} "
        "(skip reason 'rounded_to_zero', the most common reason). See result.warnings."
    )


def test_orders_outside_the_entry_window_give_no_notice():
    # 10:00 and 10:15 are after the 09:50 cut-off: refused as the window intends.
    assert notices(intraday, BuysFourTimes, entry_until="09:50") == []


def test_the_notice_counts_only_what_the_run_did_not_ask_for():
    # 09:30: margin; 09:45, 10:00 and 10:15: after the 09:40 cut-off, not counted, so
    # the notice gives no total of orders (the other three were not placed either).
    (notice,) = notices(intraday, BuysTooMuchThenLate, entry_until="09:40")
    assert str(notice.message) == (
        f"1 order was rejected because {MARGIN} (reject_code 'margin'). See result.warnings."
    )


def test_an_order_a_size_cap_skipped_gives_no_notice():
    assert notices(intraday, CappedWithoutStop) == []


def test_the_notice_points_at_the_line_that_ran_the_backtest():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bt.backtest(SellsToOpen, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)
    (notice,) = [item for item in caught if item.category is OrdersNotPlacedWarning]
    assert notice.filename == __file__


def test_the_notice_is_a_user_warning_that_its_class_silences():
    assert issubclass(OrdersNotPlacedWarning, UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", OrdersNotPlacedWarning)
        daily(SellsToOpen)
        daily(BuysWhileFlat, capital=500)
    assert [item for item in caught if item.category is OrdersNotPlacedWarning] == []


def test_a_run_whose_orders_were_all_placed_gives_no_notice():
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
    assert "OrdersNotPlacedWarning" not in capfd.readouterr().err
