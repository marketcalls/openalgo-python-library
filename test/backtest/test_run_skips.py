"""Orders skipped before they were placed are reported: one line in `result.warnings` per
skip reason that occurred, with how many orders it skipped, when the first was decided
(IST), what happened and what to do. A `target()` or `rebalance()` target that rounds
down to no shares while nothing is held is a skipped order too.

The runs use the bars of `accounting_runs`, with no charges and no slippage: the daily
bars of 2026-11-02 to 2026-11-04 (a daily strategy decides at each 15:30 close, at
1,000.00) and one NSE session on 2026-11-04 in 1-minute bars (a 15-minute strategy
decides first at 09:30, at 1,000.00).
"""

import pytest
from accounting_runs import CAPITAL, daily_bars, round_trip, session_bars

import openalgo.backtest as bt
from openalgo.backtest._skip_reasons import SKIP_REASONS

# Every reason the kernel skips a sized order with (`SkipReason::code` in
# rust/oa_bt/src/sizing/mod.rs).
KERNEL_SKIP_REASONS = {"rounded_to_zero", "cap", "margin"}
ROUNDED_TO_ZERO = (
    "the sized quantity was less than one share or, for a derivative, one lot; give the "
    "strategy more capital, size with a larger percent, trade a cheaper share, or pass "
    'quantity_mode="fractional" for part shares (a sizing function that returns 0 skips '
    "its order on purpose)"
)


def skip_warnings(result: bt.Result) -> list[str]:
    return [text for text in result.warnings if "skipped with skip reason" in text]


class BuysWhileFlat(bt.Strategy):
    """A daily stock strategy that buys with the engine default at every close while
    flat."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy()


def daily(strategy: type, capital: float = CAPITAL) -> bt.Result:
    return bt.backtest(strategy, data=daily_bars(), capital=capital, costs=None, slippage=0)


def test_every_kernel_skip_reason_has_a_meaning_and_what_to_do():
    assert set(SKIP_REASONS) == KERNEL_SKIP_REASONS
    assert all(len(meaning) == 2 and all(meaning) for meaning in SKIP_REASONS.values())


def test_orders_whose_size_rounds_to_zero_are_one_warning_with_what_to_do():
    # 500 rupees less the 2 percent cash buffer is 490 at 1,000.00: 0.49 shares, so 0,
    # at each of the three closes.
    result = daily(BuysWhileFlat, capital=500)
    assert result.orders.empty and len(result.skipped) == 3
    assert skip_warnings(result) == [
        "3 orders were skipped with skip reason 'rounded_to_zero' (result.skipped), the "
        "first decided at 2026-11-02 15:30 IST: " + ROUNDED_TO_ZERO
    ]


def test_skips_are_the_first_warnings_of_the_result():
    result = daily(BuysWhileFlat, capital=500)
    assert result.warnings[0] == skip_warnings(result)[0]


def test_an_order_a_size_cap_skipped_is_a_warning_with_what_to_do():
    class CappedWithoutStop(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"
        size_caps = bt.size.caps(risk=bt.pct(0.5))

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.buy(size=bt.size.percent(50))

    result = bt.backtest(
        CappedWithoutStop, data=session_bars(), capital=CAPITAL, base="1m", costs=None, slippage=0
    )
    assert skip_warnings(result) == [
        "1 order was skipped with skip reason 'cap' (result.skipped), decided at "
        "2026-11-04 09:30 IST: a size cap (size_caps) left no room for the order, or a risk "
        "cap met an order with no stop; loosen size_caps, or give the order a stop so a "
        "risk cap can size it"
    ]


def test_print_shows_how_many_orders_were_skipped_by_reason():
    result = daily(BuysWhileFlat, capital=500)
    assert "\n  skipped orders: 3 (rounded_to_zero 3)\n" in repr(result)


def test_a_run_without_skips_has_no_skip_line():
    result = round_trip({"qty": 10}, costs=None, slippage=0)
    assert skip_warnings(result) == []
    assert "skipped orders" not in repr(result)


def targets(amounts: dict) -> type:
    """A daily stock strategy that calls `target(**amounts)` at its first close and
    keeps what it returned."""

    class Targets(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "D"

        def init(self):
            self.returned = []

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.returned.append(self.target(**amounts))

    return Targets


@pytest.mark.parametrize("amounts", [{"value": 500}, {"percent": 0.05}])
def test_a_target_that_rounds_to_no_shares_while_flat_is_a_skipped_order(amounts):
    # 500 rupees, or 0.05 percent of 10,00,000, at 1,000.00 is half a share: 0.
    result = daily(targets(amounts))
    assert result.orders.empty
    assert list(result.skipped["reason"]) == ["rounded_to_zero"]
    assert list(result.skipped["decided_at"].astype(str)) == ["2026-11-02 15:30:00+05:30"]


def test_a_target_of_flat_while_flat_is_not_a_skip():
    result = daily(targets({"value": 0}))
    assert result.orders.empty and result.skipped.empty


def test_a_rebalance_weight_that_rounds_to_no_shares_is_a_skipped_order():
    # 0.01 percent of 10,00,000 less the 2 percent cash buffer is 98 rupees at 1,000.00:
    # 0.098 shares, so 0.
    class Tiny(bt.Strategy):
        symbols, exchange, interval = ("SBIN",), "NSE", "D"

        def on_bar(self, bars):
            if len(self.data["SBIN"].close) == 1:
                self.rebalance({"SBIN": 0.01})

    result = bt.backtest(Tiny, data={"SBIN": daily_bars()}, capital=CAPITAL, costs=None, slippage=0)
    assert result.orders.empty
    assert list(result.skipped["symbol"]) == ["SBIN"]
    assert list(result.skipped["reason"]) == ["rounded_to_zero"]
