"""Rejected orders are reported: one warning per reject code that occurred, with how many
of the run's orders it refused, when the first was decided (IST), what the code means and
what to do; and a line in `print(result)`.

The runs use the bars of `accounting_runs`: one NSE session on 2026-11-04 in 1-minute
bars (a 15-minute strategy decides at 09:30, 09:45, ...), and the daily bars of
2026-11-02 to 2026-11-04 (a daily strategy decides at each 15:30 close). Capital is
10,00,000 and every price is near 1,000.00.
"""

from accounting_runs import CAPITAL, daily_bars, round_trip, session_bars

import openalgo.backtest as bt
from openalgo.backtest._reject_reasons import PRODUCT_RULE_CASES, REASONS

# Every code a kernel check can reject an order with (`RejectCode` in
# rust/oa_bt/src/exec/checks.rs).
KERNEL_REJECT_CODES = {
    "quantity_rule",
    "tick",
    "freeze_qty",
    "product_rule",
    "margin",
    "entry_window",
    "market_closed",
    "daily_loss_gate",
    "cooldown",
    "max_positions",
    "halted",
    "rate_limit",
    "instrument_unknown",
    "reverse_not_allowed",
    "scale_in_limit",
}
MARGIN_ADVICE = (
    "raise capital, lower the size, or set size_caps, for example size_caps = "
    "bt.size.caps(leverage=1) for a position of at most 1x the equity"
)
DELIVERY_SHORT = (
    "a delivery (CNC) short is not allowed in Indian cash markets; short a stock intraday "
    "with MIS, or trade futures"
)


def rejection_warnings(result: bt.Result) -> list[str]:
    return [text for text in result.warnings if "rejected with reject_code" in text]


class BuysTwice(bt.Strategy):
    """Buys 5,000 shares (50,00,000 at 1,000.00) at its first two bar closes."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) <= 2:
            self.buy(qty=5_000)


class BuysFourTimes(bt.Strategy):
    """Buys 10 shares at each of its first four bar closes (09:30 to 10:15)."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"

    def on_bar(self, bar):
        if len(self.close) <= 4:
            self.buy(qty=10)


class SellsToOpen(bt.Strategy):
    """A daily stock strategy that sells 10 shares at every close while flat."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.sell(qty=10)


def intraday(strategy: type, **settings) -> bt.Result:
    arguments = {"data": session_bars(), "capital": CAPITAL, "base": "1m", "product": "MIS"}
    return bt.backtest(strategy, costs=None, slippage=0, **arguments, **settings)


def test_every_kernel_reject_code_has_a_meaning_and_what_to_do():
    assert set(REASONS) | {"product_rule"} == KERNEL_REJECT_CODES
    assert set(PRODUCT_RULE_CASES) == {"delivery_short", "gtt_on_mis", "index"}


def test_an_order_rejected_for_margin_is_a_warning_with_what_to_do():
    # 5,000 shares at 1,000.00 on delivery need 50,00,000 against 10,00,000 of cash.
    result = round_trip({"qty": 5_000}, costs=None, slippage=0, product="CNC")
    assert rejection_warnings(result) == [
        "1 of 1 order was rejected with reject_code 'margin' (result.orders), decided at "
        "2026-11-04 09:30 IST: the margin the order needed was more than the free cash; "
        + MARGIN_ADVICE
    ]


def test_each_reject_code_gets_one_warning_in_the_order_it_first_occurred():
    # 09:30: 5,000 shares on MIS at 1x need 50,00,000 (margin). 09:45: after the
    # 09:40 entry cut-off (entry_window).
    result = intraday(BuysTwice, entry_until="09:40")
    texts = rejection_warnings(result)
    assert [text.split("'")[1] for text in texts] == ["margin", "entry_window"]
    assert texts[1].startswith(
        "1 of 2 orders was rejected with reject_code 'entry_window' (result.orders), "
        "decided at 2026-11-04 09:45 IST"
    )


def test_many_entry_window_rejections_are_one_line():
    result = intraday(BuysFourTimes, entry_until="09:20")
    assert rejection_warnings(result) == [
        "4 of 4 orders were rejected with reject_code 'entry_window' (result.orders), the "
        "first decided at 2026-11-04 09:30 IST: orders after the MIS entry cut-off "
        "(entry_until) are rejected, and so are entries before entry_from; check self.now "
        "or set entry_until"
    ]


def test_a_daily_stock_strategy_that_sells_to_open_is_told_delivery_cannot_be_shorted():
    result = bt.backtest(SellsToOpen, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)
    assert result.trades.empty
    assert rejection_warnings(result) == [
        "3 of 3 orders were rejected with reject_code 'product_rule' (result.orders), the "
        "first decided at 2026-11-02 15:30 IST: " + DELIVERY_SHORT
    ]


def test_a_short_condition_on_delivery_is_told_delivery_cannot_be_shorted():
    shorts = bt.from_arrays(
        None, None, shorts=[True, True, True], symbol="SBIN", exchange="NSE", interval="D",
        qty=10,
    )  # fmt: skip
    result = bt.backtest(shorts, data=daily_bars(), capital=CAPITAL, costs=None, slippage=0)
    (text,) = rejection_warnings(result)
    assert text.endswith(": " + DELIVERY_SHORT)


def test_a_gtt_on_mis_is_told_to_use_cnc_or_nrml():
    result = round_trip(
        {"qty": 10, "price": 990.0, "validity": "gtt"}, costs=None, slippage=0, product="MIS"
    )
    assert rejection_warnings(result) == [
        "1 of 1 order was rejected with reject_code 'product_rule' (result.orders), decided "
        "at 2026-11-04 09:30 IST: MIS cannot hold a GTT order; place a GTT on CNC or NRML"
    ]


def test_print_shows_how_many_orders_were_rejected_by_code():
    result = intraday(BuysTwice, entry_until="09:40")
    assert "\n  rejected orders: 2 of 2 (margin 1, entry_window 1)\n" in repr(result)


def test_a_run_without_rejections_has_no_rejection_warning_or_line():
    result = round_trip({"qty": 10}, costs=None, slippage=0)
    assert rejection_warnings(result) == []
    assert "rejected orders" not in repr(result)
