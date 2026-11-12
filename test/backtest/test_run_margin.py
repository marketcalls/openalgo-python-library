"""Margin in a backtest (04-execution-and-market-model 5): the dated default table,
checked at the fill bar's open; intraday leverage only when the order asks for it.

Capital is 10,00,000 and entries fill at 1,000.00 (see accounting_runs). Each table
row's figure is tested in the kernel (`market::margin`); these tests check the run.
The futures margin (the default of decision D23 and `margin=`) is tested in
`test_run_futures_margin`.
"""

import pytest
from accounting_runs import at, round_trip

from openalgo.backtest.errors import ConfigError


def margin_after_entry(result):
    """Margin blocked at the close of the entry slice (09:30 to 09:31)."""
    return result.equity.loc[at(9, 31), "margin_rupees"]


def test_a_delivery_entry_beyond_the_free_cash_is_rejected_at_the_fill():
    # 1,001 shares at 1,000.00 need 10,01,000.
    result = round_trip({"qty": 1_001}, costs=None, slippage=0, product="CNC")
    assert result.fills.empty
    assert list(result.orders["reject_code"]) == ["margin"]


def test_an_unleveraged_intraday_entry_blocks_its_value():
    result = round_trip({"qty": 400}, costs=None, slippage=0)
    assert margin_after_entry(result) == 400_000.0


def test_intraday_leverage_asked_for_blocks_the_value_over_the_leverage():
    # 4,000 shares at 1,000.00 are 40,00,000; at 5x they block 8,00,000.
    result = round_trip({"qty": 4_000, "leverage": 5}, costs=None, slippage=0)
    assert margin_after_entry(result) == 800_000.0
    assert len(result.fills) == 2


def test_leverage_above_the_tables_limit_is_refused():
    with pytest.raises(ConfigError, match="5"):
        round_trip({"qty": 100, "leverage": 6}, costs=None, slippage=0)


def test_leverage_on_delivery_is_refused():
    with pytest.raises(ConfigError, match="leverage"):
        round_trip({"qty": 100, "leverage": 2}, costs=None, slippage=0, product="CNC")
