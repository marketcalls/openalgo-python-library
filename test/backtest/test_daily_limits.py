"""The daily limits of a backtest (18-multi-strategy 2.5a): the trade caps, the daily
profit targets and the order value cap, for one strategy (`bt.backtest(Strategy, ...)`)
and for the account of a list of strategies.

The runs are 1-minute SBIN strategies on in-process bars of the NSE session of 2025-07-01
(`portfolio_samples`), no charges and no slippage; nothing contacts a server. A strategy
acts on a plan by bar number: bar n opens at 09:14 + n IST, its decision is at its close
(09:15 + n) and fills at the open of the next bar. Every expected figure is worked by
hand from the plan and the bars in the comment above it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pytest
from portfolio_samples import flat_bars, no_costs, stepped_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError, OrdersNotPlacedWarning

CAPITAL = 1_000_000
#: The settings of the daily limits, which a run without them never names.
DAILY_LIMITS = ("max_trades_per_day", "daily_profit_target", "max_order_value")


class Planned(bt.Strategy):
    """Acts on `plan()`: by bar number, the order method to call and its arguments."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def plan(self) -> dict[int, tuple[str, dict[str, Any]]]:
        return {}

    def on_bar(self, bar):
        number = self.state.get("bars", 0) + 1
        self.state["bars"] = number
        action = self.plan().get(number)
        if action is not None:
            method, arguments = action
            getattr(self, method)(**arguments)


class InAndOutTwice(Planned):
    """Buys 10 at bar 1, exits at bar 3, buys 10 again at bar 5."""

    def plan(self):
        return {1: ("buy", {"qty": 10}), 3: ("exit", {}), 5: ("buy", {"qty": 10})}


class CappedInClass(InAndOutTwice):
    max_trades_per_day = 2


class Flip(Planned):
    """Buys 10 at bar 1 and sells 20 with reverse=True at bar 3."""

    def plan(self):
        return {1: ("buy", {"qty": 10}), 3: ("sell", {"qty": 20, "reverse": True})}


class BuyAndRetry(Planned):
    """Buys 100 at bar 1 and 10 more at bar 8."""

    def plan(self):
        return {1: ("buy", {"qty": 100}), 8: ("buy", {"qty": 10})}


def run(strategy: type[bt.Strategy], bars: pd.DataFrame, **settings: Any) -> bt.Result:
    with warnings.catch_warnings():
        warnings.simplefilter("error", OrdersNotPlacedWarning)
        return bt.backtest(strategy, data=bars, capital=CAPITAL, **no_costs(**settings))


def codes(result: bt.Result) -> list:
    """Each order's reject code, in id order, None for an order not rejected."""
    return [None if pd.isna(code) else code for code in result.orders["reject_code"]]


def test_the_trade_cap_refuses_the_entry_past_it_and_lets_the_exit_go():
    # Cap 1: the buy of bar 1 is the day's entry; the exit of bar 3 goes; the buy of bar 5
    # (decided at 09:20) is refused trade_limit. No OrdersNotPlacedWarning: the cap works
    # as asked.
    result = run(InAndOutTwice, flat_bars(100.0), max_trades_per_day=1)
    assert codes(result) == [None, None, "trade_limit"]
    lines = [line for line in result.warnings if "'trade_limit'" in line]
    assert lines == [
        "1 of 3 orders was rejected with reject_code 'trade_limit' (result.orders), decided "
        "at 2025-07-01 09:20 IST: an entry came once the strategy's entries of the day "
        "(those that filled and those still working) had reached max_trades_per_day; a "
        "reversal at the cap was cut to its close, its opening part refused; this is the cap "
        "working: exits still go; raise max_trades_per_day if it is too tight"
    ]


def test_a_class_attribute_cap_wins_over_the_run_argument():
    result = run(CappedInClass, flat_bars(100.0), max_trades_per_day=1)
    assert codes(result) == [None, None, None]


def test_a_reversal_at_the_cap_closes_the_position_and_refuses_its_opening_part():
    # Cap 1: long 10 from 09:16; the sale of 20 decided at 09:18 sells 10 at the 09:18
    # open and the opening 10 is refused trade_limit, as its own order.
    result = run(Flip, flat_bars(100.0), max_trades_per_day=1)
    orders = result.orders[["side", "quantity_units", "status"]]
    assert orders.to_dict("records") == [
        {"side": "BUY", "quantity_units": 10.0, "status": "Filled"},
        {"side": "SELL", "quantity_units": 10.0, "status": "Filled"},
        {"side": "SELL", "quantity_units": 10.0, "status": "Rejected"},
    ]
    assert codes(result) == [None, None, "trade_limit"]
    assert result.trades["exit_reason"].tolist() == ["signal"]
    assert result.final_positions.empty or (result.final_positions["quantity_units"] == 0).all()


def test_the_daily_profit_target_exits_the_position_and_refuses_later_entries():
    # Long 100 bought at 100.00 at 09:16; from the bar opening 09:20 the price is 110.00:
    # at its close the day's measure is 100 x 10 = 1,000.00, the target. The long exits
    # at the 09:21 open at 110.00 (daily_profit_target) and the buy of bar 8 (09:23) is
    # refused daily_profit_gate.
    result = run(BuyAndRetry, stepped_bars(100.0, 110.0, 5), daily_profit_target=1_000)
    trades = result.trades
    assert trades["exit_reason"].tolist() == ["daily_profit_target"]
    assert trades["exit_price_rupees"].tolist() == [110.0]
    assert codes(result) == [None, "daily_profit_gate"]
    hit = result.risk_events[result.risk_events["name"] == "daily_profit_hit"]
    assert hit[["scope", "level_rupees", "observed_rupees"]].to_dict("records") == [
        {"scope": "session", "level_rupees": 1_000.0, "observed_rupees": 1_000.0}
    ]


def test_the_order_value_cap_refuses_an_entry_above_it_and_never_an_exit():
    # Cap 999.00: 10 units at the last close of 100.00 are 1,000.00, refused.
    result = run(InAndOutTwice, flat_bars(100.0), max_order_value=999)
    assert codes(result)[0] == "order_value_limit"
    # Cap 1,000.00: the entry passes; at 150.00 the exit is worth 1,500.00 and still goes.
    rising = stepped_bars(100.0, 150.0, 2)
    passed = run(InAndOutTwice, rising, max_order_value=1_000)
    assert codes(passed)[:2] == [None, None]
    assert passed.trades["exit_price_rupees"].tolist()[0] == 150.0


def test_the_assumptions_name_the_daily_limits_only_when_set():
    plain = run(InAndOutTwice, flat_bars(100.0))
    assert not set(DAILY_LIMITS) & set(plain.assumptions)
    capped = run(
        InAndOutTwice,
        flat_bars(100.0),
        max_trades_per_day=3,
        daily_profit_target=bt.rupees(5_000),
        max_order_value=200_000,
    )
    named = {name: capped.assumptions[name] for name in DAILY_LIMITS}
    assert named == {
        "max_trades_per_day": 3,
        "daily_profit_target": 5_000.0,
        "max_order_value": 200_000.0,
    }


@pytest.mark.parametrize(
    ("setting", "value", "words"),
    [
        ("max_trades_per_day", 0, "max_trades_per_day takes a positive whole number"),
        ("max_trades_per_day", 2.5, "max_trades_per_day takes a positive whole number"),
        ("daily_profit_target", -1, "daily_profit_target takes a positive amount of rupees"),
        ("max_order_value", "big", "max_order_value takes a positive amount of rupees"),
        ("account_max_trades_per_day", 2, "set the account of a list of strategies"),
    ],
)
def test_unusable_daily_limits_are_refused_before_any_data_is_loaded(setting, value, words):
    with pytest.raises(ConfigError, match=words):
        bt.backtest(InAndOutTwice, data=flat_bars(100.0), capital=CAPITAL, **{setting: value})


class First(Planned):
    def plan(self):
        return {1: ("buy", {"qty": 100}), 3: ("exit", {}), 5: ("buy", {"qty": 100})}


class Second(Planned):
    def plan(self):
        return {1: ("buy", {"qty": 100})}


def test_the_accounts_trade_cap_refuses_the_entry_past_it_across_strategies():
    # A cap of 2 entries for the account: First and Second each enter at 09:16 (2 in all);
    # First's buy decided at 09:20 is refused account_trade_limit. Second keeps its long.
    pf = bt.backtest(
        [First, Second],
        data=flat_bars(100.0),
        capital=CAPITAL,
        account_max_trades_per_day=2,
        **no_costs(),
    )
    refused = pf.refused[["strategy", "order_id", "reason"]].to_dict("records")
    assert refused == [{"strategy": "First", "order_id": 3, "reason": "account_trade_limit"}]
    assert pf.assumptions["account_max_trades_per_day"] == 2
    assert (
        pf.strategy("First")
        .assumptions["account"]
        .endswith("; the account's cap of 2 entries a day refused 1 entry")
    )
    assert pf.strategy("Second").trades["entry_units"].tolist() == [100.0]


class Long(Planned):
    def plan(self):
        return {1: ("buy", {"qty": 100}), 8: ("buy", {"qty": 10})}


class AlsoLong(Planned):
    def plan(self):
        return {1: ("buy", {"qty": 100})}


def test_the_accounts_profit_target_exits_every_strategy_and_locks_their_entries():
    # Both long 100 from 09:16 at 100.00; from the bar opening 09:20 the price is 105.00:
    # at its close each is up 500.00, the account 1,000.00, its target. Both exit at the
    # 09:21 open at 105.00 (account_profit_target); Long's buy at 09:23 is refused
    # account_profit_gate.
    pf = bt.backtest(
        [Long, AlsoLong],
        data=stepped_bars(100.0, 105.0, 5),
        capital=CAPITAL,
        account_daily_profit_target=1_000,
        **no_costs(),
    )
    for name in ("Long", "AlsoLong"):
        trades = pf.strategy(name).trades
        assert trades["exit_reason"].tolist() == ["account_profit_target"]
        assert trades["exit_price_rupees"].tolist() == [105.0]
    refused = pf.refused[["strategy", "order_id", "reason"]].to_dict("records")
    assert refused == [{"strategy": "Long", "order_id": 2, "reason": "account_profit_gate"}]
    names = pf.risk_events["name"].tolist()
    assert names.count("account_profit_hit") == 2


def test_none_for_a_strategy_of_a_list_leaves_it_without_the_cap():
    # First and InAndOutTwice each enter at bar 1, exit at bar 3 and enter again at bar 5
    # (decided at 09:20). First, given None, has no cap: its second entry goes. The cap of 1
    # refuses InAndOutTwice's second entry trade_limit.
    pf = bt.backtest(
        [First, InAndOutTwice],
        data=flat_bars(100.0),
        capital=CAPITAL,
        max_trades_per_day={"First": None, "InAndOutTwice": 1},
        **no_costs(),
    )
    assert codes(pf.strategy("First")) == [None, None, None]
    assert codes(pf.strategy("InAndOutTwice")) == [None, None, "trade_limit"]
    assert "max_trades_per_day" not in pf.strategy("First").assumptions
    assert pf.strategy("InAndOutTwice").assumptions["max_trades_per_day"] == 1


def test_each_strategy_of_a_list_takes_its_own_cap_by_name():
    pf = bt.backtest(
        [First, Second],
        data=flat_bars(100.0),
        capital=CAPITAL,
        max_trades_per_day={"First": 1, "Second": 5},
        **no_costs(),
    )
    assert codes(pf.strategy("First")) == [None, None, "trade_limit"]
    assert pf.strategy("First").assumptions["max_trades_per_day"] == 1
    assert pf.strategy("Second").assumptions["max_trades_per_day"] == 5
