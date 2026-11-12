"""`self.target(...)`: trade to a target position (12-position-sizing 5, FR-SIZE-08).

Bars are the rising 1-minute bars of `strategy_samples` on 2025-07-01: bar k opens at
500.00 + 0.10 k and closes 0.05 above its open, so a decision at the close of bar 0 is
taken at 500.05 and fills at the open of bar 1, 500.10. No charges, no slippage, capital
10,00,000. The strategies act on the number of bars seen (`len(self.close)`).
"""

import pytest
from accounting_runs import NIFTY_FUTURE, session_bars
from strategy_samples import minute_bars, run

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError


def scripted(plan: dict, product: str = "CNC") -> type:
    """A strategy that calls `target(**plan[n])` after its n-th bar and keeps what each
    call returned."""

    class Scripted(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.returned = []

        def on_bar(self, bar):
            for arguments in plan.get(len(self.close), ()):
                self.returned.append(self.target(**arguments))

    Scripted.product = product
    return Scripted


def traded(result) -> list[tuple[str, float, str]]:
    """(side, units, reason) of the fills of the strategy's orders (not the engine's
    square-off of an MIS position)."""
    fills = result.fills[result.fills["order_id"].notna()]
    return list(zip(fills["side"], fills["quantity_units"], fills["reason"], strict=True))


def test_a_target_from_flat_buys_the_target():
    result = run(scripted({1: [{"qty": 50}]}))
    assert traded(result) == [("BUY", 50.0, "entry")]


def test_a_target_resizes_up_then_down_then_to_flat():
    plan = {1: [{"qty": 50}], 3: [{"qty": 80}], 5: [{"qty": 20}], 7: [{"qty": 0}]}
    result = run(scripted(plan))
    assert traded(result) == [
        ("BUY", 50.0, "entry"),
        ("BUY", 30.0, "add"),
        ("SELL", 60.0, "signal"),
        ("SELL", 20.0, "signal"),
    ]


def test_a_target_on_the_other_side_closes_and_reverses_in_one_order():
    result = run(scripted({1: [{"qty": 50}], 3: [{"qty": -30}]}, product="MIS"))
    assert traded(result) == [
        ("BUY", 50.0, "entry"),
        ("SELL", 50.0, "signal"),
        ("SELL", 30.0, "entry"),
    ]
    assert len(result.orders) == 2


def test_a_target_value_is_rounded_down_to_whole_shares_at_the_decision_price():
    # 10,000 rupees at the decision price 500.05 is 19.998 shares: 19.
    result = run(scripted({1: [{"value": 10_000}]}), quantity_mode="venue")
    assert traded(result) == [("BUY", 19.0, "entry")]


def test_a_target_percent_is_of_the_equity_at_the_decision_price():
    # 10 percent of 10,00,000 is 1,00,000 at 500.05: 199.98 shares, so 199.
    result = run(scripted({1: [{"percent": 10}]}), quantity_mode="venue")
    assert traded(result) == [("BUY", 199.0, "entry")]


def test_a_negative_target_value_opens_a_short():
    # -10,000 rupees at 500.05: a short of 19 shares (MIS).
    result = run(scripted({1: [{"value": -10_000}]}, product="MIS"), quantity_mode="venue")
    assert traded(result) == [("SELL", 19.0, "entry")]


def test_a_target_in_lots_trades_whole_lots():
    # Two lots of 75 units of the NIFTY future.
    class Lots(bt.Strategy):
        symbol, exchange, interval, product = NIFTY_FUTURE.symbol, "NFO", "15m", "NRML"

        def on_bar(self, bar):
            if len(self.close) == 1:
                self.target(lots=2)

    data = session_bars(minutes=385)
    result = run(Lots, data, instrument=NIFTY_FUTURE, base="1m")
    assert traded(result)[0] == ("BUY", 150.0, "entry")


def test_a_target_already_on_its_way_sends_no_second_order():
    strategy = scripted({1: [{"qty": 50}, {"qty": 50}, {"qty": 70}]})
    result = run(strategy)
    assert traded(result) == [("BUY", 50.0, "entry"), ("BUY", 20.0, "add")]
    assert len(result.orders) == 2


def test_a_target_returns_none_when_the_position_is_the_target():
    returned = []

    class Holds(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

        def on_bar(self, bar):
            if len(self.close) in (1, 3):
                returned.append(self.target(qty=50))

    run(Holds)
    assert returned[0].status in ("Submitted", "Filled")
    assert returned[1] is None


@pytest.mark.parametrize(
    "arguments",
    [{}, {"qty": 10, "value": 1_000}, {"lots": 1, "percent": 5}],
)
def test_a_target_takes_exactly_one_amount(arguments):
    with pytest.raises(ConfigError, match="exactly one of"):
        run(scripted({1: [arguments]}))


@pytest.mark.parametrize(
    "arguments",
    [{"qty": "10"}, {"qty": True}, {"lots": 1.5}, {"value": float("nan")}],
)
def test_a_target_amount_must_be_a_number(arguments):
    with pytest.raises(ConfigError):
        run(scripted({1: [arguments]}))


def test_a_target_leaves_a_resting_order_working():
    # A buy limit far below the rising market never fills; the target buys 50 anyway
    # and the limit order is still working after it.
    working = []

    class WithLimit(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

        def on_bar(self, bar):
            if len(self.close) == 1:
                limit = self.buy(qty=10, price=400.0)
                self.target(qty=50)
                working.append(limit.id in [order.id for order in self.orders])

    result = run(WithLimit, minute_bars(count=30))
    assert working == [True]
    assert traded(result) == [("BUY", 50.0, "entry")]
