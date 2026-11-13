"""`self.rebalance(weights)` (11-portfolio-rebalancing 2.1, the subset built): weights in
percent of equity by symbol, sales before buys, a cash buffer, buys that are intended
adds, every order tagged "rebalance".

Unless a test says otherwise: NSE shares registered in the order given, daily bars from
2026-11-02 that open and close at the prices given (high and low 10.00 either side),
capital 10,00,000, no charges, no slippage, CNC. A rebalance decided at a close fills at
the next session's open. Expected values are worked by hand in each test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CAPITAL = 1_000_000


def sessions(start: str, end: str) -> list[dt.date]:
    """The NSE sessions of the built-in calendar from `start` to `end`."""
    return list(bt.calendar.default_calendar().sessions("NSE", start, end)["date"])


DAYS = sessions("2026-11-02", "2026-11-06")


def daily(opens: list[float], closes: list[float], days: list[dt.date] = DAYS) -> pd.DataFrame:
    """One bar per session, stamped 09:15 IST."""
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days[: len(opens)]],
        name="timestamp",
    )
    opens_, closes_ = np.array(opens, dtype=float), np.array(closes, dtype=float)
    columns = {
        "open": opens_,
        "high": np.maximum(opens_, closes_) + 10.0,
        "low": np.minimum(opens_, closes_) - 10.0,
        "close": closes_,
        "volume": np.full(len(opens_), 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def flat(sessions_count: int = 4, price: float = 1_000.0) -> pd.DataFrame:
    return daily([price] * sessions_count, [price] * sessions_count)


def scripted(symbols: tuple[str, ...], plan: dict[int, list[dict]]) -> type[bt.Strategy]:
    """A CNC strategy that, after its n-th bar, calls `self.rebalance(**call)` for each
    call of `plan[n]` and keeps what each returned."""

    class Scripted(bt.Strategy):
        exchange, interval, product = "NSE", "D", "CNC"

        def init(self):
            self.returned = []

        def on_bar(self, bars):
            for call in plan.get(len(self.data[self.symbols[0]].close), ()):
                self.returned.append(self.rebalance(**call))

    Scripted.symbols = symbols
    return Scripted


def run(strategy: type[bt.Strategy], data: dict[str, pd.DataFrame], **settings) -> bt.Result:
    arguments = {"data": data, "capital": CAPITAL, "costs": None, "slippage": 0, **settings}
    return bt.backtest(strategy, **arguments)


def fills(result: bt.Result) -> list[tuple]:
    """(session date, symbol, side, units, reason) of every fill, in order."""
    frame = result.fills
    return list(
        zip(
            frame["time"].dt.date,
            frame["symbol"],
            frame["side"],
            frame["quantity_units"],
            frame["reason"],
            strict=True,
        )
    )


def test_every_sale_fills_before_every_buy_whatever_the_registration_order():
    # BBB is registered first. All 10,00,000 go into 1,000 AAA at 1,000.00; then the
    # weights move everything to BBB: AAA (weight left out) is sold whole and 1,000 BBB
    # are bought with its proceeds at the same open, after the sale.
    plan = {
        1: [{"weights": {"AAA": 100}, "cash_buffer": 0}],
        2: [{"weights": {"BBB": 100}, "cash_buffer": 0}],
    }
    result = run(scripted(("BBB", "AAA"), plan), {"BBB": flat(), "AAA": flat()})
    assert fills(result) == [
        (DAYS[1], "AAA", "BUY", 1_000.0, "entry"),
        (DAYS[2], "AAA", "SELL", 1_000.0, "signal"),
        (DAYS[2], "BBB", "BUY", 1_000.0, "entry"),
    ]
    assert result.orders["reject_code"].isna().all()


def averaging_down_run(method: str) -> bt.Result:
    """AAA and BBB at 49.5 percent each (495 units at 1,000.00, 10,000 of cash left);
    AAA then closes at 800.00 and BBB at 1,200.00, and the portfolio is put back to
    49.5 percent each with `method`: equity 10,000 + 3,96,000 + 5,94,000 = 10,00,000, so
    4,95,000 each: 618.75 AAA (buy 123.75 for 99,000, below AAA's average of 1,000.00)
    and 412.5 BBB (sell 82.5 for 99,000)."""
    data = {
        "AAA": daily([1_000, 1_000, 800, 800], [1_000, 800, 800, 800]),
        "BBB": daily([1_000, 1_000, 1_200, 1_200], [1_000, 1_200, 1_200, 1_200]),
    }

    class TwoStep(bt.Strategy):
        symbols, exchange, interval, product = ("AAA", "BBB"), "NSE", "D", "CNC"

        def on_bar(self, bars):
            bar = len(self.data["AAA"].close)
            if bar == 1 or (bar == 2 and method == "rebalance"):
                self.rebalance({"AAA": 50, "BBB": 50}, cash_buffer=1)
            elif bar == 2:
                self.target(value=495_000, symbol="BBB")
                self.target(value=495_000, symbol="AAA")

    return run(TwoStep, data)


def test_a_rebalance_buy_below_the_average_price_is_an_intended_add():
    result = averaging_down_run("rebalance")
    assert fills(result)[2:] == [
        (DAYS[2], "BBB", "SELL", 82.5, "signal"),
        (DAYS[2], "AAA", "BUY", 123.75, "add"),
    ]


def test_a_plain_target_buy_below_the_average_price_is_refused_by_default():
    orders = averaging_down_run("target").orders
    aaa_add = orders[(orders["symbol"] == "AAA") & (orders["side"] == "BUY")].iloc[-1]
    assert aaa_add["reject_code"] == "scale_in_limit"


def test_every_order_of_a_rebalance_is_tagged_rebalance():
    orders = averaging_down_run("rebalance").orders
    assert list(orders["tag"]) == ["rebalance"] * 4


def test_the_cash_buffer_is_left_out_of_every_target():
    # 2 percent buffer: each of two 50 percent weights is 4,90,000, 490 units at 1,000.00.
    plan = {1: [{"weights": {"AAA": 50, "BBB": 50}, "cash_buffer": 2}]}
    result = run(scripted(("AAA", "BBB"), plan), {"AAA": flat(), "BBB": flat()})
    assert list(result.fills["quantity_units"]) == [490.0, 490.0]


def test_the_default_cash_buffer_is_two_percent():
    plan = {1: [{"weights": {"AAA": 100}}]}
    result = run(scripted(("AAA",), plan), {"AAA": flat()})
    assert list(result.fills["quantity_units"]) == [980.0]


def test_a_rebalance_returns_the_orders_it_sent_by_symbol():
    # The second call at the same decision finds both targets already on their way.
    weights = {"AAA": 50, "BBB": 50}
    strategy = scripted(("AAA", "BBB"), {1: [{"weights": weights}, {"weights": weights}]})
    returned = []
    strategy.on_stop = lambda self: returned.extend(self.returned)
    run(strategy, {"AAA": flat(), "BBB": flat()})
    assert sorted(returned[0]) == ["AAA", "BBB"]
    assert all(order.status == "Filled" for order in returned[0].values())
    assert returned[1] == {}


class RetargetedByATimer(bt.Strategy):
    """Puts everything in AAA at its first close, then a timer at the next open, before
    that buy has filled, moves AAA to 50 percent."""

    symbols, exchange, interval, product = ("AAA",), "NSE", "D", "CNC"

    def init(self):
        self.weight = None
        self.at("09:15", self.retarget)

    def on_bar(self, bars):
        if len(self.data["AAA"].close) == 1:
            self.rebalance({"AAA": 100}, cash_buffer=0)
            self.weight = 50

    def retarget(self):
        if self.weight is not None:
            self.rebalance({"AAA": self.weight}, cash_buffer=0)
            self.weight = None


def test_a_timer_rebalance_trades_from_the_buy_due_at_its_open():
    # The first close buys 1,000 AAA (10,00,000 at 1,000.00), due at the second open.
    # The timer at that open targets 5,00,000, 500 AAA: the buy fills first, then 500
    # are sold, both at that open.
    result = run(RetargetedByATimer, {"AAA": flat()})
    assert fills(result) == [
        (DAYS[1], "AAA", "BUY", 1_000.0, "entry"),
        (DAYS[1], "AAA", "SELL", 500.0, "signal"),
    ]
    assert result.final_positions["quantity_units"].tolist() == [500.0]


@pytest.mark.parametrize(
    "call, message",
    [
        ({"weights": [("AAA", 50)]}, "dict of percent"),
        ({"weights": {"CCC": 50}}, "not traded"),
        ({"weights": {"AAA": -5}}, "from 0 to 100"),
        ({"weights": {"AAA": 101}}, "from 0 to 100"),
        ({"weights": {"AAA": float("nan")}}, "from 0 to 100"),
        ({"weights": {"AAA": True}}, "from 0 to 100"),
        ({"weights": {"AAA": 60, "BBB": 50}}, "add up to 110"),
        ({"weights": {"AAA": 50}, "cash_buffer": 100}, "cash_buffer"),
        ({"weights": {"AAA": 50}, "cash_buffer": -1}, "cash_buffer"),
        ({"weights": {"AAA": 50}, "cash_buffer": "2"}, "cash_buffer"),
    ],
)
def test_unusable_weights_or_buffer_are_refused(call, message):
    with pytest.raises(ConfigError, match=message):
        run(scripted(("AAA", "BBB"), {1: [call]}), {"AAA": flat(), "BBB": flat()})


# ----- the whole flow: a four-stock portfolio rebalanced quarterly ----------------------

FOUR = ("RELIANCE", "HDFCBANK", "INFY", "TCS")
# (first open, daily drift, swing): two rise, two fall, all swing, so every quarter has
# holdings over and under their weight.
PATHS = {
    "RELIANCE": (1_300.0, 0.0006, 0.06),
    "HDFCBANK": (1_700.0, -0.0004, 0.08),
    "INFY": (1_900.0, 0.0003, 0.10),
    "TCS": (4_200.0, -0.0002, 0.05),
}
# NSE equity delivery (Zerodha brokerage and charges page; NSE): STT 0.1 percent of the
# value on buys and sells, stamp duty 0.015 percent on buys, no brokerage, and the DP
# charge of Rs 13 plus 18 percent GST, Rs 15.34, once per scrip on each day it is sold
# from holdings.
STT_RATE = 0.001
STAMP_RATE_BUY = 0.00015
DP_CHARGE = 15.34
TARGET_PERCENT = 25.0
BUFFER_PERCENT = 2.0


def walk(first_open: float, drift: float, swing: float, days: list[dt.date]) -> pd.DataFrame:
    """Closes on a drifting, swinging path; each session opens 0.1 percent above the
    previous close; prices on the 0.05 tick."""
    k = np.arange(len(days))
    closes = np.round(first_open * np.exp(drift * k + swing * np.sin(k / 23.0)) * 20) / 20
    opens = np.round(np.concatenate([[first_open], closes[:-1] * 1.001]) * 20) / 20
    return daily(list(opens), list(closes), days)


class QuarterlyEqualWeight(bt.Strategy):
    """Equal weights, put back on the first session of the run and of each quarter."""

    symbols, exchange, interval, product = FOUR, "NSE", "D", "CNC"

    def init(self):
        self.quarter = None

    def on_bar(self, bars):
        quarter = (self.now.year, (self.now.month - 1) // 3)
        if quarter != self.quarter:
            self.quarter = quarter
            self.rebalance({symbol: TARGET_PERCENT for symbol in FOUR})


@pytest.fixture(scope="module")
def quarterly() -> bt.Result:
    # 2024-10-07 to the fill of the 2026-07-01 rebalance (the session after it), with
    # the default charges and slippage.
    days = sessions("2024-10-07", "2026-07-31")
    last_fill = days[days.index(dt.date(2026, 7, 1)) + 1]
    days = days[: days.index(last_fill) + 1]
    data = {symbol: walk(*PATHS[symbol], days) for symbol in FOUR}
    return bt.backtest(QuarterlyEqualWeight, data=data, capital=CAPITAL)


def test_a_quarterly_rebalance_has_no_rejected_order(quarterly):
    orders = quarterly.orders
    # The first allocation and seven quarters, four symbols each.
    assert len(orders) == 32
    assert orders["reject_code"].isna().all()
    assert (orders["status"] == "Filled").all()


def test_after_the_last_rebalance_every_weight_is_within_the_buffer_of_its_target(quarterly):
    final = quarterly.final_positions
    equity = quarterly.equity["equity_rupees"].iloc[-1]
    weights = final["quantity_units"] * final["mark_rupees"] / equity * 100
    assert len(weights) == 4
    assert all(abs(weight - TARGET_PERCENT) <= BUFFER_PERCENT for weight in weights)


def test_a_quarterly_rebalance_is_charged_as_delivery(quarterly):
    costs = quarterly.costs
    value = costs["quantity_units"] * costs["price_rupees"]
    buys, sells = costs["side"] == "BUY", costs["side"] == "SELL"
    assert set(costs["segment"]) == {"NSE_EQ_DELIVERY"}
    assert (costs["brokerage"] == 0).all()
    assert np.allclose(costs["stt"], STT_RATE * value)
    assert np.allclose(costs.loc[buys, "stamp_duty"], STAMP_RATE_BUY * value[buys])
    assert (costs.loc[sells, "stamp_duty"] == 0).all()
    assert (costs.loc[buys, "dp_charge"] == 0).all()
    assert np.allclose(costs.loc[sells, "dp_charge"], DP_CHARGE)
