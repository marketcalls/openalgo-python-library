"""LIMIT, stop and GTT orders from `buy` and `sell` through `bt.backtest`: how the price
arguments become an order type (14-strategy-api 2.4) and what a strategy sees of them
in the result (fills, orders, events).

One NSE session on 2025-07-01, a 1-minute strategy on 1-minute bars, no charges. Every
bar is 100.00 with a high of 101.00 and a low of 99.00 unless a test overrides a minute.
An order placed at the close of bar n is first tested against bar n + 1.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._orders import order_shape
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
MINUTES = 20
CALM = (100.0, 101.0, 99.0, 100.0)


def at(minute: int) -> dt.datetime:
    """The bar opening `minute` minutes after 09:15."""
    return SESSION_OPEN + dt.timedelta(minutes=minute)


def bars(overrides: dict[int, tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [overrides.get(k, CALM) for k in range(MINUTES)]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    frame.index = pd.DatetimeIndex([at(k) for k in range(MINUTES)], name="timestamp")
    return frame


def run(strategy, overrides=None, product="MIS", slippage=0):
    return bt.backtest(
        strategy,
        data=bars(overrides or {}),
        capital=1_000_000,
        costs=None,
        slippage=slippage,
        base="1m",
        product=product,
        end=at(MINUTES - 1),
    )


def scripted(actions, handles=None):
    """A 1-minute strategy that makes `actions[n](self)` at its n-th bar close and keeps
    the handles the calls return."""

    class Scripted(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def on_bar(self, bar):
            action = actions.get(len(self.close) - 1)
            if action is not None:
                handle = action(self)
                if handles is not None:
                    handles.append(handle)

    return Scripted


# ---------------------------------------------------------------- argument conversion


@pytest.mark.parametrize(
    ("price", "trigger_price", "expected"),
    [
        (None, None, {"order_type": "MARKET", "price": None, "trigger": None}),
        (101.5, None, {"order_type": "LIMIT", "price": 101.5, "trigger": None}),
        (101.5, 101.0, {"order_type": "SL", "price": 101.5, "trigger": 101.0}),
        (None, 101.0, {"order_type": "SL-M", "price": None, "trigger": 101.0}),
    ],
    ids=["market", "limit", "stop-limit", "stop-market"],
)
def test_the_price_arguments_give_openalgos_order_types(price, trigger_price, expected):
    assert order_shape(price, trigger_price) == expected


# ---------------------------------------------------------------- through the kernel


def test_a_limit_buy_fills_at_its_limit_in_the_bar_that_reaches_it_without_slippage():
    # Placed at bar 2's close at 99.50; bar 3 ranges down to 99.00. Two ticks of
    # slippage would make a market buy worse, a resting limit takes none.
    strategy = scripted({2: lambda s: s.buy(qty=10, price=99.5)})
    fills = run(strategy, slippage=bt.ticks(2)).fills
    assert list(fills["time"]) == [at(3)]
    assert list(fills["step"]) == [3]
    assert list(fills["price_rupees"]) == [99.5]
    assert list(fills["fill_at"]) == ["level"]


def test_a_stop_market_buy_fills_at_its_trigger_when_the_range_reaches_it():
    strategy = scripted({2: lambda s: s.buy(qty=10, trigger_price=100.5)})
    fills = run(strategy).fills
    assert list(fills["price_rupees"]) == [100.5]


def test_a_limit_off_the_tick_grid_is_rejected_tick():
    handles = []
    strategy = scripted({2: lambda s: s.buy(qty=10, price=99.52)}, handles)
    result = run(strategy)
    assert (handles[0].status, handles[0].reject_code) == ("Rejected", "tick")
    assert result.fills.empty


def test_the_orders_table_shows_the_order_type_prices_and_validity():
    strategy = scripted({2: lambda s: s.buy(qty=10, price=99.0, trigger_price=99.5)})
    orders = run(strategy).orders
    row = orders.iloc[0]
    assert (row["order_type"], row["limit_rupees"], row["trigger_rupees"]) == ("SL", 99.0, 99.5)
    assert row["validity"] == "DAY"


def test_a_gtt_order_without_a_price_is_a_configuration_error():
    strategy = scripted({2: lambda s: s.buy(qty=10, validity="gtt")})
    with pytest.raises(ConfigError, match="price="):
        run(strategy, product="CNC")


def test_a_gtt_order_on_mis_is_rejected_product_rule():
    handles = []
    strategy = scripted({2: lambda s: s.buy(qty=10, price=99.0, validity="gtt")}, handles)
    run(strategy)
    assert (handles[0].status, handles[0].reject_code) == ("Rejected", "product_rule")


def test_a_gtt_buy_rests_as_a_gtt_and_its_fill_names_it():
    # Placed at bar 2's close (100.00) at 98.50: a GTT waiting for a fall. Bar 5 dips to
    # 98.00, reaching the trigger; the child buys at 98.50.
    strategy = scripted({2: lambda s: s.buy(qty=10, price=98.5, validity="gtt")})
    result = run(strategy, overrides={5: (99.0, 99.5, 98.0, 98.8)}, product="CNC")
    fills = result.fills
    assert list(fills["time"]) == [at(5)]
    assert list(fills["price_rupees"]) == [98.5]
    assert list(fills["gtt_id"]) == [1]
    events = result.events
    names = list(events.loc[events["type"] == "gtt", "name"])
    assert names == ["GttPlaced", "GttTriggered", "GttChildFilled"]
    assert result.orders.iloc[0]["validity"] == "GTT"
