"""The scale-in limits through `bt.backtest` (12-position-sizing 2, decisions SPEC-14 and
SPEC-15, reject code `scale_in_limit` of 05-risk-management 7): the class attributes
`max_adds`, `max_position`, `min_spacing`, `max_risk` and `allow_averaging_down`.

A 15-minute MIS strategy on a 15-minute base on 2025-07-01 (25 bars, 09:15 to 15:15).
Bar k opens and closes at the prices `prices(k)` gives (100.00 and 100.00 unless a test
says otherwise), so an order decided at the close of bar k fills at the open of bar
k + 1. The generic NSE cash-equity instrument has lot 1 and tick 0.05; capital is
10,00,000.00.
"""

import datetime as dt
from typing import ClassVar

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BARS = 25
REFUSED = "scale_in_limit"


def bars(prices=None) -> pd.DataFrame:
    """The bars of the module docstring; `prices(k)` is (open, close) of bar k."""
    start = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
    rows = [(100.0, 100.0) if prices is None else prices(k) for k in range(BARS)]
    frame = pd.DataFrame(
        {
            "open": [o for o, _ in rows],
            "high": [max(o, c) for o, c in rows],
            "low": [min(o, c) for o, c in rows],
            "close": [c for _, c in rows],
            "volume": 1_000.0,
        },
        index=pd.DatetimeIndex(
            [start + dt.timedelta(minutes=15 * k) for k in range(BARS)], name="timestamp"
        ),
    )
    return frame


class Scripted(bt.Strategy):
    """Buys at given bar numbers (0 is the 09:15 bar) with the arguments given there."""

    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"
    script: ClassVar[dict[int, dict]] = {}

    def init(self):
        self.bar_number = -1
        self.handles = []

    def on_bar(self, bar):
        self.bar_number += 1
        arguments = self.script.get(self.bar_number)
        if arguments is not None:
            self.handles.append(self.buy(**arguments))


def reject_codes(script: dict, prices=None, **limits) -> list:
    """The reject code of every order the script placed (None when not rejected)."""
    instances = []

    class Run(Scripted):
        def init(self):
            super().init()
            instances.append(self)

    for name, value in {"script": script, **limits}.items():
        setattr(Run, name, value)
    bt.backtest(Run, data=bars(prices), capital=1_000_000, costs=None, slippage=0, base="15m")
    # init runs on a probe instance first (the reference bars), then on the run's.
    return [order.reject_code for order in instances[-1].handles]


TEN = {"qty": 10}
# An entry at bar 0 and adds at bars 2 and 4.
THREE_BUYS = {0: TEN, 2: TEN, 4: TEN}


def test_no_scale_in_limit_is_set_unless_given():
    assert reject_codes(THREE_BUYS) == [None, None, None]


def test_max_adds_refuses_the_add_beyond_the_limit():
    assert reject_codes(THREE_BUYS, max_adds=1) == [None, None, REFUSED]


@pytest.mark.parametrize(
    ("cap", "codes"),
    [
        # After the second add the position is 30 units, 3000.00 at the decision price
        # 100.00, which is 0.3 percent of the equity of 10,00,000.00.
        (30, [None, None, None]),
        (25, [None, None, REFUSED]),
        (bt.size.qty(25), [None, None, REFUSED]),
        (bt.size.lots(30), [None, None, None]),
        (bt.size.lots(29), [None, None, REFUSED]),
        (bt.rupees(3_000), [None, None, None]),
        (bt.size.value(2_999), [None, None, REFUSED]),
        (bt.pct(0.3), [None, None, None]),
        (bt.size.percent(0.29), [None, None, REFUSED]),
    ],
    ids=str,
)
def test_max_position_caps_the_position_after_the_add(cap, codes):
    assert reject_codes(THREE_BUYS, max_position=cap) == codes


def test_an_entry_from_flat_is_not_held_to_the_scale_in_limits():
    assert reject_codes({0: {"qty": 50}}, max_position=10, max_adds=0) == [None]


def test_min_spacing_in_bars_counts_the_bars_closed_after_the_last_tranche():
    # The entry fills in bar 1; adds decided at the closes of bars 1 and 2 are refused,
    # the one at bar 3 is accepted and fills in bar 4, so bar 4's add is refused.
    script = {k: TEN for k in range(5)}
    assert reject_codes(script, min_spacing=2) == [None, REFUSED, REFUSED, None, REFUSED]


def test_min_spacing_as_a_price_distance_is_measured_from_the_last_tranche():
    # Bar k opens and closes at 100.00 + 0.50 k. The entry fills at 100.50 (bar 1); bar
    # 1 closes there and bar 2 at 101.00, short of 1 point; bar 3 closes at 101.50.
    script = {k: TEN for k in range(4)}

    def rising(k):
        return (100.0 + 0.5 * k, 100.0 + 0.5 * k)

    codes = reject_codes(script, rising, min_spacing=bt.points(1))
    assert codes == [None, REFUSED, REFUSED, None]


def stop_then_rise(k):
    """Bar 1 opens at 100.00 and closes at 104.00; the rest are flat at 104.00."""
    return (100.0, 100.0) if k < 1 else ((100.0, 104.0) if k == 1 else (104.0, 104.0))


@pytest.mark.parametrize(
    ("cap", "codes"),
    [(100, [None, None]), (bt.rupees(99), [None, REFUSED]), (bt.pct(0.01), [None, None])],
    ids=str,
)
def test_max_risk_caps_the_open_risk_of_the_position_after_the_add(cap, codes):
    # Bought 10 at 100.00 with a 5-point stop; adding 10 at the decision price 104.00
    # gives an average of 102.00 and a stop of 97.00, an open risk of 5 x 20 = 100.00,
    # which is 0.01 percent of 10,00,000.00.
    script = {0: {"qty": 10, "stop": bt.points(5)}, 1: TEN}
    assert reject_codes(script, stop_then_rise, max_risk=cap) == codes


def test_max_risk_refuses_an_add_to_a_position_without_a_stop():
    assert reject_codes({0: TEN, 2: TEN}, max_risk=1_000_000) == [None, REFUSED]


def falls_after_the_entry(k):
    """The entry fills at 100.00 at the bar 1 open; bar 1 closes at 99.00."""
    return (100.0, 100.0) if k < 1 else ((100.0, 99.0) if k == 1 else (99.0, 99.0))


def test_an_add_below_the_average_is_refused_by_default():
    assert reject_codes({0: TEN, 1: TEN}, falls_after_the_entry) == [None, REFUSED]


def test_allow_averaging_down_lets_an_add_below_the_average_through():
    codes = reject_codes({0: TEN, 1: TEN}, falls_after_the_entry, allow_averaging_down=True)
    assert codes == [None, None]


def test_a_condition_strategy_adds_only_within_the_scale_in_limits():
    # pyramiding 5 with an entry condition true on every bar: the entry, then one add.
    class Pyramid(bt.Strategy):
        symbol, exchange, interval, product = "SBIN", "NSE", "15m", "MIS"
        pyramiding = 5
        max_adds = 1

        def init(self):
            self.entry_when([True] * BARS, qty=10)

    result = bt.backtest(
        Pyramid, data=bars(), capital=1_000_000, costs=None, slippage=0, base="15m"
    )
    entries = result.fills[result.fills["reason"].isin(["entry", "add"])]
    assert list(entries["reason"]) == ["entry", "add"]
    assert (result.orders["reject_code"] == REFUSED).sum() > 0


def test_the_assumptions_list_the_scale_in_limits():
    instances = []

    class Limited(Scripted):
        script: ClassVar[dict[int, dict]] = {0: TEN}
        max_adds = 2
        max_position = bt.size.lots(4)
        min_spacing = bt.points(1)
        max_risk = bt.pct(1)

        def init(self):
            super().init()
            instances.append(self)

    result = bt.backtest(
        Limited, data=bars(), capital=1_000_000, costs=None, slippage=0, base="15m"
    )
    block = result.assumptions
    assert block["max_adds"] == 2
    assert block["max_position"] == "4 lots"
    assert block["min_spacing"] == "1 points"
    assert block["max_risk"] == "1 percent of equity"
    assert block["allow_averaging_down"] is False


@pytest.mark.parametrize(
    "limits",
    [
        {"max_adds": -1},
        {"max_adds": 1.5},
        {"max_adds": True},
        {"max_position": 0},
        {"max_position": bt.points(5)},
        {"max_position": bt.size.fixed_risk(1)},
        {"max_position": bt.size.percent(10, basis="cash")},
        {"min_spacing": 0},
        {"min_spacing": 1.5},
        {"min_spacing": bt.rupees(5)},
        {"min_spacing": bt.atr(1, 14)},
        {"min_spacing": "30m"},
        {"max_risk": 0},
        {"max_risk": bt.points(5)},
        {"max_risk": bt.pct(100)},
        {"allow_averaging_down": "yes"},
    ],
    ids=str,
)
def test_unusable_scale_in_limits_raise_a_config_error(limits):
    with pytest.raises(ConfigError):
        reject_codes({0: TEN}, **limits)
