"""Leg rules from `buy` and `sell` through `bt.backtest`: what the strategy layer adds
to the kernel (ATR distances resolved at the decision, argument checks) and what a
strategy sees (exit fills with their reason and target, `reverse=True`).

One NSE session on 2025-07-01, a 1-minute strategy on 1-minute bars, MIS, no charges
and no slippage. Every bar is 100.00 with a high of 101.00 and a low of 99.00 unless a
test overrides a minute, so its true range is 2.00 and every ATR of it is 2.00.
"""

import datetime as dt

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._atr import decision_atr
from openalgo.backtest._orders import leg_rules
from openalgo.backtest._series import BarStore
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


def run(strategy, overrides=None):
    return bt.backtest(
        strategy,
        data=bars(overrides or {}),
        capital=1_000_000,
        costs=None,
        slippage=0,
        base="1m",
        product="MIS",
        end=at(MINUTES - 1),
    )


def scripted(actions):
    """A 1-minute strategy that makes `actions[n](self)` at its n-th bar close."""

    class Scripted(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def on_bar(self, bar):
            action = actions.get(len(self.close) - 1)
            if action is not None:
                action(self)

    return Scripted


def never(_period):
    raise AssertionError("no ATR distance here")


# ---------------------------------------------------------------- argument conversion


def test_distances_cross_as_unit_value_and_atr():
    rules = leg_rules(
        bt.points(4),
        bt.pct(2),
        None,
        bt.trail(3, 1),
        bt.breakeven(2, bt.ticks(5)),
        "breakeven",
        atr_at=never,
    )
    assert rules == {
        "stop": ("points", 4.0, None),
        "target": ("pct", 2.0, None),
        "targets": [],
        "trail": (("points", 3.0, None), ("points", 1.0, None)),
        "breakeven": (("points", 2.0, None), ("ticks", 5.0, None)),
        "after_first_target": "breakeven",
    }


def test_an_atr_distance_carries_the_atr_of_its_period_at_the_decision():
    rules = leg_rules(None, None, None, bt.trail_atr(1.5, 14), None, None, atr_at=lambda p: p / 7)
    assert rules["trail"] == (("atr", 1.5, 2.0), None)


def test_no_rule_gives_no_rules():
    assert leg_rules(None, None, None, None, None, None, atr_at=never) is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"targets": [bt.points(5)]},
        {"trail": bt.points(3)},
        {"breakeven": 2},
        {"after_first_target": "trail"},
    ],
    ids=["target without fraction", "trail as a distance", "bare breakeven", "unknown move"],
)
def test_malformed_leg_arguments_are_configuration_errors(arguments):
    given = {"stop": None, "target": None, "targets": None, "trail": None, "breakeven": None}
    given["after_first_target"] = None
    given.update(arguments)
    with pytest.raises(ConfigError):
        leg_rules(**given, atr_at=never)


# ---------------------------------------------------------------- the ATR at the decision


def store_of(rows):
    store = BarStore("SBIN")
    for number, (high, low, close) in enumerate(rows):
        store.append((0, number, number + 1, close, high, low, close, 0.0))
    return store


def test_the_decision_atr_is_wilders_atr_at_the_latest_closed_bar():
    # True ranges 1.00, 1.50, 1.50, 1.50: ATR(3) is 4.00 / 3 at the third bar, then
    # (4.00 / 3 x 2 + 1.50) / 3 = 25 / 18 at the fourth.
    store = store_of([(10.0, 9.0, 9.5), (11.0, 9.5, 10.5), (12.0, 10.5, 11.5), (11.5, 10.0, 11.0)])
    assert decision_atr(store, 3) == pytest.approx(25 / 18)


def test_an_atr_needs_its_period_of_closed_bars():
    with pytest.raises(ConfigError, match="needs 3 closed strategy bars"):
        decision_atr(store_of([(10.0, 9.0, 9.5), (11.0, 9.5, 10.5)]), 3)


# ---------------------------------------------------------------- through the kernel


def test_an_atr_stop_sits_its_multiple_of_the_atr_below_the_fill():
    # The ATR(3) at the third close is 2.00; stop 2 x 2.00 = 4.00 under the 100.00 fill.
    result = run(scripted({2: lambda s: s.buy(qty=10, stop=bt.atr(2, 3))}))
    first = result.positions.iloc[0]
    assert (first["time"], first["stop_rupees"]) == (at(3), 96.0)


def test_an_atr_trail_arms_at_its_distance_and_trails_by_it():
    # bt.trail_atr(1.5, 3): x = 1.5 x 2.00 = 3.00 points from the 100.00 fill. The 09:20
    # bar's high 103.50 arms it at 100.50.
    overrides = {5: (101.0, 103.5, 100.5, 103.0)}
    strategy = scripted({2: lambda s: s.buy(qty=10, trail=bt.trail_atr(1.5, 3))})
    armed = run(strategy, overrides).positions.iloc[1]
    assert (armed["time"], armed["trail_armed"], armed["stop_rupees"]) == (at(5), True, 100.5)


def test_an_atr_order_before_the_period_has_closed_is_a_configuration_error():
    with pytest.raises(ConfigError, match="needs 14 closed strategy bars"):
        run(scripted({2: lambda s: s.buy(qty=10, stop=bt.atr(2, 14))}))


def test_targets_scale_out_with_their_index_and_the_stop_takes_the_rest():
    # 100 at 100.00: 50 at 102.00 (target 0), 30 at 103.00 (target 1), stop 98.00. The
    # 09:20 bar reaches 103.50: both fill, nearest first. The 09:22 bar opens through the
    # stop at 97.00 and the stop takes the 20 left.
    overrides = {5: (100.5, 103.5, 100.5, 103.0), 7: (97.0, 97.5, 96.5, 97.0)}
    targets = [(bt.points(2), 0.5), (bt.points(3), 0.3)]
    strategy = scripted({2: lambda s: s.buy(qty=100, stop=bt.points(2), targets=targets)})
    fills = run(strategy, overrides).fills
    exits = fills[fills["reason"] != "entry"]
    assert list(exits["quantity_units"]) == [50, 30, 20]
    assert list(exits["price_rupees"]) == [102.0, 103.0, 97.0]
    assert list(exits["reason"]) == ["leg_target", "leg_target", "leg_sl"]
    assert [None if pd.isna(i) else int(i) for i in exits["target_index"]] == [0, 1, None]


def test_reverse_closes_and_opens_the_rest_on_the_other_side():
    strategy = scripted({2: lambda s: s.buy(qty=100), 4: lambda s: s.sell(qty=150, reverse=True)})
    result = run(strategy)
    assert list(result.fills["quantity_units"]) == [100, 100, 50]
    assert list(result.fills["reason"]) == ["entry", "signal", "entry"]
    assert result.final_positions["quantity_units"].iloc[0] == -50


def test_an_order_beyond_the_position_without_reverse_is_rejected():
    handles = []
    strategy = scripted({2: lambda s: s.buy(qty=100), 4: lambda s: handles.append(s.sell(qty=150))})
    result = run(strategy)
    assert (handles[0].status, handles[0].reject_code) == ("Rejected", "reverse_not_allowed")
    assert len(result.fills) == 1


def test_an_add_that_changes_the_stop_is_a_configuration_error():
    strategy = scripted(
        {2: lambda s: s.buy(qty=10, stop=bt.points(4)), 4: lambda s: s.buy(qty=10, stop=5)}
    )
    with pytest.raises(ConfigError, match="may not change them"):
        run(strategy)


def test_leg_rules_on_a_delivery_position_are_not_built_yet():
    strategy = scripted({2: lambda s: s.buy(qty=10, stop=bt.points(4), product="CNC")})
    with pytest.raises(NotImplementedError, match="GTT"):
        run(strategy)
