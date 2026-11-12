"""Strategy and session risk rules end to end through `bt.backtest` (05-risk-management
4 and 5, decisions SPEC-16 to SPEC-18, lead rulings on the daily loss gate, the drawdown
halt, the cooldown code and per-strategy settings).

The daily cases replay the daily-loss sequence of trace 8b on its near contract alone,
as a generic NSE cash-equity stand-in (futures margin is not built yet): bought 75 at
the 2025-07-28 open, 25010.00, the run measures -3750.00, -5250.00 and then -16125.00 at
the 2025-07-30 close, which breaches a 10000.00 limit. The intraday cases use a
15-minute strategy on a 15-minute base on 2025-07-01, every bar at 100.00 unless stated.
"""

import datetime as dt
from typing import ClassVar

import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LOT = 75
# (day of July 2025, open, high, low, close) of trace 8b's near contract.
NEAR_ROWS = [
    (25, 24950.0, 25040.0, 24900.0, 25000.0),
    (28, 25010.0, 25010.0, 24930.0, 24960.0),
    (29, 24950.0, 25090.0, 24900.0, 24940.0),
    (30, 24920.0, 24965.0, 24785.0, 24795.0),
    (31, 24805.0, 24855.0, 24765.0, 24835.0),
]


def daily_bars() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [dt.datetime(2025, 7, day, 9, 15, tzinfo=IST) for day, *_ in NEAR_ROWS],
        name="timestamp",
    )
    rows = [prices for _, *prices in NEAR_ROWS]
    frame = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000.0
    return frame


def day(n: int) -> dt.date:
    return dt.date(2025, 7, n)


class BuyOnDays(bt.Strategy):
    """Buys one lot at the closes of 2025-07-25, 30 and 31 (trace 8b's decisions)."""

    symbol, exchange, interval = "NEAR", "NSE", "D"
    buy_days = (25, 30, 31)

    def init(self):
        self.handles = []
        self.risk_events = []

    def on_bar(self, bar):
        if bar.time.day in self.buy_days:
            self.handles.append(self.buy(qty=LOT))

    def on_risk_event(self, event):
        self.risk_events.append(event)


def run_daily(strategy=BuyOnDays, **settings):
    arguments = {
        "data": daily_bars(),
        "capital": 2_500_000,
        "costs": None,
        "slippage": 0,
        "quantity_mode": "venue",
        "start": day(25),
        "end": day(31),
        **settings,
    }
    return bt.backtest(strategy, **arguments)


def capture(strategy):
    """A subclass that keeps its instance, so a test can read what it recorded."""
    instances = []

    class Captured(strategy):
        def init(self):
            super().init()
            instances.append(self)

    return Captured, instances


def test_the_daily_loss_exit_fills_at_the_close_of_the_day_that_breached():
    fills = run_daily(max_daily_loss=10_000).fills
    assert list(fills["reason"]) == ["entry", "daily_loss_limit"]
    assert list(fills["price_rupees"]) == [25010.0, 24795.0]
    assert list(fills["fill_at"]) == ["open", "close"]
    assert list(fills["step"]) == [2, 6]
    assert [list(flags) for flags in fills["flags"]] == [[], ["filled_at_last_close"]]


def test_an_entry_after_the_breach_is_rejected_with_the_daily_loss_gate_until_the_next_day():
    strategy, instances = capture(BuyOnDays)
    run_daily(strategy, max_daily_loss=10_000)
    handles = instances[0].handles
    assert [(order.status, order.reject_code) for order in handles] == [
        ("Filled", None),
        ("Rejected", "daily_loss_gate"),
        ("Submitted", None),
    ]


def test_on_risk_event_receives_the_breach_and_the_trading_state_changes():
    strategy, instances = capture(BuyOnDays)
    run_daily(strategy, max_daily_loss=10_000)
    seen = [
        (event["name"], event.get("trading_state"), event.get("observed_rupees"))
        for event in instances[0].risk_events
    ]
    assert seen == [
        ("daily_loss_limit_hit", None, -16125.0),
        ("trading_state_changed", "REDUCING", None),
        ("trading_state_changed", "ACTIVE", None),
    ]
    assert instances[0].risk_events[2]["cause"] == "session_boundary"


def test_the_gross_pnl_is_the_loss_at_the_breach():
    result = run_daily(max_daily_loss=10_000)
    assert result.gross_pnl["realized"] == pytest.approx(-16_125.0)
    assert result.gross_pnl["unrealized"] == 0.0


def test_a_class_attribute_daily_loss_limit_wins_over_the_run_argument():
    class OwnLimit(BuyOnDays):
        max_daily_loss = 10_000

    fills = run_daily(OwnLimit, max_daily_loss=100_000).fills
    assert list(fills["reason"]) == ["entry", "daily_loss_limit"]


@pytest.mark.parametrize(
    ("name", "own", "run_argument", "used"),
    [
        ("max_daily_loss", 50_000, 90_000, 50_000.0),
        ("drawdown_halt", bt.pct(5), 9, 5.0),
        ("cooldown", 2, 7, 2),
    ],
    ids=["max_daily_loss", "drawdown_halt", "cooldown"],
)
def test_a_strategys_own_risk_setting_wins_over_the_run_argument(name, own, run_argument, used):
    strategy = type("OwnSetting", (BuyOnDays,), {name: own})
    assumptions = run_daily(strategy, **{name: run_argument}).assumptions
    assert assumptions[name] == used


def test_the_assumptions_list_the_risk_rules_and_where_their_exits_fill():
    assumptions = run_daily(max_daily_loss=bt.rupees(10_000), cooldown=2).assumptions
    assert assumptions["max_daily_loss"] == 10_000.0
    assert assumptions["risk_basis"] == "gross"
    assert assumptions["cooldown"] == 2
    assert assumptions["drawdown_halt"] is None
    assert assumptions["risk_exit_fill"] == "at the close of the day that breached, flagged"


def test_a_drawdown_halt_given_in_percent_latches_and_rejects_entries_as_halted():
    # Equity 2500000.00 falls to 2500000.00 - 16125.00 = 2483875.00 on 2025-07-30,
    # 0.645 percent below the peak: a 0.5 percent halt fires there.
    strategy, instances = capture(BuyOnDays)
    result = run_daily(strategy, drawdown_halt=bt.pct(0.5))
    assert list(result.fills["reason"]) == ["entry", "drawdown_halt"]
    codes = [order.reject_code for order in instances[0].handles]
    assert codes == [None, "halted", "halted"]


# ----- intraday --------------------------------------------------------------------


def intraday_bars(opens=None) -> pd.DataFrame:
    """The 25 fifteen-minute bars of 2025-07-01; bar k opens and closes at opens(k)."""
    start = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
    prices = [100.0 if opens is None else opens(k) for k in range(25)]
    index = pd.DatetimeIndex(
        [start + dt.timedelta(minutes=15 * k) for k in range(25)], name="timestamp"
    )
    columns = {"open": prices, "high": prices, "low": prices, "close": prices}
    return pd.DataFrame({**columns, "volume": 1_000.0}, index=index)


class Scripted(bt.Strategy):
    """Acts at given bar numbers (0 is the 09:15 bar): "buy" 10 units or "exit"."""

    symbol, exchange, interval, product = "SBIN", "NSE", "15m", "CNC"
    script: ClassVar[dict[int, str]] = {}

    def init(self):
        self.handles = []
        self.bar_number = -1

    def on_bar(self, bar):
        self.bar_number += 1
        action = self.script.get(self.bar_number)
        if action == "buy":
            self.handles.append(self.buy(qty=10))
        elif action == "exit":
            self.handles.append(self.exit())


def run_intraday(strategy, opens=None, **settings):
    arguments = {
        "data": intraday_bars(opens),
        "capital": 1_000_000,
        "costs": None,
        "slippage": 0,
        "base": "15m",
        **settings,
    }
    return bt.backtest(strategy, **arguments)


def test_an_entry_inside_the_cooldown_is_rejected_with_the_cooldown_code():
    # The exit fills in bar 3; with a cooldown of 2 the entries decided at the closes of
    # bars 3 and 4 are refused and the one at bar 5 is accepted.
    class Cooling(Scripted):
        script = {0: "buy", 2: "exit", 3: "buy", 4: "buy", 5: "buy"}

    strategy, instances = capture(Cooling)
    run_intraday(strategy, cooldown=2)
    codes = [order.reject_code for order in instances[0].handles]
    assert codes == [None, None, "cooldown", "cooldown", None]


def test_a_position_held_max_hold_bars_exits_with_reason_time_exit():
    # Bar k opens at 100 + k: bought at the bar 1 open (101.00), out at the bar 4 open.
    class Holding(Scripted):
        script = {0: "buy"}
        max_hold = 3

    fills = run_intraday(Holding, opens=lambda k: 100.0 + k).fills
    assert list(fills["reason"]) == ["entry", "time_exit"]
    assert list(fills["price_rupees"]) == [101.0, 104.0]


# ----- settings ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        {"max_daily_loss": -1},
        {"max_daily_loss": bt.pct(1)},
        {"drawdown_halt": 150},
        {"drawdown_halt": bt.points(5)},
        {"cooldown": 0},
        {"cooldown": 1.5},
        {"max_positions": 0},
    ],
    ids=str,
)
def test_unusable_risk_settings_raise_a_config_error(settings):
    with pytest.raises(ConfigError):
        run_daily(**settings)


def test_a_cooldown_given_as_a_time_is_not_built_yet():
    with pytest.raises(NotImplementedError, match="cooldown as a time"):
        run_daily(cooldown="30m")


def test_a_max_hold_given_as_a_time_is_not_built_yet():
    class TimedHold(BuyOnDays):
        max_hold = "2h"

    with pytest.raises(NotImplementedError, match="max_hold as a time"):
        run_daily(TimedHold)


def test_combined_rules_are_refused_until_phase_3():
    class Combined(BuyOnDays):
        combined_stop = bt.rupees(5_000)

    with pytest.raises(NotImplementedError, match="Phase 3"):
        run_daily(Combined)
