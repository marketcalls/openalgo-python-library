"""The public `bt.live`, on the fake OpenAlgo (`run.live`; 16-session-design 1.1, 1.2 and 1.4,
unit 13): its signature, the arguments it refuses and the strategies it does not run yet,
each refused before any call to OpenAlgo, the errors it raises, and a day restarted
through it.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16, SBIN trades at 500.00 every 10 s. Expected values come from the fake's
books and the scenario.
"""

from __future__ import annotations

import inspect
import logging
import threading

import numpy as np
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import CAPITAL, MAX_DAILY_LOSS, BuyAndHold, Raises, ist, rig
from stops_samples import OPTION, OPTION_LOT, option_world
from test_session_figures import DAY_ENDS, TwoRoundTrips, fills_gross, world_of

import openalgo.backtest as bt
from openalgo.backtest._live._plan_args import MODE_ARGUMENTS
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import ConfigError, LiveError, PreflightError

#: The arguments of `bt.live`, in order (16-session-design 1.2).
ARGUMENTS = [
    "strategy",
    "client",
    "capital",
    "max_daily_loss",
    "account_max_daily_loss",
    "alerts_to",
    "account",
    "params",
    "costs",
    "risk_basis",
    "after_risk_exit",
    "entry_from",
    "entry_until",
    "square_off",
    "max_positions",
    "drawdown_halt",
    "cooldown",
    "round_levels",
    "strict_config",
    "live_dir",
    "on_stop",
    "max_trades_per_day",
    "daily_profit_target",
    "max_order_value",
    "account_max_trades_per_day",
    "account_daily_profit_target",
    "max_orders_per_second",
    "positional_stops",
    "not_live",
]


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


# ----- the signature ----------------------------------------------------------------------


def test_bt_live_takes_the_arguments_of_a_live_session():
    assert list(inspect.signature(bt.live).parameters) == ARGUMENTS


@pytest.mark.parametrize("name", ["capital", "max_daily_loss"])
def test_the_risk_limits_are_mandatory_keywords(name):
    parameter = inspect.signature(bt.live).parameters[name]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_a_call_without_a_risk_limit_is_refused_naming_it():
    with pytest.raises(TypeError, match="max_daily_loss"):
        bt.live(BuyAndHold, None, capital=CAPITAL)


def test_the_settings_shared_with_bt_backtest_take_its_defaults():
    live_parameters = inspect.signature(bt.live).parameters
    backtest = inspect.signature(bt.backtest).parameters
    shared = [
        name
        for name, parameter in live_parameters.items()
        if name in backtest and parameter.default is not inspect.Parameter.empty
    ]
    # The twelve settings of a strategy and the account's daily loss limit, then the three
    # daily limits of a strategy and the account's two (18-multi-strategy 2.5a), then how a
    # CNC or NRML position's stop is held (20-positional-live 6).
    assert len(shared) == 19
    assert {name: live_parameters[name].default for name in shared} == {
        name: backtest[name].default for name in shared
    }


def test_the_order_rate_defaults_to_the_live_layers_figure():
    from openalgo.backtest._live._order_rate import DEFAULT_ORDERS_PER_SECOND

    default = inspect.signature(bt.live).parameters["max_orders_per_second"].default
    assert default == DEFAULT_ORDERS_PER_SECOND == 8


def test_no_argument_of_bt_live_chooses_the_mode():
    assert not set(inspect.signature(bt.live).parameters) & MODE_ARGUMENTS


# ----- refused before any call to OpenAlgo -----------------------------------------------


def variant(name: str = "Variant", **attributes) -> type:
    """A `BuyAndHold` with class attributes changed."""
    return type(name, (BuyAndHold,), attributes)


class Conditions(bt.Strategy):
    """A condition strategy: its rule is declared in init."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.entry_when(self.close > self.open)


def signals(bars):
    return bars["close"] > bars["open"], bars["close"] < bars["open"]


REFUSED = {
    "condition strategy": (Conditions, {}, "condition strategy"),
    "signal strategy": (bt.from_signals(signals, "SBIN", "NSE", "1m"), {}, "bt.from_signals"),
    "array strategy": (
        bt.from_arrays(np.array([True]), symbol="SBIN", exchange="NSE", interval="1m"),
        {},
        "bt.from_arrays",
    ),
    "several symbols": (variant(symbol=None, symbols=("SBIN", "INFY")), {}, "several symbols"),
    "a universe": (variant(symbol=None, universe="NIFTY50"), {}, "a universe"),
    "futures series": (
        variant(symbol=None, underlying="NIFTY", expiry="current_month", exchange="NFO"),
        {},
        "futures series",
    ),
    "daily interval in MIS": (
        variant(interval="D", product="MIS"),
        {},
        "a D strategy in MIS in a live session",
    ),
    "two CNC on one symbol": (
        [variant("HoldA", product="CNC"), variant("HoldB", product="CNC")],
        {},
        "two CNC or NRML strategies on one symbol and product",
    ),
    "two with one name": (
        [variant("Twin"), variant("Twin", symbol="INFY")],
        {},
        "two strategies with the same name",
    ),
    "bars from candles": (BuyAndHold, {"bars": "candles"}, "bars='candles'"),
}


@pytest.mark.parametrize(("strategy", "arguments", "words"), REFUSED.values(), ids=REFUSED)
def test_what_live_trading_does_not_run_is_refused_before_any_call(
    world, strategy, arguments, words
):
    with pytest.raises(NotImplementedError, match=words) as refused:
        live(world, strategy, **arguments)
    assert " is not supported in this version: " in str(refused.value)
    assert world.fake.log.calls() == []


UNUSABLE = {
    "a mode": ({"mode": "live"}, "takes no mode=: sandbox or live is chosen inside OpenAlgo"),
    "a confirmation": ({"confirm": True}, "takes no confirm=: sandbox or live is chosen"),
    "part shares": ({"quantity_mode": "fractional"}, "takes no quantity_mode=: a live session"),
    "data": ({"data": "bars.csv"}, "takes no data=: the session trades on the broker's data"),
    "a clock": ({"now": 0}, "takes no now=: the session runs on the clock"),
    "a session's own seam": ({"threads": False}, "has no argument threads="),
    "an unknown stop": ({"on_stop": "exit"}, "on_stop takes None .* .flatten."),
    "no capital": ({"capital": None}, "bt.live needs capital="),
}


@pytest.mark.parametrize(("arguments", "words"), UNUSABLE.values(), ids=UNUSABLE)
def test_an_unusable_argument_is_refused_before_any_call(world, arguments, words):
    with pytest.raises(ConfigError, match=words):
        live(world, BuyAndHold, **arguments)
    assert world.fake.log.calls() == []


def test_off_the_main_thread_bt_live_is_refused_before_any_call(make_fake_openalgo, tmp_path):
    fake = make_fake_openalgo()
    caught: list[BaseException] = []

    def start() -> None:
        try:
            bt.live(
                BuyAndHold,
                fake.client(),
                capital=CAPITAL,
                max_daily_loss=MAX_DAILY_LOSS,
                live_dir=tmp_path,
            )
        except BaseException as error:
            caught.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    thread.join(30)
    assert isinstance(caught[0], LiveError) and "main thread" in str(caught[0])
    assert fake.log.calls() == []


# ----- the errors of a session that starts ----------------------------------------------


def test_a_live_session_without_alerts_fails_its_preflight(world):
    with pytest.raises(PreflightError) as failed:
        live(world, BuyAndHold, alerts_to=None)
    assert any("alerts" in failure for failure in failed.value.failures)
    assert world.fake.orders() == []


def test_a_feed_that_does_not_connect_is_a_live_error_with_nothing_sent(world):
    world.fake.refuse_feed()
    with pytest.raises(LiveError, match="feed"):
        live(world, BuyAndHold)
    assert world.fake.orders() == []


def test_the_strategys_own_error_is_raised_once_the_session_stopped_safely(world):
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        live(world, Raises)
    assert world.fake.position("SBIN", "NSE") == 0


class SellsAnOption(bt.Strategy):
    """Sells a lot of the option short at its first decision."""

    symbol, exchange, interval = OPTION, "NFO", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(OPTION_LOT)


def test_a_short_entry_on_an_option_is_refused_as_it_is_placed_and_nothing_is_sent(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = option_world(make_fake_openalgo, monkeypatch, tmp_path)
    seam(world, monkeypatch)
    with pytest.raises(NotImplementedError, match="selling an option to open a short position"):
        live(world, SellsAnOption)
    assert world.fake.orders() == []


# ----- a day restarted through bt.live ---------------------------------------------------


def test_a_restarted_days_result_adds_both_starts_once(make_fake_openalgo, monkeypatch, tmp_path):
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path)
    seam(world, monkeypatch)
    world.stop_at(ist(14, 46, 30))
    first = live(world, TwoRoundTrips)
    world.stop = type(world.stop)()
    world.stop_at(DAY_ENDS)
    second = live(world, TwoRoundTrips)
    assert (first.restarts, second.restarts) == (0, 1)
    assert (first.summary.closed_trades, second.summary.closed_trades) == (1, 2)
    assert second.summary.gross_pnl == fills_gross(world)
    assert len(second.results["TwoRoundTrips"].trades) == 1
