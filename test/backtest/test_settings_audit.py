"""No setting is accepted and then ignored: every class attribute a strategy declares and
every argument `bt.backtest` takes either works or raises NotImplementedError naming
the feature; a value that cannot apply to the run says so in the result's warnings.

The runs use `accounting_runs`: one NSE share on 2026-11-04, 1-minute bars, a 15-minute
strategy that buys at its first close (filled at the 09:30 open, 1,000.00) and exits at
its second; capital 10,00,000.00.
"""

import inspect

import numpy as np
import pytest
from accounting_runs import CAPITAL, at, round_trip_strategy, session_bars

import openalgo.backtest as bt
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.strategy import DECLARATIONS

# Every class attribute of `Strategy` with a meaning of its own, and every argument of
# `bt.backtest`, as audited. A name added later must be audited (made to work, or made
# to raise NotImplementedError) and then listed here.
AUDITED_DECLARATIONS = {
    "symbol",
    "symbols",
    "universe",
    "underlying",
    "expiry",
    "exchange",
    "interval",
    "product",
    "leverage",
    "size",
    "reinvest",
    "size_caps",
    "pyramiding",
    "max_adds",
    "max_position",
    "min_spacing",
    "max_risk",
    "allow_averaging_down",
    "max_hold",
    "max_daily_loss",
    "entry_from",
    "entry_until",
    "square_off",
    "drawdown_halt",
    "cooldown",
    "combined_stop",
    "combined_target",
    "lock_profit",
    "trail_sl_to_entry",
    "state",
}
AUDITED_RUN_ARGUMENTS = {
    "strategy",
    "client",
    "capital",
    "start",
    "end",
    "data",
    "instrument",
    "calendar",
    "source",
    "tz",
    "base",
    "fill",
    "quantity_mode",
    "costs",
    "slippage",
    "product",
    "entry_from",
    "entry_until",
    "square_off",
    "max_daily_loss",
    "risk_basis",
    "after_risk_exit",
    "max_positions",
    "drawdown_halt",
    "cooldown",
    "round_levels",
    "strict_config",
    "roll",
    "params",
}


def run(strategy, **settings):
    arguments = {
        "data": session_bars(),
        "capital": CAPITAL,
        "base": "1m",
        "costs": None,
        "slippage": 0,
        **settings,
    }
    return bt.backtest(strategy, **arguments)


def margin_after_entry(result) -> float:
    """Margin blocked at the close of the entry slice (09:30 to 09:31)."""
    return result.equity.loc[at(9, 31), "margin_rupees"]


def test_every_declaration_and_run_argument_has_been_audited():
    assert set(DECLARATIONS) == AUDITED_DECLARATIONS
    assert set(inspect.signature(bt.backtest).parameters) == AUDITED_RUN_ARGUMENTS


# ----- leverage: a strategy-wide default, overridden on an order (14-strategy-api 2.4) --


def test_a_class_leverage_is_the_default_of_every_order():
    # 1,000 shares at 1,000.00 are 10,00,000.00; at 5x they block 2,00,000.00.
    strategy = type("Levered", (round_trip_strategy(qty=1_000),), {"leverage": 5})
    assert margin_after_entry(run(strategy, product="MIS")) == 200_000.0


def test_leverage_on_an_order_wins_over_the_class_leverage():
    # The order asks for 2x: 10,00,000.00 / 2 = 5,00,000.00.
    strategy = type("Levered", (round_trip_strategy(qty=1_000, leverage=2),), {"leverage": 5})
    assert margin_after_entry(run(strategy, product="MIS")) == 500_000.0


def test_a_condition_strategy_takes_the_class_leverage():
    entries = np.zeros(25, dtype=bool)
    entries[0] = True
    exits = np.zeros(25, dtype=bool)
    exits[1] = True
    arrays = bt.from_arrays(
        entries, exits, symbol="SBIN", exchange="NSE", interval="15m", product="MIS", qty=1_000
    )
    strategy = type("LeveredArrays", (arrays,), {"leverage": 5})
    assert margin_after_entry(run(strategy)) == 200_000.0


@pytest.mark.parametrize("leverage", [0.5, "5", True], ids=str)
def test_an_unusable_class_leverage_raises_a_config_error(leverage):
    strategy = type("Levered", (round_trip_strategy(qty=10),), {"leverage": leverage})
    with pytest.raises(ConfigError, match="leverage"):
        run(strategy, product="MIS")


def test_the_assumptions_state_the_class_leverage():
    strategy = type("Levered", (round_trip_strategy(qty=10),), {"leverage": 5})
    assert run(strategy, product="MIS").assumptions["leverage"] == 5.0


# ----- settings that cannot apply ----------------------------------------------------


def test_a_class_attribute_state_is_refused():
    # self.state is the strategy's own dict; a class-level dict would be dropped.
    strategy = type("Stateful", (round_trip_strategy(qty=10),), {"state": {"seen": 0}})
    with pytest.raises(ConfigError, match="state"):
        run(strategy)


def test_after_risk_exit_other_than_the_default_is_not_built_yet():
    # It governs only the combined rules' exits, which are Phase 3.
    with pytest.raises(NotImplementedError, match="after_risk_exit"):
        run(round_trip_strategy(qty=10), after_risk_exit="allow_reentry")


def test_pyramiding_on_an_on_bar_strategy_is_reported_as_unused():
    strategy = type("Pyramid", (round_trip_strategy(qty=10),), {"pyramiding": 2})
    warnings = run(strategy).warnings
    assert any("pyramiding" in text and "max_adds" in text for text in warnings)


def test_a_client_given_with_own_data_is_reported_as_unused():
    class Client:
        """A stand-in that is never called."""

    warnings = run(round_trip_strategy(qty=10), client=Client()).warnings
    assert any(text.startswith("client= is not used") for text in warnings)


def test_intraday_times_on_a_delivery_run_are_reported_as_applying_to_mis_only():
    warnings = run(round_trip_strategy(qty=10), product="CNC", entry_until="14:00").warnings
    assert any("entry_until" in text and "MIS" in text for text in warnings)


# ----- the Assumptions block --------------------------------------------------------


def test_the_assumptions_state_the_entry_window_start_of_an_intraday_run():
    # The NSE session opens at 09:15 IST.
    assert run(round_trip_strategy(qty=10), product="MIS").assumptions["entry_from"] == "09:15"


def test_the_assumptions_state_a_given_entry_from():
    result = run(round_trip_strategy(qty=10), product="MIS", entry_from="09:20")
    assert result.assumptions["entry_from"] == "09:20"


def test_a_delivery_run_has_no_entry_window():
    assert run(round_trip_strategy(qty=10), product="CNC").assumptions["entry_from"] is None
