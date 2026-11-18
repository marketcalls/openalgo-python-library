"""The exit reason an order's fills record, through the binding
(`Engine.order_exit_reason`; rust/oa_py/src/backtest/engine/session.rs): the day summary's
slippage (`_live/_session_slippage.py`) tells by it the exit a leg's target makes on a tick
from the strategy's own exits, which share its role "exit".

No server is involved: the test plays the venue. The sessions are `test_kernel_legs`' (a
buy of 10 with a stop 3 points below, order 2) and `test_kernel_session`'s (a buy of 100,
order 1, held at the close of minute 1).
"""

from __future__ import annotations

import pytest
from test_kernel_legs import STOP, at, holding_with_stop, placed
from test_kernel_session import EXIT, holding, places

from openalgo.backtest._calendar_convert import native_attr


def test_the_exit_a_targets_tick_makes_records_leg_target():
    engine = holding_with_stop({"target": ("points", 2.0, None)})
    engine.push_tick(0, 502.0, at(1, 5))
    engine.take_commands()
    engine.apply_cancel(STOP)
    (target_exit,) = placed(engine)
    assert engine.order_exit_reason(target_exit["order_id"]) == "leg_target"


@pytest.mark.parametrize(
    ("reason", "exit_reason"),
    [
        ("square_off", "scheduler"),
        ("time_exit", "time_exit"),
        ("daily_loss", "daily_loss_limit"),
        ("stop_unfilled", "leg_sl"),
        ("operator_stop", "manual"),
    ],
    ids=["square-off", "time exit", "daily loss", "stop unfilled", "Ctrl+C"],
)
def test_an_exit_of_the_engine_records_the_reason_its_trade_reads(reason, exit_reason):
    engine = holding(100.0)
    assert engine.advance()[0] == "waiting"
    assert engine.request_exit(0, "MIS", reason) == "sent"
    [(order_id, *_)] = places(engine)
    assert engine.order_exit_reason(order_id) == exit_reason


@pytest.mark.parametrize(
    ("request_", "exit_reason"),
    [(EXIT, None), (EXIT | {"exit_reason": "manual"}, "manual")],
    ids=["exit()", "exit(reason='manual')"],
)
def test_a_strategys_exit_records_the_reason_it_gave_or_none(request_, exit_reason):
    engine = holding(100.0)
    outcome, order_id, _ = engine.submit(request_)
    assert outcome == "submitted"
    assert engine.order_exit_reason(order_id) == exit_reason


def test_an_entry_records_no_exit_reason():
    engine = holding(100.0)
    assert engine.order_exit_reason(1) is None


def test_an_order_the_run_does_not_know_is_an_error():
    engine = holding(100.0)
    with pytest.raises(native_attr("NativeEngineError"), match="no order"):
        engine.order_exit_reason(99)
