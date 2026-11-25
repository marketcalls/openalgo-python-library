"""A sold option carried overnight is refused in a live session before anything is sent
(20-positional-live 1 and decision 11.5): the kernel of a live session refuses, at
submission, every sale of an option that would open or add to a short position, on NRML
and on MIS alike (short-option margin is not built: rust/oa_bt/src/market/margin.rs), so
the strategy's call raises NotImplementedError and no command reaches the hand-over.
Option buying on NRML passes.

The kernel session is played by hand (no server): a NIFTY call on NFO, lot 75, on
1-minute slices of Tuesday 2025-07-01 from 09:15 IST, bars flat at 100.00.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 10**9
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
LOT = 75
CALL = {
    "symbol": "NIFTY31JUL2525000CE",
    "exchange": "NFO",
    "kind": "OPT",
    "lot_size": LOT,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": to_day(dt.date(2025, 7, 31)),
}


def open_ns() -> int:
    """09:15 IST of `DAY`, UTC nanoseconds."""
    return int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9


def session() -> Any:
    """A live kernel session of `DAY` on the call, paused at its first bar's close."""
    config = {
        "first_day": to_day(DAY),
        "last_day": to_day(DAY),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 10_000_000.0,
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }
    engine = native_attr("Engine").session(default_calendar()._cal, config, [CALL], [NO_BARS])
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, (open_ns(), 100.0, 100.0, 100.0, 100.0, 1_000.0))
    engine.release(1)
    assert engine.advance()[0] == "bar_close"
    return engine


def order(side: str, product: str) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": float(LOT)},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": product,
        "reduce_only": False,
        "reverse": False,
        "rules": {},
    }


@pytest.mark.parametrize("product", ["NRML", "MIS"])
def test_a_sale_that_opens_a_short_option_is_refused_at_submission_and_never_sent(product):
    engine = session()

    with pytest.raises(NotImplementedError) as refused:
        engine.submit(order("SELL", product))

    assert str(refused.value) == (
        "selling an option to open a short position (short-option margin) is not supported "
        "in this version: buy options, or trade futures"
    )
    assert engine.advance()[0] == "waiting"
    assert engine.take_commands() == []


def test_an_option_bought_on_nrml_is_placed():
    engine = session()
    engine.submit(order("BUY", "NRML"))
    assert engine.advance()[0] == "waiting"

    (placed,) = engine.take_commands()

    assert (placed["command"], placed["side"], placed["product"]) == ("place", "BUY", "NRML")
