"""An option leg's stop in a backtest is the stop-limit order the exchange takes for it
(16-session-design R12, through the binding's `Engine`): its limit 5 percent beyond the
stop, and a gap through the limit fills at the next bar. Cash equity keeps its
stop-market rule.

The strategy buys 75 units at the close of minute 0 (09:15 IST on 2025-07-01) with a stop
3 points below, so the position opens at 500.10 at the open of minute 1 with its stop at
497.10 and, for the option, a limit 5 percent below that, 472.245, down to the tick 472.20.
The bars rise 0.10 a minute from 500.00 until a row overrides them. No costs, and no
slippage but where a test says otherwise.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
# 09:15 IST is 03:45 UTC.
OPEN_NS = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
SESSION_MINUTES = 375
SBIN = {
    "symbol": "SBIN",
    "exchange": "NSE",
    "kind": "EQ",
    "lot_size": 1,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": None,
}
NIFTY_CALL = {
    **SBIN,
    "symbol": "NIFTY31JUL2525000CE",
    "exchange": "NFO",
    "kind": "OPT",
    "lot_size": 75,
    "last_trading_day": to_day(dt.date(2025, 7, 31)),
}
GAPPED_FOR_THE_BAR = {5: (470.0, 471.0, 469.0, 470.5), 6: (468.0, 469.0, 467.0, 468.5)}


def config(slippage: tuple = ("zero", 0.0, 0)) -> dict[str, Any]:
    """A cost-free 1-minute run on 1-minute bars of DAY, slippage-free unless given."""
    return {
        "first_day": to_day(DAY),
        "last_day": to_day(DAY),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 1_000_000.0,
        "costs": None,
        "slippage": slippage,
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }


def bars(overrides: dict[int, tuple[float, float, float, float]]) -> tuple:
    """The session's 1-minute bars as arrays: open 500.00 + 0.10 k, high the open plus
    0.50, low the open less 0.50, close the open plus 0.05, unless overridden."""
    rows = []
    for k in range(SESSION_MINUTES):
        open_ = 500.0 + 0.1 * k
        rows.append(overrides.get(k, (open_, open_ + 0.5, open_ - 0.5, open_ + 0.05)))
    prices = np.array(rows, dtype=np.float64)
    ts = np.array([OPEN_NS + k * MINUTE_NS for k in range(SESSION_MINUTES)], dtype=np.int64)
    volume = np.full(SESSION_MINUTES, 1_000.0)
    return (ts, prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3], volume)


def backtest(
    instrument: dict[str, Any],
    overrides: dict,
    run_config: dict[str, Any] | None = None,
    stop_points: float = 3.0,
) -> dict[str, Any]:
    """The backtest's result tables for the buy with its stop `stop_points` below."""
    engine_class = native_attr("Engine")
    calendar = default_calendar()._cal
    engine = engine_class(calendar, run_config or config(), [instrument], [bars(overrides)])
    buy = {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": 75},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        "rules": {"stop": ("points", stop_points, None)},
    }
    while (pause := engine.advance())[0] != "finished":
        if pause[0] == "bar_close" and pause[1] == OPEN_NS + MINUTE_NS:
            assert engine.submit(buy)[0] == "submitted"
    return engine.finish()


@pytest.mark.parametrize(
    ("instrument", "overrides", "minute", "price", "where"),
    [
        (NIFTY_CALL, {5: (500.5, 501.0, 497.0, 500.6)}, 5, 497.1, "level"),
        (NIFTY_CALL, {5: (490.0, 491.0, 489.0, 490.5)}, 5, 490.0, "open"),
        (NIFTY_CALL, {5: (470.0, 473.0, 469.0, 471.0)}, 5, 472.2, "level"),
        (NIFTY_CALL, GAPPED_FOR_THE_BAR, 6, 468.0, "open"),
        (SBIN, GAPPED_FOR_THE_BAR, 5, 470.0, "open"),
    ],
    ids=[
        "option reached inside the range: at the stop",
        "option gapped between the stop and its limit: at the open",
        "option gapped beyond its limit and back to it: at the limit",
        "option gapped beyond its limit for the bar: at the next open",
        "share gapped through its stop: at the open, as before",
    ],
)
def test_an_option_legs_stop_fills_as_the_stop_limit_the_exchange_takes(
    instrument, overrides, minute, price, where
):
    fills = backtest(instrument, overrides)["fills"]
    assert fills["slice_ts"][-1] == OPEN_NS + minute * MINUTE_NS
    assert fills["price_rupees"][-1] == pytest.approx(price, abs=1e-9)
    assert (fills["fill_at"][-1], fills["reason"][-1]) == (where, "leg_sl")


@pytest.mark.parametrize(
    ("instrument", "price"),
    [(NIFTY_CALL, 465.6), (SBIN, 460.6)],
    ids=["option: held at its limit", "share: the slipped open"],
)
def test_slippage_never_takes_an_option_legs_stop_past_its_limit(instrument, price):
    # 2 percent slippage: the buy fills at 510.15, its stop 20 points below at 490.15 and
    # the option's limit at 465.60. Minute 5 gaps to 470.00, between the two; slipped, the
    # open would sell at 460.60.
    slipping = config(slippage=("percent", 2.0, 0))
    gap = {5: (470.0, 471.0, 469.0, 470.5)}
    fills = backtest(instrument, gap, slipping, stop_points=20.0)["fills"]
    assert fills["slice_ts"][-1] == OPEN_NS + 5 * MINUTE_NS
    assert fills["price_rupees"][-1] == pytest.approx(price, abs=1e-9)
