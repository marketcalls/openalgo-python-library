"""An option carried into the last trading day of its contract in `bt.live`, on the fake
OpenAlgo (design 20-positional-live sections 4.2 and 9; the lead's integration ruling 7c of
review round 4 of unit N5a): the start of its last trading day restores the position and
its stop GTT from Monday's day-close record (no second place); the kernel exits it at the
NFO square-off of that day, exit reason `expiry_exit`, its stop GTT cancelled first; and an
entry after that time is refused `contract_expiring`, with nothing sent.

The market is `expiry_samples`': the NIFTY weekly call of Tuesday 2026-11-17 at 24000,
trading one lot (65 units) every 10 s at 120.00. On Monday 2026-11-16 the session starts at
15:00:05 IST; `OptionKept` buys one lot NRML at its first decision with a stop 10 points
below, held by the broker as a SINGLE GTT, and the session ends at NFO's close with the leg
carried. The fake's day then ends (the NRML position carries over, the GTT stays), and
Tuesday's session starts at 15:00:05 IST and is stopped (Ctrl+C) at 15:14. Each test runs
in the broker's book and the sandbox's, on both GTT book shapes. Expected values come from
the scenario and the calendar.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import pytest
from expiry_samples import LOT, OPTION, OPTION_DAY, PRICE, expiry_world, flat_candles, ist
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls
from session_samples import NS, TRADE_EVERY_S, StopFlag, alerts

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

MONDAY = dt.date(2026, 11, 16)
#: Each day's session start, IST.
START = (15, 0, 5)
#: The strategy's capital and daily loss limit, rupees.
LIMITS = {"capital": 100_000, "max_daily_loss": 5_000}
#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class OptionKept(bt.Strategy):
    """Buys one lot of the option NRML at its first decision with a stop 10 points below,
    the entry kept in `self.state`; flat again, it buys once more (refused once the
    contract's exit has passed), the refusal's code kept in `refused`."""

    symbol, exchange, interval, product = OPTION, "NFO", "1m", "NRML"

    def init(self):
        self.refused: list[str] = []

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(LOT, stop=bt.points(10))
        elif self.position.is_flat and not self.refused:
            order = self.buy(LOT, stop=bt.points(10))
            if order is not None and order.reject_code is not None:
                self.refused.append(order.reject_code)


def carried_to_its_last_day(make_fake_openalgo, monkeypatch, tmp_path, *, analyzer, history):
    """Monday's session of `OptionKept` run to NFO's close (module docstring), then the move
    to Tuesday's start, the option trading every 10 s from then until 15:31."""

    def build(**options):
        return make_fake_openalgo(**{**options, "gtt_history": history})

    world = expiry_world(
        build,
        monkeypatch,
        tmp_path,
        symbol=OPTION,
        day=MONDAY,
        start=START,
        analyzer=analyzer,
        expiry=OPTION_DAY,
    )
    assert world.run(OptionKept, **LIMITS).stopped_by == "session_end"
    fake, script = world.fake, world.script
    script.sleep((ist(MONDAY, 15, 35) - fake.clock.now()) / NS)
    fake.end_day()
    tuesday = ist(OPTION_DAY, *START)
    fake.seed_candles(OPTION, "NFO", flat_candles(OPTION_DAY, ist(OPTION_DAY, *START[:2])))
    script.sleep((tuesday - fake.clock.now()) / NS)

    def trade():
        fake.trade(OPTION, "NFO", PRICE, volume=LOT)

    trade()
    for instant in range(tuesday + TRADE_EVERY_S * NS, ist(OPTION_DAY, 15, 31), TRADE_EVERY_S * NS):
        script.at(instant, trade)
    world.stop, world.feed = StopFlag(), None
    world.stop_at(ist(OPTION_DAY, 15, 14))
    return world


@MODES
@BOOKS
def test_an_option_carried_into_its_last_day_exits_at_the_square_off_its_gtt_cancelled_first(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_to_its_last_day(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )
    caplog.clear()

    outcome = world.run(OptionKept, **LIMITS)

    (trade,) = session_result(outcome).results["OptionKept"].trades.itertuples()
    assert trade.exit_reason == "expiry_exit"
    assert (trade.entry_units, trade.exit_price_rupees) == (LOT, PRICE)
    exited = pd.Timestamp(trade.exit_ts, tz="UTC").tz_convert("Asia/Kolkata")
    assert (exited.date(), exited.strftime("%H:%M")) == (OPTION_DAY, "15:10")
    # Restored with Monday's stop GTT (no second place), which is cancelled before the exit.
    tuesday = ist(OPTION_DAY, *START)
    assert [c.what for c in calls(world, *SENDS) if c.ts_ns >= tuesday] == [
        "cancelgttorder",
        "placeorder",
    ]
    assert [(o.action, o.price_type, o.quantity, o.status) for o in world.orders()] == [
        ("SELL", "MARKET", LOT, "complete")
    ]
    assert world.fake.position(OPTION, "NFO") == 0
    assert alerts(caplog, "error") == []


@MODES
@BOOKS
def test_an_entry_after_the_carried_options_exit_is_refused_contract_expiring(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_to_its_last_day(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )

    outcome = world.run(OptionKept, **LIMITS)

    assert outcome.plans["OptionKept"].strategy.refused == ["contract_expiring"]
    assert [o.action for o in world.orders()] == ["SELL"]
    assert outcome.open_positions == ()
