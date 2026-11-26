"""A daily strategy's code raising at its decision in `bt.live`, beside another daily
strategy, on the fake OpenAlgo (design 20-positional-live section 8, 16-session-design 3.5;
review round 2 of unit N5b): the safe stop that follows cancels every entry decided at this
start before it reaches the broker, at a start before the open as after it. So the
strategies after the one that raised, in the order given, do not decide: their bars stay
undecided, and a later start of the day decides them. One that decided before it is named
in a warning with the orders the stop withdrew, whatever the clock.

The market is `daily_samples`', with INFY beside SBIN (its daily candles of Thursday and
Friday, `session_samples.rig`'s). Expected values come from the scenario.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from daily_samples import UNITS, Recorded, close_of, daily_world
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES
from restart_samples import restart
from session_samples import alerts, ist
from test_session_daily import buys, journal

from openalgo.backtest._live._types import LOGGER_NAME

FRIDAY = dt.date(2026, 11, 13)
#: A start before NSE's open (09:15 IST) and one after it, IST (hour, minute, second).
START_TIMES = pytest.mark.parametrize(
    "start", [(9, 5, 0), (9, 20, 5)], ids=["before_the_open", "after_the_open"]
)
#: The warning naming Buyer's entry the stop withdrew unsent.
WITHDRAWN = (
    "Buyer: the stop cancels the orders of its decision on its D bar closing on 2026-11-13 "
    "(BUY 10 MARKET) before they went to the broker, and that bar is not decided again today"
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class Buyer(Recorded):
    """Recorded: buys 10 SBIN at its first decision."""

    decided: list[tuple[int, int, int]] = []


class RaisesOnInfy(Recorded):
    """A daily CNC strategy on INFY whose code raises at each decision, once recorded."""

    symbol = "INFY"
    decided: list[tuple[int, int, int]] = []

    def on_bar(self, bar):
        type(self).decided.append((bar.ts_open, bar.ts_close, len(self.close)))
        raise RuntimeError("the strategy's own error")


def two_daily(make_fake_openalgo, monkeypatch, tmp_path, *, start, analyzer):
    """`daily_samples.daily_world` with INFY, both strategies' records cleared."""
    for strategy in (Buyer, RaisesOnInfy):
        strategy.decided.clear()
    return daily_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        start=start,
        analyzer=analyzer,
        symbols=("SBIN", "INFY"),
    )


@MODES
@START_TIMES
def test_an_entry_decided_before_another_strategy_raises_is_named_as_withdrawn(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, start
):
    world = two_daily(make_fake_openalgo, monkeypatch, tmp_path, start=start, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run([Buyer, RaisesOnInfy])

    assert [bar_close for _, bar_close, _ in Buyer.decided] == [close_of(FRIDAY)]
    assert world.orders() == []
    assert f"Warning: {WITHDRAWN}" in world.lines
    (error,) = alerts(caplog, "error")
    assert error.endswith(
        "RaisesOnInfy raised RuntimeError: the strategy's own error. The session stops safely"
    )


@MODES
@START_TIMES
def test_a_strategy_after_one_that_raises_does_not_decide_and_a_later_start_decides_it(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, start
):
    world = two_daily(make_fake_openalgo, monkeypatch, tmp_path, start=start, analyzer=analyzer)

    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run([RaisesOnInfy, Buyer])

    assert Buyer.decided == []
    assert world.orders() == []
    assert not [line for line in world.lines if "before they went to the broker" in line]
    with journal(world) as kept:
        assert kept.decided_bar("Buyer") is None

    restart(world, [RaisesOnInfy, Buyer], stop_ns=ist(9, 40))

    # The raising strategy's bar was decided at the first start and is not decided again.
    assert [bar_close for _, bar_close, _ in RaisesOnInfy.decided] == [close_of(FRIDAY)]
    assert [bar_close for _, bar_close, _ in Buyer.decided] == [close_of(FRIDAY)]
    assert buys(world) == [("MARKET", UNITS, "CNC", "complete")]
