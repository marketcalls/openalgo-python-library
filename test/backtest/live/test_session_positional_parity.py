"""Parity of a live CNC leg held by a stop GTT with a backtest of the bars its strategy
decided on (15-phase2-plan section 4; 20-positional-live 2 and 3; the lead's ruling N3b-2:
"compare each closed trade with a bt.backtest of the same bars and fills where the path
allows it"; review finding 6 a of unit N3b).

The session runs on the fake OpenAlgo under the recorder (`session_recorder`), as
`test_session_parity` does: SBIN on NSE from 14:40:05 IST, trading 10 units every 10 s at
500.00 until a test moves it. Each strategy buys 10 SBIN CNC at its first decision
(14:42:01, filled at 500.00) and never trades again. The backtest runs the same strategy
with the same settings on the bars its series held, from the first slice the session
decided on. In each scenario the two runs close the same trades on the same slices for
the same reasons:

- **The stop GTT fires.** SBIN falls to 496.00 at 14:44:20: the GTT's trigger (497.00)
  is crossed and its child, a LIMIT sell at 492.00, fills at the market's 496.00 live;
  the backtest's leg stop closes the trade in the same slice, reason "leg_sl".
- **The target.** SBIN rises to 503.00 at 14:44:20, past the 502.00 target: live, the
  stop GTT is cancelled and the exit sold at 503.00; the backtest's target closes it in
  the same slice, reason "leg_target".
- **A partial target.** Two targets, half each at 502.00 and 504.00: SBIN at 502.50 from
  14:44:20 takes the first 5 units (the GTT cut to 5 first), and at 505.00 from 14:46:20
  the other 5.

Live fills are at the broker's prices with no modelled slippage, so prices are compared
by hand for the live run only (the backtest's fills are its own model's).
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import CncHold, CncTarget, CncTwoTargets, gtt_world
from session_recorder import record_sessions
from session_samples import CAPITAL, DAY, MAX_DAILY_LOSS, ist
from stops_samples import at_price

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

IST = "Asia/Kolkata"
#: What each trade is compared by.
COLUMNS = ("entry_time", "exit_time", "direction", "entry_units", "max_units", "exit_reason")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def ist_time(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(ts_ns, tz="UTC").tz_convert(IST)


def rows(frame, *columns):
    return [tuple(row) for row in frame.loc[:, list(columns)].itertuples(index=False)]


def both_runs(make_fake_openalgo, monkeypatch, tmp_path, strategy, moves):
    """(the live session's Result, the backtest's Result of the bars its strategy held) of
    `strategy`, SBIN moved to each (instant, price) of `moves`; Ctrl+C at 14:49 keeps
    what is open."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    for instant, price in moves:
        at_price(world, instant, price)
    recorder = record_sessions(monkeypatch)
    world.stop_at(ist(14, 49))
    outcome = world.run(strategy)
    (record,) = recorder.finish()
    name = strategy.__name__
    bars = record.bars(name)
    first = ist_time(record.decisions_from(name)) - pd.Timedelta(minutes=1)
    backtest = bt.backtest(
        strategy,
        data=bars,
        capital=CAPITAL,
        max_daily_loss=MAX_DAILY_LOSS,
        start=first.tz_localize(None),
        end=DAY,
    )
    return session_result(outcome).results[name], backtest


def test_a_stop_gtt_that_fires_closes_the_same_trade_as_the_backtests_leg_stop(
    make_fake_openalgo, monkeypatch, tmp_path
):
    live, backtest = both_runs(
        make_fake_openalgo, monkeypatch, tmp_path, CncHold, [(ist(14, 44, 20), 496.0)]
    )

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    assert list(live.trades["exit_reason"]) == ["leg_sl"]
    # The child, a LIMIT sell at 492.00, filled at the market's 496.00.
    assert list(live.trades["exit_price_rupees"]) == [496.0]


def test_a_target_closes_the_same_trade_as_the_backtests_target(
    make_fake_openalgo, monkeypatch, tmp_path
):
    live, backtest = both_runs(
        make_fake_openalgo, monkeypatch, tmp_path, CncTarget, [(ist(14, 44, 20), 503.0)]
    )

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    assert list(live.trades["exit_reason"]) == ["leg_target"]
    assert list(live.trades["exit_price_rupees"]) == [503.0]


def test_a_partial_target_closes_the_same_units_on_the_same_slices_as_the_backtest(
    make_fake_openalgo, monkeypatch, tmp_path
):
    moves = [(ist(14, 44, 20), 502.5), (ist(14, 46, 20), 505.0)]
    live, backtest = both_runs(make_fake_openalgo, monkeypatch, tmp_path, CncTwoTargets, moves)

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    sold = live.fills[live.fills["side"] == "SELL"]
    columns = ("time", "quantity_units", "reason")
    assert rows(sold, *columns) == rows(backtest.fills[backtest.fills["side"] == "SELL"], *columns)
    # 5 units at 502.50, then 5 at 505.00, as the broker filled them.
    assert list(sold["quantity_units"]) == [5, 5]
    assert list(sold["price_rupees"]) == [502.5, 505.0]
