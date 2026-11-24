"""The day's figures of a live session across its restarts (`_live/_session_figures.py`,
`_live/_session_saved.py`, `_live/_session_close.py`; 16-session-design 1.4 and 2.5 F,
unit 13): each start saves its running figures with the state they describe, and a
restart adds the earlier starts' to its own, counting nothing twice.

The session tests compare a day split by a restart with the same day run unbroken, on the
fake OpenAlgo: `TwoRoundTrips` buys 10 SBIN on the bars closing at 14:42 and 14:48 IST and
sells them on those closing at 14:45 and 14:49, at the prices the script gives (500.00,
then 505.00 from 14:44:30, then 510.00 from 14:48:30): two round trips of 50 rupees gross
each, whichever start makes them. Expected values come from the fake's fills and the
unbroken run.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import ist, rig
from stops_samples import at_price

import openalgo.backtest as bt
from openalgo.backtest._live._alerts_summary import summary_text
from openalgo.backtest._live._session_figures import (
    FIGURES,
    SESSION_KEY,
    DayFigures,
    Earlier,
    SessionFigures,
    StrategyFigures,
)
from openalgo.backtest._live._session_slippage import Slippage
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._version_id import version_of
from openalgo.backtest.errors import LiveError

# ----- the figures, with stand-ins for the session --------------------------------------


def saved_state(closed: int, gross: float, charges: float) -> dict:
    """A strategy's saved state as journal L keeps it, with its figures."""
    return {
        "live_state": {},
        FIGURES: {"closed_trades": closed, "gross_pnl": gross, "charges": charges},
    }


def session_counts(rejected: int, mismatches: int, unprotected: float, undelivered: int) -> dict:
    return {
        "rejected_orders": rejected,
        "reconcile_mismatches": mismatches,
        "unprotected_seconds": unprotected,
        "undelivered_alerts": undelivered,
    }


def finished(trades: int, gross: float, charges: float) -> tuple[dict, tuple[float, float]]:
    """A kernel session's `finish()` tables with `trades` closed, and its account's gross
    P&L and charges."""
    return {"trades": {"trade_id": list(range(trades))}}, (gross, charges)


def this_start(rejected: int = 0, mismatches: int = 0, undelivered: int = 0) -> SimpleNamespace:
    """This start's own counts."""
    sender = SimpleNamespace(undelivered=undelivered)
    return SimpleNamespace(
        rejected=rejected, mismatches=mismatches, parts=SimpleNamespace(sender=sender)
    )


@pytest.mark.parametrize(
    ("saved", "running", "carried", "absent"),
    [
        ({}, ["A"], {}, StrategyFigures()),
        ({"A": saved_state(1, 50.0, 20.0)}, ["A"], {"A": StrategyFigures(1, 50.0, 20.0)}, None),
        (
            {"A": saved_state(1, 50.0, 20.0), "B": saved_state(2, -30.0, 15.0)},
            ["A"],
            {"A": StrategyFigures(1, 50.0, 20.0)},
            StrategyFigures(2, -30.0, 15.0),
        ),
        ({"A": {"live_state": {}}}, ["A"], {"A": StrategyFigures()}, None),
    ],
    ids=[
        "a first start",
        "a restart of the same strategy",
        "a strategy this start does not run",
        "a state saved without figures",
    ],
)
def test_a_start_reads_what_the_earlier_starts_saved(saved, running, carried, absent):
    figures = DayFigures.at_start(saved, running)
    assert figures.carried == carried
    assert figures.absent == (absent or StrategyFigures())


def test_a_start_reads_the_sessions_own_counts_apart_from_the_strategies():
    saved = {"A": saved_state(1, 50.0, 20.0), SESSION_KEY: session_counts(1, 2, 3.5, 4)}
    figures = DayFigures.at_start(saved, ["A"])
    assert figures.session == SessionFigures(1, 2, 3.5, 4)
    assert SESSION_KEY not in figures.carried


@pytest.mark.parametrize(
    "saved",
    [
        {"A": {FIGURES: {"closed_trades": 1.5}}},
        {"A": {FIGURES: {"closed_trades": -1}}},
        {"A": {FIGURES: {"gross_pnl": "50"}}},
        {"A": {FIGURES: ["closed_trades", 1]}},
        {SESSION_KEY: {"unprotected_seconds": float("nan")}},
    ],
    ids=["a part of a trade", "a negative count", "a figure in words", "no record", "not finite"],
)
def test_figures_that_cannot_be_read_refuse_the_start(saved):
    with pytest.raises(LiveError, match="figures"):
        DayFigures.at_start(saved, ["A"])


def test_a_restored_engines_gross_and_charges_are_the_days_and_its_trades_are_added():
    # The kernel restores the account, so the engine's gross P&L and charges hold the
    # earlier start's; the round trips it closed are this start's only.
    figures = DayFigures.at_start({"A": saved_state(1, 50.0, 20.0)}, ["A"])
    traded, _ = figures.day(this_start(), {"A": finished(1, 100.0, 40.0)}, 0.0, slippages={})
    assert traded == StrategyFigures(2, 100.0, 40.0)


def test_a_strategy_this_start_does_not_run_adds_its_saved_figures_whole():
    saved = {"A": saved_state(1, 50.0, 20.0), "B": saved_state(2, -30.0, 15.0)}
    figures = DayFigures.at_start(saved, ["A"])
    traded, _ = figures.day(this_start(), {"A": finished(0, 50.0, 20.0)}, 0.0, slippages={})
    assert traded == StrategyFigures(3, 20.0, 35.0)


def saved_slippage(rupees: float, not_compared: int, compared: int) -> dict:
    """A strategy's saved state whose figures hold a slippage and its counts only."""
    figures = {
        "slippage": rupees,
        "slippage_not_compared": not_compared,
        "slippage_compared": compared,
    }
    return {"live_state": {}, FIGURES: figures}


@pytest.mark.parametrize(
    ("saved", "own", "shown"),
    [
        ((0.0, 1, 0), Slippage(0.0, 1, 0), None),
        ((-1.0, 0, 1), Slippage(0.0, 1, 0), -1.0),
        ((0.0, 1, 0), Slippage(1.5, 0, 1), 1.5),
        ((1.0, 0, 1), Slippage(-1.0, 0, 1), 0.0),
        ((0.0, 0, 0), Slippage(), 0.0),
    ],
    ids=[
        "none compared in either start",
        "one compared before the restart",
        "one compared after it",
        "compared fills that net to nothing",
        "no MARKET fill",
    ],
)
def test_the_days_slippage_is_missing_only_when_no_market_fill_was_compared(saved, own, shown):
    figures = DayFigures.at_start({"A": saved_slippage(*saved)}, ["A"])
    finish = {"A": finished(0, 0.0, 0.0)}
    traded, _ = figures.day(this_start(), finish, 0.0, slippages={"A": own})
    assert traded.summary_slippage == shown


def test_the_sessions_counts_add_the_earlier_starts_to_this_ones():
    figures = DayFigures.at_start({SESSION_KEY: session_counts(1, 2, 3.5, 4)}, ["A"])
    _, counted = figures.day(this_start(2, 1, 1), {"A": finished(0, 0.0, 0.0)}, 1.5, slippages={})
    assert counted == SessionFigures(3, 3, 5.0, 5)


# ----- a day split by a restart against the same day unbroken ---------------------------

BUY_AT = (ist(14, 42), ist(14, 48))
SELL_AT = (ist(14, 45), ist(14, 49))
#: Each round trip's gross P&L by the script's prices: 10 units bought and sold 5 higher.
TRIP_GROSS = 50.0
#: When the day ends: after both round trips, the NSE entry cut-off (14:50) past.
DAY_ENDS = ist(14, 51)


class TwoRoundTrips(bt.Strategy):
    """Buys 10 SBIN on the bars closing at `BUY_AT`, sells them on those at `SELL_AT`."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if bar.ts_close in BUY_AT and not self.position.qty:
            self.buy(10)
        elif bar.ts_close in SELL_AT and self.position.qty:
            self.exit()


def world_of(make_fake_openalgo, monkeypatch, tmp_path):
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    return world


@pytest.fixture
def unbroken(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The day run by one start: its summary, and the gross P&L of the fake's fills."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path / "unbroken")
    world.stop_at(DAY_ENDS)
    summary = world.run(TwoRoundTrips).summary
    return summary, fills_gross(world)


def fills_gross(world) -> float:
    return sum((1 if f.action == "SELL" else -1) * f.quantity * f.price for f in world.fake.fills())


def test_the_unbroken_day_makes_both_round_trips(unbroken):
    summary, gross = unbroken
    assert (summary.closed_trades, summary.gross_pnl, gross) == (2, 2 * TRIP_GROSS, 2 * TRIP_GROSS)


def test_a_day_restarted_after_a_stop_adds_both_starts_once(
    unbroken, make_fake_openalgo, monkeypatch, tmp_path
):
    day, gross = unbroken
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path / "restarted")
    world.stop_at(ist(14, 46, 30))
    first = world.run(TwoRoundTrips).summary
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert (first.closed_trades, first.gross_pnl) == (1, TRIP_GROSS)
    assert (summary.closed_trades, summary.gross_pnl) == (2, gross)
    assert summary.charges == pytest.approx(day.charges)


@pytest.mark.parametrize(
    "crashed_at",
    [ist(14, 45, 30), ist(14, 46, 30)],
    ids=["the sale filled after the last save", "after a save that holds the round trip"],
)
def test_a_day_restarted_after_a_crash_counts_each_round_trip_once(
    unbroken, make_fake_openalgo, monkeypatch, tmp_path, crashed_at
):
    day, gross = unbroken
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path / "crashed")
    crashing(monkeypatch)
    crash_at(world, crashed_at)
    run_to_crash(world, TwoRoundTrips)
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert (summary.closed_trades, summary.gross_pnl) == (2, gross)
    assert summary.charges == pytest.approx(day.charges)


def test_an_order_refused_before_a_restart_is_counted_in_the_days_summary(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path)
    world.fake.reject_next("RMS: blocked", price_type="MARKET", at_placement=True)
    world.stop_at(ist(14, 46, 30))
    first = world.run(TwoRoundTrips).summary
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert (first.rejected_orders, first.closed_trades) == (1, 0)
    assert (summary.rejected_orders, summary.closed_trades) == (1, 1)


def test_an_order_refused_after_the_last_save_is_counted_after_a_ctrl_c_and_a_restart(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The entry is refused at 14:42:01; Ctrl+C comes before the next slice's save at 14:43.
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path)
    world.fake.reject_next("RMS: blocked", price_type="MARKET", at_placement=True)
    world.stop_at(ist(14, 42, 30))
    first = world.run(TwoRoundTrips).summary
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert first.rejected_orders == 1
    assert summary.rejected_orders == 1


# ----- a strategy the restart does not run ------------------------------------------------


class HoldSBIN(bt.Strategy):
    """Buys 10 SBIN on the bar closing at 14:42 IST and holds them."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if bar.ts_close == ist(14, 42) and not self.position.qty:
            self.buy(10)


class IdleINFY(bt.Strategy):
    """Trades nothing."""

    symbol, exchange, interval = "INFY", "NSE", "1m"

    def on_bar(self, bar):
        pass


def two_strategies(make_fake_openalgo, monkeypatch, tmp_path):
    """SBIN at 505.00 from 14:43:05, a minute after HoldSBIN bought at 500.00; Ctrl+C at
    14:43:30."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    at_price(world, ist(14, 43, 5), 505.0)
    world.stop_at(ist(14, 43, 30))
    return world


def test_a_strategy_the_restart_drops_after_a_flatten_adds_its_round_trip_once(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The flatten's sale fills after the last save of the slices: the close saves it.
    world = two_strategies(make_fake_openalgo, monkeypatch, tmp_path)
    first = world.run([HoldSBIN, IdleINFY]).summary
    summary = restart(world, [IdleINFY], stop_ns=ist(14, 45)).summary
    assert (first.closed_trades, first.gross_pnl) == (1, TRIP_GROSS)
    assert (summary.closed_trades, summary.gross_pnl) == (1, TRIP_GROSS)
    assert summary.charges == pytest.approx(first.charges)
    assert summary.flat


def test_the_days_summary_names_a_strategy_the_restart_drops(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = two_strategies(make_fake_openalgo, monkeypatch, tmp_path)
    world.run([HoldSBIN, IdleINFY])
    summary = restart(world, [IdleINFY], stop_ns=ist(14, 45)).summary
    strategies = next(line for line in summary_text(summary).splitlines() if "Strategies" in line)
    idle = version_of(IdleINFY, None).version_id
    assert strategies.endswith(f": IdleINFY (version {idle}); earlier today: HoldSBIN")


def test_a_position_a_dropped_strategy_kept_leaves_the_day_not_flat(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = two_strategies(make_fake_openalgo, monkeypatch, tmp_path)
    world.run([HoldSBIN, IdleINFY], on_stop="keep")
    summary = restart(world, [IdleINFY], stop_ns=ist(14, 45)).summary
    assert not summary.flat
    assert summary.earlier == ("HoldSBIN (10 units of SBIN NSE open at its last save)",)


def saved_position(units: float) -> dict:
    """A strategy's saved state whose live state holds `units` of SBIN NSE in MIS."""
    live_state = {
        "instruments": [{"symbol": "SBIN", "exchange": "NSE"}],
        "positions": [{"instrument": 0, "product": "MIS", "quantity_units": units}],
    }
    return {**saved_state(1, 50.0, 20.0), "live_state": live_state}


def test_a_start_names_the_position_a_strategy_it_does_not_run_saved():
    figures = DayFigures.at_start({"B": saved_position(10.0)}, ["A"])
    assert figures.earlier == (Earlier("B", "SBIN", "NSE", 10),)
    assert figures.earlier_open


def test_a_strategy_not_run_is_named_with_the_position_it_saved_whatever_its_product():
    saved = saved_position(0.0)
    saved["live_state"]["positions"] = [{"instrument": 0, "product": "CNC", "quantity_units": 25}]

    figures = DayFigures.at_start({"B": saved}, ["A"])

    assert figures.earlier == (Earlier("B", "SBIN", "NSE", 25),)


def test_a_saved_position_that_cannot_be_read_refuses_the_start():
    saved = {
        "B": {
            **saved_state(1, 50.0, 20.0),
            "live_state": {"positions": [{"instrument": 0, "product": "MIS"}]},
        }
    }
    with pytest.raises(LiveError, match=r"saved for B cannot be read \(its saved position\)"):
        DayFigures.at_start(saved, ["A"])
