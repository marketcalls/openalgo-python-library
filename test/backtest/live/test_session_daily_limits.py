"""The daily limits of a live session on the fake OpenAlgo (18-multi-strategy 2.5a): the
strategy's trade cap and daily profit target held by its kernel, the account's trade cap
set before each strategy's decisions and its profit target firing the kill switch, a
restart that keeps the day's count and the profit lock, and the start banner.

The market is `session_samples`': SBIN and INFY on NSE on Monday 2026-11-16 from 14:40:05
IST, trading 10 units every 10 s at the script's price (500.00 unless a test says
otherwise). `HoldSbin` and `HoldInfy` buy 10 units at their first decision (14:42:01) and
again at any decision at which they hold none; `Flipper` buys 10 SBIN at one decision and
sells them at the next. Each strategy's own daily loss limit is Rs 5,000, so it never
acts. Nothing contacts a server. Expected values come from the scenario and the fake's
books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import StopWhenFlat, crash_at, crashing, restart, run_to_crash
from session_samples import alerts, ist, rig
from shared_samples import B_UNITS, SbinBreakout, SbinTrend, sbin_orders
from stops_samples import at_price, stop_world
from watch_samples import Flipper, HoldInfy, HoldSbin, events, run

from openalgo.backtest._live._types import LOGGER_NAME

#: From this instant SBIN and INFY trade at `RISEN`: each strategy's 10 units are then up
#: 10 x 8.00 = 80.00.
RISE_AT = ist(14, 43, 40)
RISEN = 508.0


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.stop_at(ist(14, 47))
    return world


def rising(world) -> None:
    world.script.price = lambda now_ns: RISEN if now_ns >= RISE_AT else 500.0


def bought(world, symbol: str = "SBIN", after_ns: int = 0) -> list:
    return [
        o
        for o in world.orders()
        if o.symbol == symbol and o.action == "BUY" and o.placed_ns > after_ns
    ]


def reject_codes(outcome, name: str) -> list:
    return [code for code in outcome.outputs[name]["orders"]["reject_code"] if code]


# ----- the strategy's own limits, held by its kernel -----------------------------------


def test_the_strategys_trade_cap_refuses_the_entry_past_it(world):
    # Flipper buys at 14:42:01, sells at 14:43:01, and would buy again at 14:44:01: with a
    # cap of 1 that entry is refused at submission, so the fake sees one buy.
    outcome = run(world, Flipper, max_trades_per_day=1)
    assert len(bought(world)) == 1
    assert "trade_limit" in reject_codes(outcome, "Flipper")


def test_the_strategys_profit_target_exits_on_the_tick_and_locks_its_entries(world):
    # 10 SBIN bought at 500.00; the first trade at 508.00 (14:43:40) puts the day at
    # +80.00, past a target of 50.00: the kernel exits on that tick (a risk exit) and the
    # strategy, flat, is refused daily_profit_gate at its later decisions.
    rising(world)
    outcome = run(world, HoldSbin, daily_profit_target=50)
    assert outcome.outputs["HoldSbin"]["trades"]["exit_reason"] == ["daily_profit_target"]
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert len(sells) == 1 and RISE_AT <= sells[0].placed_ns < ist(14, 44)
    assert bought(world, after_ns=RISE_AT) == []
    assert "daily_profit_gate" in reject_codes(outcome, "HoldSbin")
    kernel_events = outcome.outputs["HoldSbin"]["events"]
    hits = [e for e in kernel_events if e.get("name") == "daily_profit_hit"]
    assert [(hit["level_rupees"], hit["observed_rupees"]) for hit in hits] == [(50.0, 80.0)]


# ----- the account's limits ----------------------------------------------------------------


def test_the_accounts_trade_cap_refuses_the_second_strategys_entry(world):
    # A cap of 1 entry for the session: HoldSbin decides first at 14:42:01 and buys;
    # HoldInfy, deciding after it, finds the account's count at the cap at every decision.
    outcome = run(world, [HoldSbin, HoldInfy], account_max_trades_per_day=1)
    assert len(bought(world, "SBIN")) == 1
    assert bought(world, "INFY") == []
    assert set(reject_codes(outcome, "HoldInfy")) == {"account_trade_limit"}
    assert reject_codes(outcome, "HoldSbin") == []


def test_the_accounts_profit_target_fires_the_kill_switch_of_every_strategy(world, caplog):
    # Both up 80.00 from 14:43:40: the account's 160.00 is past a target of 150.00. The
    # kill switch of each strategy fires for account_profit, exits it and locks it.
    rising(world)
    outcome = run(world, [HoldSbin, HoldInfy], account_daily_profit_target=150)
    fired = [(e["strategy"], e["reason"]) for e in events(outcome, "kill_switch")]
    assert fired == [("HoldSbin", "account_profit"), ("HoldInfy", "account_profit")]
    for name in ("HoldSbin", "HoldInfy"):
        assert outcome.outputs[name]["trades"]["exit_reason"] == ["account_profit_target"]
    assert bought(world, "SBIN", RISE_AT) == [] and bought(world, "INFY", RISE_AT) == []
    words = [text for text in alerts(caplog, "error") if "account's daily profit" in text]
    assert len(words) == 1
    assert (world.fake.position("SBIN", "NSE"), world.fake.position("INFY", "NSE")) == (0, 0)


# ----- the start banner ----------------------------------------------------------------


def test_the_banner_names_the_daily_limits_and_the_order_rate(world):
    run(
        world,
        [HoldSbin, HoldInfy],
        max_trades_per_day={"HoldSbin": 3, "HoldInfy": 2},
        daily_profit_target=2_000,
        account_max_trades_per_day=4,
        account_daily_profit_target=3_000,
        max_orders_per_second=6,
    )
    lines = world.lines
    assert any(
        "HoldSbin daily limits: at most 3 trades a day, profit target Rs 2,000.00" in line
        for line in lines
    )
    assert any("HoldInfy daily limits: at most 2 trades a day" in line for line in lines)
    assert any("Account trade cap: 4 trades a day" in line for line in lines)
    assert any("Account daily profit target: Rs 3,000.00" in line for line in lines)
    assert any("Order rate: at most 6 order calls a second" in line for line in lines)


def test_a_session_without_daily_limits_names_none_but_the_order_rate(world):
    run(world, HoldSbin)
    assert not [line for line in world.lines if "daily limits" in line or "Account trade" in line]
    assert any("Order rate: at most 8 order calls a second" in line for line in world.lines)


def test_the_daily_summary_names_the_daily_limits(world):
    outcome = run(
        world,
        [HoldSbin, HoldInfy],
        max_trades_per_day={"HoldSbin": 3, "HoldInfy": 2},
        account_daily_profit_target=3_000,
    )
    assert outcome.summary.limits == (
        "HoldSbin at most 3 trades a day",
        "HoldInfy at most 2 trades a day",
        "account daily profit target Rs 3,000.00",
    )


# ----- a restart keeps the count and the lock (2.5a rule 8) ------------------------------

CRASHED = ist(14, 42, 30)


@pytest.fixture
def stops(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    return world


def test_a_restart_keeps_the_days_trade_count(stops):
    # StopWhenFlat buys 10 with its stop at 14:42:01 (the day's one entry under a cap of
    # 1), the process dies at 14:42:30, the stop fills while it is down; the restarted
    # strategy, flat, is refused trade_limit instead of buying again.
    crash_at(stops, CRASHED)
    run_to_crash(stops, StopWhenFlat, max_trades_per_day=1)
    stops.fake.trade("SBIN", "NSE", 496.0, volume=10.0)
    outcome = restart(
        stops, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep", max_trades_per_day=1
    )
    assert [o for o in stops.orders() if o.action == "BUY" and o.placed_ns > CRASHED] == []
    assert "trade_limit" in reject_codes(outcome, "StopWhenFlat")


def test_a_restart_keeps_the_profit_lock(stops):
    # 10 SBIN bought at 500.00 at 14:42:01; SBIN at 503.00 from 14:42:10 puts the day at
    # +30.00, past a target of 20.00: the position exits and entries lock. After a crash
    # at 14:43 the restarted strategy is still refused daily_profit_gate.
    at_price(stops, ist(14, 42, 10), 503.0)
    crash_at(stops, ist(14, 43))
    run_to_crash(stops, StopWhenFlat, daily_profit_target=20)
    assert stops.fake.position("SBIN", "NSE") == 0
    outcome = restart(
        stops, StopWhenFlat, stop_ns=ist(14, 46), on_stop="keep", daily_profit_target=20
    )
    assert [o for o in stops.orders() if o.action == "BUY" and o.placed_ns > ist(14, 43)] == []
    assert "daily_profit_gate" in reject_codes(outcome, "StopWhenFlat")


# ----- a shared instrument: each member's limits are its own (2.5a rule 10) ---------------


def test_a_members_profit_target_exits_its_own_units_and_leaves_the_others(stops):
    # A (SbinTrend) holds 100 SBIN and B (SbinBreakout) 50, both from 500.00 with stops at
    # 497.00. At 501.50 from 14:43 A is up 150.00, past its target of 100.00: A exits its
    # 100 as a closing order of the group; B keeps its 50 and its stop.
    at_price(stops, ist(14, 43), 501.5)
    stops.stop_at(ist(14, 46))
    outcome = stops.run(
        [SbinTrend, SbinBreakout],
        daily_profit_target={"SbinTrend": 100, "SbinBreakout": 10_000},
        on_stop="keep",
    )
    assert outcome.outputs["SbinTrend"]["trades"]["exit_reason"] == ["daily_profit_target"]
    assert outcome.outputs["SbinBreakout"]["trades"]["exit_reason"] == []
    assert stops.fake.position("SBIN", "NSE") == B_UNITS
    working = [o for o in sbin_orders(stops, "SbinBreakout") if o.status == "trigger pending"]
    assert [o.quantity for o in working] == [B_UNITS]
