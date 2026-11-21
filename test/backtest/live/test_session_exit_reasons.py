"""The reason of a closing order named in its EXIT alert and in the event of its request
(`_live/_exit_reasons.py`; 2.4.0 item M11-X3): an audit tells a target from a strategy's
exit, the square-off or the kill switch without reading the kernel's records.

The market and the scripts are `session_samples`' and `watch_samples`': the session starts
at 14:40:05 IST on Monday 2026-11-16 and SBIN trades 10 units every 10 s, five seconds past
each tenth second, at 500.00 unless a test moves it. The fake fills a MARKET order at the
last trade's price. Every alert is echoed on the console as "KIND: text".
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import BuyAndHold, Hold, ist, rig
from stops_samples import at_price
from watch_samples import HoldInfy, HoldSbin, HoldWithTarget

from openalgo.backtest._live._exit_reasons import EXIT_REASON_WORDS
from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


def console(world, kind: str) -> list[str]:
    """The console's lines of the alerts of `kind` ("ENTRY", "EXIT"), after the kind."""
    prefix = f"{kind}: "
    return [line[len(prefix) :] for line in world.lines if line.startswith(prefix)]


def placements(world) -> list[dict]:
    """The event of each order placement the session handed to its order worker, in
    order."""
    return [
        event
        for run in world.trace.runs
        for event in run.events
        if event["kind"] == "request" and event["type"] == "PlaceRequest"
    ]


def test_an_exit_at_a_legs_target_names_the_target(world):
    # HoldWithTarget buys 10 at 500.00 with a target 5 points above the fill; the 14:44:35
    # trade at 505.00 reaches it and the engine sells the 10 at 505.00.
    at_price(world, ist(14, 44, 30), 505.0)
    world.stop_at(ist(14, 47, 30))
    live(world, HoldWithTarget)
    [entry, sale] = placements(world)
    assert (sale["purpose"], sale["reason"]) == ("exit", "leg_target")
    assert "reason" not in entry
    [exit_line] = console(world, "EXIT")
    assert exit_line.startswith("SELL 10 SBIN NSE at 505.00 (exit: target level, order ")


def test_a_strategys_own_exit_names_the_strategy(world):
    # BuyAndHold buys 10 at its first decision and sells them three bars later itself.
    world.stop_at(ist(14, 47, 30))
    live(world, BuyAndHold)
    [entry, sale] = placements(world)
    assert (sale["purpose"], sale["reason"]) == ("exit", "signal")
    [entry_line] = console(world, "ENTRY")
    [exit_line] = console(world, "EXIT")
    assert entry_line.startswith("BUY 10 SBIN NSE at 500.00 (entry, order ")
    assert exit_line.startswith("SELL 10 SBIN NSE at 500.00 (exit: strategy exit, order ")


def test_the_square_off_names_the_square_off_time(world):
    # Hold never sells: the square-off at 15:00 sells its 10, with a trade half a second
    # into 15:00 so the sale fills before the session ends.
    world.script.at(ist(15, 0) + 500_000_000, world.script.trade)
    live(world, Hold)
    sale = placements(world)[-1]
    assert (sale["purpose"], sale["reason"]) == ("square_off", "scheduler")
    [exit_line] = console(world, "EXIT")
    assert "(square_off: square-off time, order " in exit_line


def test_the_kill_switchs_exits_name_the_accounts_daily_loss_limit(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # HoldSbin and HoldInfy buy 10 at 500.00 each; both fall to 492.00 from 14:43:40,
    # where their loss of Rs 160 is past the account's limit of Rs 150, and the kill
    # switch sells both.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    made.script.price = lambda now_ns: 492.0 if now_ns >= ist(14, 43, 40) else 500.0
    made.stop_at(ist(14, 47))
    seam(made, monkeypatch)
    live(made, [HoldSbin, HoldInfy], account_max_daily_loss=150.0, costs=None)
    sales = [event for event in placements(made) if event["purpose"] == "risk_exit"]
    assert [event["reason"] for event in sales] == ["account_loss_limit"] * 2
    for line in console(made, "EXIT"):
        assert "(risk_exit: kill switch at the account daily loss limit, order " in line


def test_every_exit_reason_of_the_kernel_has_words():
    # The kernel's ExitReason codes (rust/oa_bt/src/rms/vocabulary.rs), every one of which
    # a live session's order may carry.
    codes = {
        "leg_sl",
        "leg_target",
        "time_exit",
        "overall_sl",
        "overall_target",
        "lock_profit",
        "daily_loss_limit",
        "scheduler",
        "manual",
        "eod",
        "drawdown_halt",
        "margin_liquidation",
        "expiry",
        "signal",
        "account_loss_limit",
        "daily_profit_target",
        "account_profit_target",
    }
    assert set(EXIT_REASON_WORDS) == codes
