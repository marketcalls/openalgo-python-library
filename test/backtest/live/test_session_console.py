"""What a live session prints on the operator's console, on the fake OpenAlgo
(`_live/_shared_alerts.py`, `_live/_session_banner.py`, `_live/_session_loop.py`;
16-session-design 2.8 and 3.1 step 13, the sandbox smoke run of 2026-11-17, unit 13).

Every alert is echoed on the console, whether or not it reaches Telegram, so a session
without an OpenAlgo username is not silent once it trades; a line printed with its alert
is printed once. The banner states the rule of broker-held stops once, and the session
warns at the first position actually opened with no stop at the broker, never for a
strategy whose entries carry one. A warning carries one prefix.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16, SBIN trades at 500.00 every 10 s. `BuyAndHold` buys 10 SBIN at 14:42:01
and sells them three bars later; `StopHold` buys 10 with a stop 3 points below.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import BuyAndHold, Hold, ist, rig
from stops_samples import StopHold

from openalgo.backtest._live._alerts_logs import SessionLogs
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import LiveError

#: What the preflight warns of a SANDBOX session without an OpenAlgo username.
NO_ALERTS = "this SANDBOX session runs without alerts"
#: The start of the warning given at the first position opened with no stop.
NO_STOP = "a position of 10 SBIN NSE was opened with no stop at the broker"
#: The words of the alert and the warning of an event stream that cannot be written.
NO_EVENTS = "the session's event stream cannot be written"


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


@pytest.fixture
def sandbox(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)


def starting(lines: list[str], prefix: str) -> list[str]:
    return [line for line in lines if line.startswith(prefix)]


# ----- every alert on the console ------------------------------------------------------


def test_without_a_telegram_user_the_fills_are_echoed_on_the_console(sandbox):
    sandbox.run(BuyAndHold, alerts_to=None)
    [entry] = starting(sandbox.lines, "ENTRY: ")
    [exit_] = starting(sandbox.lines, "EXIT: ")
    assert entry.startswith("ENTRY: BUY 10 SBIN NSE at 500.00 (entry, order ")
    assert exit_.startswith("EXIT: SELL 10 SBIN NSE at 500.00 (exit, order ")


def test_without_a_telegram_user_the_daily_summary_is_echoed_on_the_console(sandbox):
    sandbox.run(BuyAndHold, alerts_to=None)
    [summary] = starting(sandbox.lines, "DAILY SUMMARY: ")
    assert summary.startswith(
        "DAILY SUMMARY: Session of 2026-11-16 (IST): SANDBOX mode, broker zerodha\n"
    )
    assert "Trades      : 1 closed" in summary


def test_with_a_telegram_user_the_alerts_are_echoed_too(world):
    world.run(BuyAndHold)
    assert len(starting(world.lines, "ENTRY: BUY 10 SBIN NSE at 500.00")) == 1


def test_the_banners_alert_is_not_echoed_again(world):
    world.stop_at(ist(14, 41))
    world.run(BuyAndHold)
    assert starting(world.lines, "SESSION: ") == []
    assert world.lines.count("OpenAlgo is in LIVE mode: orders go to zerodha") == 1


def test_a_line_printed_with_its_alert_is_printed_once(world):
    # Hold has no stop: with on_stop="keep" its leg is left open, one line and one alert.
    world.stop_at(ist(14, 44))
    world.run(Hold, on_stop="keep")
    assert len([line for line in world.lines if "Hold SBIN NSE 10 units: " in line]) == 1


# ----- entries without a stop -----------------------------------------------------------


def test_a_strategy_whose_entries_carry_a_stop_is_never_warned_of_entries_without_one(world):
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")
    warned = [line for line in world.lines if line.startswith("Warning: ")]
    assert not any("stop=" in line or NO_STOP in line for line in warned)
    assert not any("no stop at the broker" in warning for warning in outcome.warnings)


def test_the_first_position_opened_without_a_stop_is_warned_on_the_console(world):
    world.stop_at(ist(14, 44))
    world.run(Hold, on_stop="keep")
    [line] = [line for line in world.lines if NO_STOP in line]
    assert line.startswith(f"Warning: Hold: {NO_STOP}")
    assert world.lines.index(line) > world.lines.index(
        "OpenAlgo is in LIVE mode: orders go to zerodha"
    )


def test_the_banner_states_the_rule_of_broker_held_stops_once(world):
    world.stop_at(ist(14, 41))
    world.run(BuyAndHold)
    assert len(starting(world.lines, "Stops: an entry placed with stop= gets its stop")) == 1


# ----- one prefix per warning ------------------------------------------------------------


def test_no_console_line_carries_the_warning_prefix_twice(sandbox):
    sandbox.stop_at(ist(14, 41))
    sandbox.run(BuyAndHold, alerts_to=None)
    assert [line for line in sandbox.lines if line.lower().startswith("warning: warning")] == []
    assert len([line for line in sandbox.lines if NO_ALERTS in line]) == 1


def test_the_results_warnings_carry_no_prefix(sandbox):
    sandbox.stop_at(ist(14, 41))
    outcome = sandbox.run(BuyAndHold, alerts_to=None)
    [no_alerts] = [warning for warning in outcome.warnings if NO_ALERTS in warning]
    assert not no_alerts.lower().startswith("warning")


def test_a_warning_sent_as_an_alert_too_is_printed_once(world, monkeypatch):
    # The event stream fails at the entry's fill, after the banner: one ERROR alert and
    # one warning of the result, with the same words, and one console line.
    write = SessionLogs.event

    def failing(logs, kind, **fields):
        if kind == "fill":
            raise LiveError("disk full")
        write(logs, kind, **fields)

    monkeypatch.setattr(SessionLogs, "event", failing)
    world.stop_at(ist(14, 44))
    outcome = world.run(BuyAndHold)
    assert len([warning for warning in outcome.warnings if NO_EVENTS in warning]) == 1
    assert len([line for line in world.lines if NO_EVENTS in line]) == 1
