"""A difference acted on in time and then seen late at the close, on the fake OpenAlgo
(`_live/_reconcile_findings.py`, `_live/_reconcile_ended.py`, `_live/_session_close.py`;
16-session-design 5.4, the unit 13 follow-up item 1, lead's rule of 2026-11-18).

`Watches` trades nothing. Another client of the account buys 10 SBIN at 14:40:30 IST, so
OpenAlgo's books hold 10 units the engine does not (MORE): two comparisons in time see it
and the second acts on it (the strategy's entries are locked), a mismatch of the day.
From 14:41:30 an order of the leg stays in flight past the in-flight limit, as an entry
whose answer never comes would (a test seam holds the leg in flight: no fault of the fake
keeps a flat leg whose entries are locked in flight that long), so the close's final read
at the Ctrl+C of 14:45:30 compares the leg late and sees the same difference. The close
names it (its alert, the DAILY_SUMMARY's Reconcile and End state lines, the result's
`unconfirmed_legs` and a warning) and counts it once.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist, rig
from test_session_threads import passing, run_threaded

import openalgo.backtest as bt
from openalgo.backtest._live import reconcile as reconcile_module
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

#: When the other client buys, when the leg is held in flight from, and the Ctrl+C.
BY_HAND = ist(14, 40, 30)
IN_FLIGHT_FROM = ist(14, 41, 30)
STOPPED = ist(14, 45, 30)
#: The words of the DAILY_SUMMARY and of the close's alert for the leg.
LATE_SUMMARY = (
    "Watches SBIN NSE 10 units in OpenAlgo's books against the engine's 0 at a late read, "
    "while an order of the leg was in flight past the limit, neither carried by the "
    "engine's fills since nor confirmed by a later read"
)


class Watches(bt.Strategy):
    """Trades nothing."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        return None


@pytest.fixture(params=["inline", "threads"])
def acted_then_late(request, make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The scenario of the module docstring on the seam the parameter names: calling it
    runs the session and returns its outcome."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    other = world.fake.client()
    world.script.at(
        BY_HAND,
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        ),
    )
    world.stop_at(STOPPED)
    in_flight = reconcile_module.in_flight

    def held_in_flight(session: Any, trader: Any) -> bool:
        return session.now() >= IN_FLIGHT_FROM or in_flight(session, trader)

    monkeypatch.setattr(reconcile_module, "in_flight", held_in_flight)

    def run() -> Any:
        if request.param == "threads":
            return run_threaded(world, passing(world), "orders", (Watches,), on_stop="keep")
        return world.run(Watches, on_stop="keep")

    return run


def test_the_difference_was_acted_on_in_time_before_the_late_read(acted_then_late):
    outcome = acted_then_late()
    actions = [(row.kind, row.action) for row in outcome.reconciliations]
    assert actions[:2] == [("more", "noted"), ("more", "locked")]


def test_the_close_alerts_the_difference_its_late_read_saw_again(acted_then_late, caplog):
    acted_then_late()
    late = [text for text in alerts(caplog, "error") if "at a late read" in text]
    assert len(late) == 1 and "showed 10 units against the engine's 0" in late[0]


def test_the_summary_names_the_leg_and_counts_its_mismatch_once(acted_then_late):
    summary = acted_then_late().summary
    assert (summary.reconcile_mismatches, summary.differing) == (1, (LATE_SUMMARY,))


def test_the_result_is_not_flat_and_one_warning_names_the_leg(acted_then_late):
    result = session_result(acted_then_late())
    assert result.flat is False
    assert result.unconfirmed_legs == (LATE_SUMMARY,)
    assert [w for w in result.warnings if LATE_SUMMARY in w] != []
    assert len([w for w in result.warnings if "Watches SBIN NSE" in w]) == 1
