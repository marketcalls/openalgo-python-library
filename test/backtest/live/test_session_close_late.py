"""A leg compared late at the close, on the fake OpenAlgo (`_live/_session_close.py`,
`_live/_reconcile_findings.py`, `_live/_reconcile_close.py`; 16-session-design 5.4, the
unit 12 follow-up, unit 13).

`EntersThenStops` buys 10 SBIN at its first decision (14:42:01 IST). The placement's
answer is a gateway's page although the order reached the broker, and another client of
the account places the same order at once: the order book cannot tell the two apart, so
the entry stays unbound, in flight, for the rest of the session. Both fill at the broker
(20 units), the engine holds none. The strategy presses Ctrl+C at its fourth decision,
180 s after the entry, with on_stop="keep": the leg is in flight past the in-flight limit
when the close reads the books, so the close's read compares it late, and is the first to
see the difference. The close alerts it once, names it on the DAILY_SUMMARY's Reconcile
and End state lines as a mismatch, and reports no open position at OpenAlgo's units: a
late read's figures are neither carried nor confirmed. The entry it cannot tell apart is
an open position of its own 10 units with no stop, so the day does not end flat
(16-session-design 7.1 lists F60's leg among those the close names unprotected; unit 14).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, rig
from test_session_threads import run_threaded

import openalgo.backtest as bt
from openalgo.backtest._live._alerts_summary import summary_text
from openalgo.backtest._live._session_figures import SESSION_KEY
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import Journal

#: The units the broker holds at the close (the session's entry and the other client's),
#: and the engine's.
BROKER_UNITS = 20
ENGINE_UNITS = 0
#: The words of the close's alert and of the DAILY_SUMMARY for the leg.
LATE_ALERT = (
    "EntersThenStops SBIN NSE: OpenAlgo's books showed 20 units against the engine's 0 at a "
    "late read, while an order of the leg was in flight past the limit, neither carried by "
    "the engine's fills since nor confirmed by a later read; nothing was sent for the "
    "difference: check the position and the order book in OpenAlgo"
)
LATE_SUMMARY = (
    "EntersThenStops SBIN NSE 20 units in OpenAlgo's books against the engine's 0 at a late "
    "read, while an order of the leg was in flight past the limit, neither carried by the "
    "engine's fills since nor confirmed by a later read"
)


class EntersThenStops(bt.Strategy):
    """Buys 10 SBIN at its first decision; presses the session's stop flag `flag` at its
    fourth."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    flag: Any = None

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 4:
            type(self).flag.request()


class TwinOrder:
    """The session's orders client; its first placement is followed at once by the same
    order from another client of the account (`other`)."""

    def __init__(self, client: Any, other: Any) -> None:
        self._client, self._other = client, other
        self.freed = threading.Event()
        self.lockstep: Any = None
        self._twinned = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def placeorder(self, **order: Any) -> Any:
        answer = self._client.placeorder(**order)
        if not self._twinned:
            self._twinned = True
            self._other.placeorder(**order)
        return answer


@pytest.fixture(params=["inline", "threads"])
def late_at_the_close(request, make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """The scenario of the module docstring on the seam the parameter names, ready to run:
    calling it runs the session and returns its outcome."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    EntersThenStops.flag = world.stop
    world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)
    twin = TwinOrder(world.fake.client(), world.fake.client())

    def run() -> Any:
        if request.param == "threads":
            outcome = run_threaded(world, twin, "orders", (EntersThenStops,), on_stop="keep")
        else:
            made = world.clients()
            world.clients = lambda: type(made)(**{**made.__dict__, "orders": twin})
            outcome = world.run(EntersThenStops, on_stop="keep")
        assert world.fake.position("SBIN", "NSE") == BROKER_UNITS
        return outcome

    return run


def test_the_close_alerts_a_difference_its_late_read_saw_with_both_figures(
    late_at_the_close, caplog
):
    late_at_the_close()
    late = [text for text in alerts(caplog, "error") if "at a late read" in text]
    assert [text.split(": ", 1)[1] for text in late] == [LATE_ALERT]


def test_the_daily_summary_names_the_leg_as_a_mismatch(late_at_the_close):
    summary = late_at_the_close().summary
    assert (summary.reconcile_mismatches, summary.differing) == (1, (LATE_SUMMARY,))
    lines = summary_text(summary).splitlines()
    reconcile = next(line for line in lines if line.startswith("Reconcile"))
    end = next(line for line in lines if line.startswith("End state"))
    assert reconcile.endswith(f": 1 mismatch (at the close: {LATE_SUMMARY})")
    # The session's own entry, one of the two orders, is at the broker (unit 14, F60).
    assert end.endswith(f": not flat: a position is still open (at the close: {LATE_SUMMARY})")


def test_no_open_position_carries_the_late_reads_figure(late_at_the_close):
    units = [p.units for p in late_at_the_close().open_positions]
    assert BROKER_UNITS not in units


def test_the_entry_the_book_cannot_tell_apart_is_an_open_position_with_no_stop(late_at_the_close):
    (left,) = late_at_the_close().open_positions
    assert (left.strategy, left.units, left.stop_confirmed) == ("EntersThenStops", 10, False)
    assert "cannot be told apart in the order book from another order" in left.note


def test_the_close_saves_the_mismatch_it_names_for_a_restart_of_the_day(late_at_the_close):
    outcome = late_at_the_close()
    journal = Journal.open(
        outcome.paths["journal"],
        account=outcome.account,
        broker=outcome.broker,
        mode=Mode(outcome.mode),
        now=lambda: outcome.ended_ns,
    )
    with journal:
        counts = journal.rms_state(SESSION_KEY, day=outcome.day)
    assert counts["reconcile_mismatches"] == outcome.summary.reconcile_mismatches == 1
