"""The close sends nothing its last lines did not name, on the fake OpenAlgo
(`_live/_session_close.py`, `_live/_stops_after.py`, `_live/_reconcile_guard.py`;
16-session-design 3.3 step 5 "As built", unit 12): once the session has printed its last
lines, its final book read takes in the books and the order news only, no price reaches
the engine, and a word that ends a leg's stop is not acted on but named in the result and
in one ERROR alert.

The market is `stops_samples`': SBIN on NSE from 14:40:05 IST, trading 10 units every
10 s at 500.00. `TargetHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00 and a
target at 502.00; `StopHold` the same with the stop only. Each test stops the session
with Ctrl+C and `on_stop="keep"` at 14:46, so the leg's last line says its confirmed stop
protects it; what the test plays happens right after that line is printed, before the
close's final read. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import StopHold, TargetHold, events, stop_world

from openalgo.backtest._live import session as live_session
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.inbox import Quote

#: The end of a note whose confirmed stop covers the whole position.
PROTECTS = "it protects the position, so cancel the stop before any sale by hand"


class LinesThen(list):
    """The console's lines; `action` runs once, right after the first line that starts
    with `prefix` is printed."""

    def __init__(self, prefix: str, action: Callable[[], None]) -> None:
        super().__init__()
        self._prefix = prefix
        self._action: Callable[[], None] | None = action

    def append(self, line: str) -> None:
        super().append(line)
        if self._action is not None and line.startswith(self._prefix):
            action, self._action = self._action, None
            action()


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    made.stop_at(ist(14, 46))
    return made


def after_the_line(world: Any, strategy: str, action: Callable[[], None]) -> None:
    """Run `action` right after the leg's last line names `strategy`'s SBIN leg."""
    world.lines = LinesThen(f"{strategy} SBIN NSE", action)


def handed_after_the_final_read(outcome: Any) -> list[tuple[str, str | None]]:
    """What the session handed to the order worker after the close asked for its final
    book read (the event stream's requests after its last `ReadBooks`): (type, purpose)."""
    requests = events(outcome, "request")
    last = max(i for i, event in enumerate(requests) if event["type"] == "ReadBooks")
    return [(event["type"], event.get("purpose")) for event in requests[last + 1 :]]


def stops(world: Any) -> list[tuple[int, str]]:
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


# ----- no price reaches the engine after the last lines ---------------------------------


def test_a_tick_at_the_target_after_the_last_lines_hands_nothing_over(world):
    after_the_line(world, "TargetHold", lambda: world.fake.trade("SBIN", "NSE", 503.0, volume=10))
    outcome = world.run(TargetHold, on_stop="keep")
    assert handed_after_the_final_read(outcome) == []
    (left,) = outcome.open_positions
    assert left.note.endswith(PROTECTS)


@pytest.fixture
def quote_as_the_close_begins(world, monkeypatch):
    """The feed is dead from 14:43:30, so the session reads REST quotes of SBIN; one at
    503.00, past the target, reaches the loop as the close begins, before its final
    read."""
    world.script.at(ist(14, 43, 30), world.fake.faults.feed_dead)
    close = live_session.close_session

    def quoted(session: Any, stopped_by: str, **arguments: Any) -> Any:
        session.parts.inbox.put(Quote("SBIN", "NSE", 503.0, session.now()))
        return close(session, stopped_by, **arguments)

    monkeypatch.setattr(live_session, "close_session", quoted)
    return world


def test_a_quote_at_the_target_during_the_final_read_reaches_no_engine(quote_as_the_close_begins):
    world = quote_as_the_close_begins
    outcome = world.run(TargetHold, on_stop="keep")
    assert handed_after_the_final_read(outcome) == []
    trades = outcome.outputs["TargetHold"]["trades"]
    assert list(trades["exit_reason"]) == []
    (left,) = outcome.open_positions
    assert left.note.endswith(PROTECTS)


# ----- a stop that ends after the last lines ----------------------------------------------


@pytest.fixture
def stop_cancelled_after_the_line(world):
    """Another client of the account cancels the leg's stop right after its last line."""
    other = world.fake.client()

    def cancel_the_stop() -> None:
        stop = next(o for o in world.orders() if o.price_type == "SL-M")
        world.stop_id = stop.order_id
        other.cancelorder(order_id=stop.order_id, strategy="StopHold")

    after_the_line(world, "StopHold", cancel_the_stop)
    return world


def test_a_stop_cancelled_after_the_last_lines_is_not_placed_again(stop_cancelled_after_the_line):
    world = stop_cancelled_after_the_line
    outcome = world.run(StopHold, on_stop="keep")
    assert handed_after_the_final_read(outcome) == []
    assert stops(world) == [(10, "cancelled")]


def test_the_result_names_a_stop_cancelled_after_the_last_lines(stop_cancelled_after_the_line):
    world = stop_cancelled_after_the_line
    outcome = world.run(StopHold, on_stop="keep")
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (10, "", False)
    assert left.note == (
        f"its stop {world.stop_id} cancelled at the broker after the session's last lines; "
        "order o3 not sent; no stop at the broker, close it by hand"
    )


def test_a_stop_cancelled_after_the_last_lines_is_one_error_alert(
    stop_cancelled_after_the_line, caplog
):
    world = stop_cancelled_after_the_line
    outcome = world.run(StopHold, on_stop="keep")
    (left,) = outcome.open_positions
    # The alerts begin with the mode and the strategy: "LIVE (StopHold): ".
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        f"StopHold SBIN NSE 10 units: {left.note}"
    ]


def test_a_stop_cancelled_after_the_last_lines_is_not_said_to_be_placed_again(
    stop_cancelled_after_the_line, caplog
):
    world = stop_cancelled_after_the_line
    world.run(StopHold, on_stop="keep")
    assert not any("places it again" in text for text in alerts(caplog, "reject"))


def test_a_stop_filled_after_the_last_lines_is_one_error_alert_on_the_flat_leg(world, caplog):
    # SBIN trades at 496.00, under the 497.00 trigger, right after the leg's last line:
    # the stop fills at the broker and the engine takes the fill in the final read.
    after_the_line(world, "StopHold", lambda: world.fake.trade("SBIN", "NSE", 496.0, volume=10))
    outcome = world.run(StopHold, on_stop="keep")
    assert outcome.open_positions == ()
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        f"StopHold SBIN NSE: its stop {stop.order_id} filled at the broker after the session's "
        "last lines; the engine holds no position now: check the order book in OpenAlgo, "
        "cancelling what still works"
    ]
