"""The close sends nothing its last lines did not name, on the fake OpenAlgo
(`_live/_session_close.py`, `_live/_stops_after.py`, `_live/_reconcile_guard.py`;
16-session-design 3.3 step 5 "As built", unit 12): once the session has printed its last
lines, its final book read takes in the books and the order news only, no price reaches
the engine, and a word that ends a leg's stop is not acted on but named in the result and
in one ERROR alert. What only takes exposure away still goes, and is answered before the
order worker closes: the reconciler's own safe corrections, the stop fitted to a sale by
hand it adopts among them.

The market is `stops_samples`': SBIN on NSE from 14:40:05 IST, trading 10 units every
10 s at 500.00. `TargetHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00 and a
target at 502.00; `StopHold` the same with the stop only. Each test stops the session
with Ctrl+C and `on_stop="keep"` at 14:46, so the leg's last line says its confirmed stop
protects it; what the test plays happens right after that line is printed, before the
close's final read. The tests of a sale by hand press Ctrl+C earlier, as the first
comparison that shows the sale guards the leg, so the close's final read is the second
comparison to show it, the one that acts. Expected values come from the scenario and the
fake's books.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import (
    StopHold,
    TargetHold,
    after_the_final_read,
    ctrl_c_at_the_first_sighting,
    events,
    handed_after_the_final_read,
    held_after_the_final_read,
    sold_by_hand,
    stop_world,
)
from test_stops_silent import (
    SILENCE_ARMED,
    RestsAnEntry,
    probes_fail,
    stop_placements,
    the_entry_fills,
)

from openalgo.backtest._live import session as live_session
from openalgo.backtest._live._session_close import FINAL_READ_WAIT_S
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._worker_queue import Probe
from openalgo.backtest._live.inbox import Quote
from openalgo.backtest._live.worker import OrderWorker

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


def stops(world: Any) -> list[tuple[int, str]]:
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


# ----- no price reaches the engine after the last lines ---------------------------------


def test_a_tick_at_the_target_after_the_last_lines_hands_nothing_over(world):
    after_the_line(world, "TargetHold", lambda: world.fake.trade("SBIN", "NSE", 503.0, volume=10))
    outcome = world.run(TargetHold, on_stop="keep")
    assert handed_after_the_final_read(outcome) == []
    (left,) = outcome.open_positions
    assert left.note.endswith(PROTECTS)


def test_a_tick_at_the_target_after_the_last_lines_makes_the_engine_decide_nothing(world):
    # Had the price reached the engine, its target's exit would cancel the stop first:
    # a request the exit guard holds back, which hands nothing over either.
    after_the_line(world, "TargetHold", lambda: world.fake.trade("SBIN", "NSE", 503.0, volume=10))
    outcome = world.run(TargetHold, on_stop="keep")
    assert held_after_the_final_read(outcome) == []


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
    assert held_after_the_final_read(outcome) == []
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
    # After the last lines the session sends nothing they did not name (3.3 step 5 "As
    # built"): the leg ends with no stop, named by the ERROR alert, which the session
    # invariants take instead of a stop (16-session-design 7.1).
    world.trace.left_unprotected_by_decision(
        "StopHold", "3.3 step 5 As built: a stop cancelled after the last lines"
    )
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


# ----- the reconciler's safe corrections in the final read ------------------------------


@pytest.fixture
def first_seen_before_ctrl_c(world, monkeypatch):
    """Ctrl+C comes as the first comparison that shows a sale by hand of SBIN (made at
    14:43:00 by the test), so the close's final read is the one that acts on it."""
    ctrl_c_at_the_first_sighting(world, monkeypatch, broker_below=StopHold.qty)
    return world


def acted_in_the_final_read(outcome: Any) -> list[str]:
    """The reconciler's actions after the close asked for its final read: "adopted"."""
    return [event["kind"] for event in after_the_final_read(outcome) if event["kind"] == "adopted"]


@pytest.mark.parametrize(
    ("sold", "stop_at_the_broker"),
    [(10, (10, "cancelled")), (4, (6, "trigger pending"))],
    ids=["all of it: the stop is cancelled", "4 of 10: the stop is cut to 6"],
)
def test_a_sale_by_hand_adopted_in_the_final_read_fits_the_stop_at_the_broker(
    first_seen_before_ctrl_c, sold, stop_at_the_broker
):
    world = first_seen_before_ctrl_c
    sold_by_hand(world, sold, ist(14, 43))
    outcome = world.run(StopHold, on_stop="keep")
    assert acted_in_the_final_read(outcome) == ["adopted"]
    assert stops(world) == [stop_at_the_broker]
    assert world.fake.position("SBIN", "NSE") == StopHold.qty - sold


def test_a_part_sold_by_hand_in_the_final_read_leaves_the_rest_with_its_cut_stop(
    first_seen_before_ctrl_c,
):
    world = first_seen_before_ctrl_c
    sold_by_hand(world, 4, ist(14, 43))
    outcome = world.run(StopHold, on_stop="keep")
    (left,) = outcome.open_positions
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    assert stops(world) == [(6, "trigger pending")]
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (6, stop.order_id, True)
    assert left.note == (
        "its position went from 10 to 6 units after the session's last lines; "
        f"stop {stop.order_id} confirmed at the broker: {PROTECTS}"
    )


def test_a_sale_by_hand_of_all_of_it_in_the_final_read_is_one_error_alert(
    first_seen_before_ctrl_c, caplog
):
    world = first_seen_before_ctrl_c
    sold_by_hand(world, 10, ist(14, 43))
    outcome = world.run(StopHold, on_stop="keep")
    assert outcome.open_positions == ()
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        "StopHold SBIN NSE: its position went from 10 to 0 units after the session's last "
        f"lines, and its stop {stop.order_id} cancelled at the broker; the engine holds no "
        "position now: check the order book in OpenAlgo, cancelling what still works"
    ]


def test_the_stop_on_the_other_side_is_cancelled_at_the_broker_before_the_close(
    first_seen_before_ctrl_c,
):
    # Another client sells 20: the broker holds 10 short, and a trigger of the stop would
    # take it to 20 short. The reconciler cancels the stop in the final read.
    world = first_seen_before_ctrl_c
    sold_by_hand(world, 20, ist(14, 43))
    world.run(StopHold, on_stop="keep")
    assert stops(world) == [(10, "cancelled")]


# ----- what the order worker put back before the last lines -------------------------------


class TimedLines(list):
    """The console's lines, with the fake's time at which each was printed (`at`)."""

    def __init__(self, clock: Any) -> None:
        super().__init__()
        self._clock = clock
        self.at: list[tuple[str, int]] = []

    def append(self, line: str) -> None:
        super().append(line)
        self.at.append((line, self._clock.now()))


@pytest.fixture
def stop_put_back_at_ctrl_c(world):
    """`RestsAnEntry`'s entry fills at 14:43:10 while OpenAlgo is silent (its probes fail
    six times), so its stop waits in the order worker, put back; Ctrl+C with
    on_stop="keep" comes at 14:43:20, and the leg's last line says the stop is held back
    and not sent."""
    world.script.at(SILENCE_ARMED, lambda: world.fake.faults.drop_connection("orderstatus"))
    probes_fail(world, 6)
    the_entry_fills(world)
    world.stop_at(ist(14, 43, 20))
    world.lines = TimedLines(world.fake.clock)
    # The stop held back at the last lines is not sent after them (3.3 step 5 "As
    # built"): the leg ends with no stop, named by its line and the ERROR alert.
    world.trace.left_unprotected_by_decision(
        "RestsAnEntry", "3.3 step 5 As built: a stop held back at the last lines"
    )
    return world


def test_the_last_line_names_a_stop_put_back_as_held_back_and_not_sent(stop_put_back_at_ctrl_c):
    world = stop_put_back_at_ctrl_c
    world.run(RestsAnEntry, on_stop="keep")
    (line,) = [line for line in world.lines if line.startswith("RestsAnEntry SBIN NSE")]
    assert "held back, not sent" in line


@pytest.fixture
def workers(monkeypatch):
    """Each order worker a session builds, as built."""
    built: list[OrderWorker] = []
    build = OrderWorker.__init__

    def kept(worker: OrderWorker, *arguments: Any, **options: Any) -> None:
        build(worker, *arguments, **options)
        built.append(worker)

    monkeypatch.setattr(OrderWorker, "__init__", kept)
    return built


def test_a_stop_put_back_before_the_last_lines_is_not_placed_when_openalgo_answers_after_them(
    stop_put_back_at_ctrl_c, workers
):
    # A probe of the silent OpenAlgo asked as the leg's line is printed runs after it, and
    # OpenAlgo answers it (the order-book reads still set to fail are cleared): its hold on
    # sending ends within the close's final read, and the stop put back would go then but
    # for its withdrawal.
    world = stop_put_back_at_ctrl_c

    def probe_answered() -> None:
        while world.fake.faults.take("orderbook") is not None:
            pass
        workers[0].put(Probe())

    after_the_line(world, "RestsAnEntry", probe_answered)
    world.run(RestsAnEntry, on_stop="keep")
    assert stop_placements(world) == []


def test_the_final_read_does_not_wait_for_a_stop_put_back_before_the_last_lines(
    stop_put_back_at_ctrl_c,
):
    # The close ends once the final read's books are taken in, about a second after the
    # line, not when the read's FINAL_READ_WAIT_S run out.
    world = stop_put_back_at_ctrl_c
    outcome = world.run(RestsAnEntry, on_stop="keep")
    (printed_ns,) = [at for line, at in world.lines.at if line.startswith("RestsAnEntry SBIN")]
    (closed,) = events(outcome, "closed")
    assert closed["ts_ns"] - printed_ns < 2 * NS < FINAL_READ_WAIT_S * NS
