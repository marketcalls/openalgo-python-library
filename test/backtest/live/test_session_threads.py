"""A live session on its real threads while the order worker is blocked in a call, on the
fake OpenAlgo (`_live/session.py`; 16-session-design 2.1, 2.4, 2.9 and 7.2 rows F63 and
F64, unit 8, run without leg rules).

These tests run the session's own threads (`threads=True`): the loop on the test's
thread, the order worker, the data worker and the alert sender on theirs. The fake's
clock runs `threads_seam.SPEED` times faster than the wall clock, moved on by a driver
thread that also plays the script (the trades, and the release of a blocked call), and
only while every thread of the session waits (`threads_seam.lockstep`), so a fake minute
lasts a couple of real seconds and the session sees the same fake time however loaded
the machine is. A client double holds one call of the order worker for a span of fake
seconds, waiting on the seam's clock.

Two strategies trade: `Early` buys 10 SBIN at its first decision and sells them at its
second; `Late` buys 10 INFY at its second decision. The session starts at 14:40:05 IST
and is stopped with Ctrl+C at 14:47. The tests read when the first decision came from
what happened, and time the held call from it.

The safe stop's tests (16-session-design 3.3 step 4) run `PressesStop` alone: it buys 10
SBIN and, once it holds them, presses Ctrl+C from `on_bar`, on the loop's thread, where
the signal handler runs too, so presses made together are seen together by the loop.
`PressesStopWithTarget` does the same with a broker-held stop and a target: once the
leg's last line says the stop protects it, the close sends nothing that line did not name
(3.3 step 5 "As built", unit 12). What only takes exposure away still goes after the last
lines and is answered before the order worker closes: the reconciler's safe corrections
(`StopHold`, from `stops_samples`, with a sale by hand first seen as Ctrl+C comes) and
the safe stop's cancel of an entry bound during the close's final read (`RestingEntry`).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist, rig
from session_trace import NO_ORPHAN
from stops_samples import (
    StopHold,
    after_the_final_read,
    ctrl_c_at_the_first_sighting,
    handed_after_the_final_read,
    held_after_the_final_read,
    sold_by_hand,
)
from threads_seam import Lockstep, lockstep, wait_for

import openalgo.backtest as bt
from openalgo.backtest._live._series_store import LiveStore
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.session import run_session

#: The loop's longest wait, real seconds: a fraction of a fake second at the seam's speed.
WAIT_MAX_S = 0.01
#: Real seconds a blocked call waits at most, so a broken test cannot hang.
BLOCK_LIMIT_S = 20.0


class Early(bt.Strategy):
    """Buys SBIN at its first decision, sells at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2 and self.position.qty:
            self.exit()


class Late(bt.Strategy):
    """Buys INFY at its second decision; keeps (bar close, the fake's time) of each."""

    symbol, exchange, interval = "INFY", "NSE", "1m"
    decided: list[tuple[int, int]] = []
    clock: Any = None

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        type(self).decided.append((bar.ts_close, type(self).clock()))
        if self.calls == 2:
            self.buy(10)


class PressesStop(bt.Strategy):
    """Buys 10 SBIN at its first decision, with the leg `rules` (none here); at its
    second, holding them, asks the session's stop flag `flag` to stop `presses` times."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    presses = 1
    flag: Any = None
    rules: dict[str, Any] = {}

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, **type(self).rules)
        elif self.calls == 2 and self.position.qty:
            for _ in range(type(self).presses):
                type(self).flag.request()


class PressesStopWithTarget(PressesStop):
    """`PressesStop` whose entry has a broker-held stop 3 points below (an SL-M at 497.00)
    and a target 2 points above (502.00), which the engine holds on ticks."""

    rules = {"stop": bt.points(3), "target": bt.points(2)}


class RestingEntry(bt.Strategy):
    """Buys 10 SBIN at a limit of 480.00 at its first decision: SBIN trades at 500.00, so
    the order rests at the broker."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, price=480.0)


class Blocking:
    """A client that holds its first call of `method` made once `when()` is true for
    `hold_s` fake seconds (by `now`), then makes it; `on_hold` runs as the hold begins.
    The hold waits on the seam's clock (`threads_seam.wait_for`), marked as a wait, so
    the clock moves on meanwhile; `run_threaded` gives it the seam (`lockstep`)."""

    def __init__(
        self,
        client: Any,
        method: str,
        *,
        when: Any,
        hold_s: float,
        now: Any,
        on_hold: Any = None,
    ) -> None:
        self._client, self._method, self._when = client, method, when
        self._hold_ns, self._now = round(hold_s * NS), now
        self._on_hold = on_hold
        self.held_at: int | None = None
        self.freed = threading.Event()
        self.lockstep: Lockstep | None = None

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if name != self._method:
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            if self.held_at is None and self._when():
                self.held_at = self._now()
                if self._on_hold is not None:
                    self._on_hold()
                assert self.lockstep is not None, "run the session with run_threaded"
                until_ns = self.held_at + self._hold_ns
                wait_for(
                    self.lockstep,
                    lambda: self._now() >= until_ns or self.freed.is_set(),
                    limit_s=BLOCK_LIMIT_S,
                )
            return attribute(*args, **kwargs)

        return call


def run_threaded(
    world: Any,
    blocking: Any,
    role: str,
    strategies: tuple[Any, ...] = (Early, Late),
    **arguments: Any,
) -> Any:
    """Run `strategies` with the session's threads, `blocking` (a client double with
    `freed` and `lockstep`) standing in for the session's `role` client, and `bt.live`'s
    other `arguments` (`on_stop`), on the seam's clock (`threads_seam.lockstep`)."""
    clients = world.clients()
    clients = type(clients)(**{**clients.__dict__, role: blocking})
    clock = world.fake.clock
    with lockstep(world.script) as step:
        blocking.lockstep = step

        def wait(seconds: float) -> None:
            until_ns = clock.now() + round(seconds * NS)
            wait_for(step, lambda: clock.now() >= until_ns, limit_s=BLOCK_LIMIT_S)

        try:
            return run_session(
                list(strategies),
                world.fake.client(),
                clients=clients,
                feed=world.fake.direct_feed(),
                now=clock.now,
                sleep=wait,
                threads=True,
                signals=False,
                stop=world.stop,
                emit=world.lines.append,
                capital=100_000,
                max_daily_loss=5_000,
                alerts_to="trader1",
                live_dir=world.folder / "live",
                wait_max_s=WAIT_MAX_S,
                **arguments,
            )
        finally:
            blocking.freed.set()


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(
        make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 5), symbols=("SBIN", "INFY")
    )
    made.stop_at(ist(14, 47))
    Late.decided, Late.clock = [], made.fake.clock.now
    return made


def first_decision(world) -> int:
    """The close of the strategies' first decision."""
    return min(close for close, _ in Late.decided)


def kernel_volumes(monkeypatch) -> dict[tuple[str, int], float]:
    """The volume of each strategy bar the kernel closes, by (symbol, its open), as its
    rows hand them to the series: (instrument, ts_open, ts_close, open, high, low, close,
    volume)."""
    volumes: dict[tuple[str, int], float] = {}
    reveal = LiveStore.reveal

    def keep(self, row):
        volumes[(self.symbol, row[1])] = row[7]
        return reveal(self, row)

    monkeypatch.setattr(LiveStore, "reveal", keep)
    return volumes


def test_F63_while_an_order_call_hangs_slices_close_and_decisions_are_made_on_time(
    world, monkeypatch
):
    volumes = kernel_volumes(monkeypatch)
    blocking = Blocking(
        world.fake.client(), "placeorder", when=lambda: True, hold_s=150, now=world.fake.clock.now
    )
    outcome = run_threaded(world, blocking, "orders")
    first = first_decision(world)
    assert blocking.held_at is not None and blocking.held_at - first < 5 * NS
    late = dict(Late.decided)
    # Decided within a few fake seconds of each close while the call was held, on bars
    # built from the minute's ticks (INFY trades 10 units every 10 s).
    for close in (first + 60 * NS, first + 120 * NS):
        assert late[close] - close < 5 * NS
        assert volumes[("INFY", close - 60 * NS)] == 60
    buys = [order for order in world.orders() if order.action == "BUY"]
    assert buys[0].symbol == "SBIN"
    assert buys[0].placed_ns >= blocking.held_at + 150 * NS
    # Once freed, the held call goes first; Late's entry, decided while the call was
    # held, waited too long and was dropped unsent; the sale is the Ctrl+C's at 14:47.
    calls = world.fake.log.calls("placeorder")
    sent = [(call.detail["request"]["symbol"], call.detail["request"]["action"]) for call in calls]
    assert sent == [("SBIN", "BUY"), ("SBIN", "SELL")]
    assert list(outcome.outputs["Late"]["orders"]["status"]) == ["Rejected"]


def test_F64_an_entry_queued_behind_a_blocked_worker_expires_and_the_exit_goes(world, caplog):
    fake = world.fake

    def a_while_after_the_first_fill() -> bool:
        fills = fake.fills()
        return bool(fills) and fake.clock.now() >= fills[0].ts_ns + 50 * NS

    blocking = Blocking(
        fake.client(),
        "analyzerstatus",
        when=a_while_after_the_first_fill,
        hold_s=30,
        now=fake.clock.now,
    )
    outcome = run_threaded(world, blocking, "guard")
    sells = [order for order in world.orders() if order.action == "SELL"]
    assert sells[0].symbol == "SBIN"
    assert sells[0].placed_ns >= blocking.held_at + 30 * NS
    assert [order.symbol for order in world.orders() if order.action == "BUY"] == ["SBIN"]
    assert list(outcome.outputs["Late"]["orders"]["status"]) == ["Rejected"]
    assert any("stale" in text for text in alerts(caplog, "reject"))


@pytest.fixture
def alone(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """A rig for `PressesStop` alone, which stops the session itself."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(14, 40, 5))
    PressesStop.flag = made.stop
    return made


def test_two_ctrl_c_at_once_send_the_flatten_exit_and_stop_flat(alone):
    # Decided for unit 10: two presses seen together act as the first followed by the
    # second, as presses a moment apart do.
    PressesStop.presses = 2
    passing = Blocking(
        alone.fake.client(), "placeorder", when=lambda: False, hold_s=0, now=alone.fake.clock.now
    )
    outcome = run_threaded(alone, passing, "orders", (PressesStop,))
    assert outcome.stopped_by == "ctrl_c"
    assert [order.action for order in alone.orders()] == ["BUY", "SELL"]
    assert alone.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    assert not any("close it by hand" in line for line in alone.lines)


def test_an_exit_waiting_for_a_slow_read_is_not_sent_after_the_last_lines(alone):
    # The flatten's exit waits for a read of the books (unit 12), whose position book call
    # is held 7.5 fake seconds: past the second Ctrl+C's 5 s wait, within the close's 5 s
    # final read. The last lines say the exit was not sent; the close's read then agrees,
    # and the exit must still not be sent.
    PressesStop.presses = 2
    slow = Blocking(
        alone.fake.client(),
        "positionbook",
        when=lambda: alone.stop.count >= 1,
        hold_s=7.5,
        now=alone.fake.clock.now,
    )
    outcome = run_threaded(alone, slow, "orders", (PressesStop,))
    assert slow.held_at is not None
    assert [order.action for order in alone.orders()] == ["BUY"]
    assert alone.fake.position("SBIN", "NSE") == 10
    (left,) = outcome.open_positions
    assert left.note == "order o2 not sent; no stop at the broker, close it by hand"


def test_a_second_ctrl_c_while_the_exit_is_sent_waits_for_its_fill(alone):
    # The safe stop's SELL is held in its call for 1 fake second, well inside the second
    # Ctrl+C's bounded wait at any real-time pace of the run; the second Ctrl+C comes as
    # the hold begins.
    PressesStop.presses = 1
    clock = alone.fake.clock
    held = Blocking(
        alone.fake.client(),
        "placeorder",
        when=lambda: alone.stop.count >= 1,
        hold_s=1,
        now=clock.now,
        on_hold=alone.stop.request,
    )
    outcome = run_threaded(alone, held, "orders", (PressesStop,))
    assert held.held_at is not None
    assert outcome.stopped_by == "ctrl_c"
    assert [order.action for order in alone.orders()] == ["BUY", "SELL"]
    assert alone.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    assert not any("close it by hand" in line for line in alone.lines)


# ----- the close sends nothing its last lines did not name ------------------------------


@pytest.mark.parametrize(
    ("on_stop", "presses"),
    [("keep", 1), ("flatten", 2)],
    ids=["on_stop keep", "two Ctrl+C at once"],
)
def test_a_tick_at_the_target_during_the_closes_final_read_sends_nothing(alone, on_stop, presses):
    # The leg's last line says its stop protects it. The close's final read then holds
    # its position book call 3 fake seconds, and SBIN trades at 503.00, past the 502.00
    # target, from the moment the hold begins: the stop must stay at the broker, and
    # nothing may be handed to the order worker after the read was asked for.
    PressesStop.presses = presses
    fake = alone.fake

    def at_the_target() -> None:
        alone.script.price = lambda now_ns: 503.0
        for _ in range(3):
            fake.trade("SBIN", "NSE", 503.0, volume=10.0)

    slow = Blocking(
        fake.client(),
        "positionbook",
        when=lambda: any(line.startswith("PressesStopWithTarget SBIN") for line in alone.lines),
        hold_s=3.0,
        now=fake.clock.now,
        on_hold=at_the_target,
    )
    outcome = run_threaded(alone, slow, "orders", (PressesStopWithTarget,), on_stop=on_stop)
    assert slow.held_at is not None
    assert handed_after_the_final_read(outcome) == []
    assert held_after_the_final_read(outcome) == []
    # The engine values the position at the last price it took before the lines, 500.00
    # (the fill's): a price of 503.00 reaching it would show 30.00 of gross profit.
    assert outcome.summary.gross_pnl == 0.0
    sells = [(o.price_type, o.status) for o in alone.orders() if o.action == "SELL"]
    assert sells == [("SL-M", "trigger pending")]
    (left,) = outcome.open_positions
    assert left.note.endswith(
        "it protects the position, so cancel the stop before any sale by hand"
    )


# ----- what still goes after the last lines, answered before the worker closes ----------


def passing(world: Any) -> Blocking:
    """An orders client that holds no call."""
    return Blocking(
        world.fake.client(), "placeorder", when=lambda: False, hold_s=0, now=world.fake.clock.now
    )


def stops_at_the_broker(world: Any) -> list[tuple[int, str]]:
    return [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]


@pytest.mark.parametrize(
    ("sold", "stop_at_the_broker"),
    [(10, (10, "cancelled")), (4, (6, "trigger pending"))],
    ids=["all of it: the stop is cancelled", "4 of 10: the stop is cut to 6"],
)
def test_a_sale_by_hand_adopted_in_the_final_read_fits_the_stop_at_the_broker(
    alone, monkeypatch, sold, stop_at_the_broker
):
    # Another client sells at 14:43; Ctrl+C with on_stop="keep" comes as the first
    # comparison that shows it, so the close's final read is the one that adopts it.
    sold_by_hand(alone, sold, ist(14, 43))
    ctrl_c_at_the_first_sighting(alone, monkeypatch, broker_below=StopHold.qty)
    outcome = run_threaded(alone, passing(alone), "orders", (StopHold,), on_stop="keep")
    assert "adopted" in [event["kind"] for event in after_the_final_read(outcome)]
    assert stops_at_the_broker(alone) == [stop_at_the_broker]
    assert alone.fake.position("SBIN", "NSE") == StopHold.qty - sold


def test_the_stop_on_the_other_side_is_cancelled_at_the_broker_before_the_close(alone, monkeypatch):
    # Another client sells 20: the broker holds 10 short, and a trigger of the stop would
    # take it to 20 short. The reconciler cancels the stop in the close's final read.
    sold_by_hand(alone, 20, ist(14, 43))
    ctrl_c_at_the_first_sighting(alone, monkeypatch, broker_below=StopHold.qty)
    outcome = run_threaded(alone, passing(alone), "orders", (StopHold,), on_stop="keep")
    assert ("CancelRequest", "stop") in handed_after_the_final_read(outcome)
    assert stops_at_the_broker(alone) == [(10, "cancelled")]


@pytest.mark.parametrize(
    ("on_stop", "presses"),
    [("keep", 1), ("flatten", 2)],
    ids=["on_stop keep", "two Ctrl+C at once"],
)
def test_an_entry_bound_during_the_final_read_is_cancelled_at_the_broker(alone, on_stop, presses):
    # The entry's placement call is held 8 fake seconds and Ctrl+C comes as the hold
    # begins: the safe stop's cancel of the entry waits for the bind, which comes during
    # the close's final read. The cancel never adds exposure, so it goes then.
    fake = alone.fake

    def press() -> None:
        for _ in range(presses):
            alone.stop.request()

    slow = Blocking(
        fake.client(),
        "placeorder",
        when=lambda: True,
        hold_s=8.0,
        now=fake.clock.now,
        on_hold=press,
    )
    outcome = run_threaded(alone, slow, "orders", (RestingEntry,), on_stop=on_stop)
    assert ("CancelRequest", "entry") in handed_after_the_final_read(outcome)
    assert [(o.action, o.price_type, o.status) for o in alone.orders()] == [
        ("BUY", "LIMIT", "cancelled")
    ]
    assert outcome.open_positions == ()


def test_an_entry_filled_during_the_final_read_is_one_error_alert(alone, caplog):
    # As above with a MARKET entry and a stop rule: it fills as its answer comes, during
    # the close's final read, and the stop placed with the fill is not sent (3.3 step 5
    # "As built"): the leg ends with no stop, named by the ERROR alert, which the session
    # invariants take instead of a stop.
    alone.trace.left_unprotected_by_decision(
        "StopHold", "3.3 step 5 As built: an entry filled after the last lines"
    )
    fake = alone.fake
    slow = Blocking(
        fake.client(),
        "placeorder",
        when=lambda: True,
        hold_s=8.0,
        now=fake.clock.now,
        on_hold=alone.stop.request,
    )
    outcome = run_threaded(alone, slow, "orders", (StopHold,), on_stop="keep")
    assert fake.position("SBIN", "NSE") == 10
    (left,) = outcome.open_positions
    assert left.note == (
        "its position went from 0 to 10 units after the session's last lines; order o2 not "
        "sent; no stop at the broker, close it by hand"
    )
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        f"StopHold SBIN NSE 10 units: {left.note}"
    ]


def test_an_entry_unanswered_at_the_close_is_one_error_alert(alone, caplog):
    # The placement call is held 12 fake seconds: its answer comes after the close's final
    # read, so the entry's cancel is never sent and the order rests at the broker. The
    # session ends with that order working unknown to it, named by the ERROR alert (3.3
    # step 5 "As built"): it is no orphan the session could have cancelled.
    alone.trace.waive(NO_ORPHAN, "the entry's answer comes after the session closed")
    fake = alone.fake
    slow = Blocking(
        fake.client(),
        "placeorder",
        when=lambda: True,
        hold_s=12.0,
        now=fake.clock.now,
        on_hold=alone.stop.request,
    )
    outcome = run_threaded(alone, slow, "orders", (RestingEntry,), on_stop="keep")
    assert outcome.open_positions == ()
    named = [text for text in alerts(caplog, "error") if "RestingEntry SBIN NSE" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        "RestingEntry SBIN NSE: entry order BUY 10 (entry, intent "
        "20261116-live-1-RestingEntry-o1) in flight, not answered yet; the engine holds no "
        "position now: check the order book in OpenAlgo, cancelling what still works"
    ]


@pytest.mark.parametrize("book", ["positionbook", "orderbook"])
def test_a_stop_cancelled_during_the_final_read_is_named_in_the_result(alone, caplog, book):
    # The leg's last line says its stop protects it. The close's final read then holds
    # its call of `book` 1 fake second, and another client cancels the stop as the hold
    # begins: the stream's word of it comes while the books are read. The leg ends with no
    # stop, named by the ERROR alert, which the session invariants take instead.
    alone.trace.left_unprotected_by_decision(
        "PressesStopWithTarget", "3.3 step 5 As built: a stop cancelled in the final read"
    )
    PressesStop.presses = 1
    fake = alone.fake
    other = fake.client()
    cancelled: list[str] = []

    def cancel_the_stop() -> None:
        stop = next(o for o in alone.orders() if o.price_type == "SL-M")
        cancelled.append(stop.order_id)
        other.cancelorder(order_id=stop.order_id, strategy="PressesStopWithTarget")

    slow = Blocking(
        fake.client(),
        book,
        when=lambda: any(line.startswith("PressesStopWithTarget SBIN") for line in alone.lines),
        hold_s=1.0,
        now=fake.clock.now,
        on_hold=cancel_the_stop,
    )
    outcome = run_threaded(alone, slow, "orders", (PressesStopWithTarget,), on_stop="keep")
    assert stops_at_the_broker(alone) == [(10, "cancelled")]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == ("", False)
    assert left.note.startswith(
        f"its stop {cancelled[0]} cancelled at the broker after the session's last lines; "
    )
    named = [text for text in alerts(caplog, "error") if "after the session's last lines" in text]
    assert [text.split(": ", 1)[1] for text in named] == [
        f"PressesStopWithTarget SBIN NSE 10 units: {left.note}"
    ]
