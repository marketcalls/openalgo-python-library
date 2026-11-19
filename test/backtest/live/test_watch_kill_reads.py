"""The kill switch's position-book reads, with in-process doubles of the session loop, one
strategy and the order worker (`_live/_kill_switch.py`; 16-session-design 5.4, unit 11):
a read counts only when it began once the state it judges had begun, reads judged in a row
are at least the read pace apart, at most one read is asked for at a time, a flat strategy
is reported with units left at the broker only when two reads in a row agree on them, and a
lasting disagreement is read less often once it is alerted.

The worker double reads the position book one request at a time, in order, each read
stamped with its start, showing the broker's units as they were then, and answered
`read_s` seconds later. The loop double steps every 0.1 s meanwhile, as the loop does with
threads while the worker's call is in flight (under the test seam it cannot). Nothing
contacts a server.
"""

from __future__ import annotations

import collections
from types import SimpleNamespace

from openalgo.backtest._live._kill_switch import (
    ACCOUNT_LOSS,
    KILL_READ_BACKOFF_S,
    KillSwitch,
)
from openalgo.backtest._live._worker_queue import ReadPositions
from openalgo.backtest._live.inbox import PositionRows

NS = 10**9
STEP_NS = NS // 10
#: When the kill switch fires: 14:44:01 IST on Monday 2026-11-16, UTC nanoseconds.
FIRED_NS = 1_794_820_441 * NS


class Strategy:
    """The session's strategy double: `units` of SBIN NSE (signed), no working order, the
    kernel's exits it was asked for in `exits`."""

    name, symbol, exchange = "HoldSbin", "SBIN", "NSE"
    finished = False
    square_off_ns = FIRED_NS + 16 * 60 * NS

    def __init__(self, units: float) -> None:
        self.units = units
        self.exits: list[str] = []
        self.engine = self

    def position(self, instrument: int, product: str) -> dict | None:
        if self.units == 0:
            return None
        return {"quantity_units": float(self.units), "tranches": [(1, 500.0, self.units, 0)]}

    def working(self) -> list[int]:
        return []

    def entry_orders(self) -> list[int]:
        return []

    def lock(self, reason: str) -> bool:
        return True

    def cancel(self, kernel_order: int) -> None:
        raise AssertionError("the strategy has no working order to cancel")

    def request_exit(self, reason: str) -> str:
        self.exits.append(reason)
        return "sent"


class Worker:
    """The order worker double: position-book reads one at a time, in order, each taking
    `read_s` seconds and showing `broker` (signed units of SBIN; None for a book OpenAlgo
    cannot read) as it was at its start."""

    def __init__(self, read_s: float, broker: int | None) -> None:
        self.read_ns = round(read_s * NS)
        self.broker = broker
        self.queue: collections.deque[ReadPositions] = collections.deque()
        self.reading: tuple[int, int] | None = None
        self.most_waiting = 0

    def put(self, request: ReadPositions) -> None:
        assert isinstance(request, ReadPositions)
        self.queue.append(request)
        self.most_waiting = max(self.most_waiting, self.waiting)

    @property
    def waiting(self) -> int:
        """The reads asked for and not answered yet."""
        return len(self.queue) + (self.reading is not None)

    def step(self, now_ns: int, kill: KillSwitch) -> None:
        """Answer the read under way once its time is up, then begin the next."""
        if self.reading is not None and now_ns >= self.reading[0] + self.read_ns:
            started, units = self.reading
            self.reading = None
            kill.positions(PositionRows(book(units), started))
        if self.reading is None and self.queue:
            self.queue.popleft()
            self.reading = (now_ns, self.broker)


def book(units: int | None) -> dict:
    if units is None:
        return {"status": "error", "message": "the broker's position book is unavailable"}
    return {
        "status": "success",
        "data": [{"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": units}],
    }


class Loop:
    """The session loop double: its clock, its strategy, the worker it hands requests to,
    and the events and alerts it was given."""

    locked = False

    def __init__(self, strategy: Strategy, worker: Worker) -> None:
        self.clock = FIRED_NS
        self.strategy, self.worker = strategy, worker
        self.asked_ns: list[int] = []
        self.events: list[tuple[str, dict]] = []
        self.alerts: list[str] = []
        arguments = SimpleNamespace(on_stop="keep")
        self.parts = SimpleNamespace(spec=SimpleNamespace(arguments=arguments))

    def now(self) -> int:
        return self.clock

    def trader_named(self, name: str) -> Strategy:
        assert name == self.strategy.name
        return self.strategy

    def hand_over(self, requests: list) -> None:
        for request in requests:
            self.asked_ns.append(self.clock)
            self.worker.put(request)

    def event(self, kind: str, **fields) -> None:
        self.events.append((kind, fields))

    def alert(self, kind, text: str) -> None:
        self.alerts.append(text)


def fired(units: float, *, read_s: float, broker: int) -> tuple[Loop, KillSwitch]:
    """A loop whose strategy holds `units`, its kill switch fired at `FIRED_NS`."""
    loop = Loop(Strategy(units), Worker(read_s, broker))
    kill = KillSwitch(loop)
    kill.fire(loop.strategy, ACCOUNT_LOSS)
    kill.duties(loop.clock)
    return loop, kill


def step_for(loop: Loop, kill: KillSwitch, seconds: float) -> None:
    """Step the loop every 0.1 s for `seconds`: the worker, then the kill switch's duty."""
    for _ in range(round(seconds * NS) // STEP_NS):
        loop.clock += STEP_NS
        loop.worker.step(loop.clock, kill)
        kill.duties(loop.clock)


def test_a_read_slower_than_the_read_pace_still_lets_the_kill_switch_exit():
    loop, kill = fired(10, read_s=3.0, broker=10)
    step_for(loop, kill, 10)
    assert loop.strategy.exits == [ACCOUNT_LOSS]


def test_no_second_read_is_asked_for_while_one_is_on_its_way():
    # The broker holds 4 of the engine's 10, so the kill switch keeps wanting reads.
    loop, kill = fired(10, read_s=3.0, broker=4)
    step_for(loop, kill, 60)
    assert loop.worker.most_waiting == 1


def flat_reports(loop: Loop) -> list[str]:
    """The alerts that report a fired strategy flat."""
    return [text for text in loop.alerts if " is flat in " in text]


#: The flat report of a broker that is flat too.
FLAT_TOO = (
    "kill switch (account_loss): HoldSbin is flat in SBIN NSE, flat at the broker too; "
    "its entries stay locked for the day"
)


def exited_then_flat(broker: int | None) -> tuple[Loop, KillSwitch]:
    """The kill switch exits 10 SBIN; 5 s after its last read the exit fills, so the engine
    is flat and the next read is asked for at once, OpenAlgo's book then showing `broker`."""
    loop, kill = fired(10, read_s=0.2, broker=10)
    step_for(loop, kill, 5)
    assert loop.strategy.exits == [ACCOUNT_LOSS]
    loop.strategy.units, loop.worker.broker = 0, broker
    return loop, kill


def test_a_read_begun_before_the_position_went_flat_is_not_taken_for_its_report():
    loop, kill = fired(10, read_s=0.5, broker=10)
    step_for(loop, kill, 1)
    assert loop.strategy.exits == [ACCOUNT_LOSS]
    # Another reader's read begins with the exit still working; the exit then fills.
    loop.worker.put(ReadPositions())
    step_for(loop, kill, 0.2)
    loop.strategy.units = loop.worker.broker = 0
    step_for(loop, kill, 3)
    assert flat_reports(loop) == [FLAT_TOO]


def test_a_position_book_that_lags_the_exits_fill_by_a_second_is_reported_flat():
    loop, kill = exited_then_flat(broker=10)
    step_for(loop, kill, 1)
    loop.worker.broker = 0
    step_for(loop, kill, 10)
    assert flat_reports(loop) == [FLAT_TOO]


def test_a_second_read_begun_within_the_read_pace_of_the_first_is_not_taken_for_the_report():
    loop, kill = exited_then_flat(broker=10)
    # The kill switch's read (begun at 5.2 s) shows 10; another reader's, begun 0.4 s
    # later, shows 10 too; the book catches up 1.5 s after the fill.
    step_for(loop, kill, 0.5)
    loop.worker.put(ReadPositions())
    step_for(loop, kill, 1)
    loop.worker.broker = 0
    step_for(loop, kill, 10)
    assert flat_reports(loop) == [FLAT_TOO]


def test_a_position_book_that_keeps_showing_the_units_is_reported_once_to_close_by_hand():
    loop, kill = exited_then_flat(broker=10)
    step_for(loop, kill, 60)
    assert flat_reports(loop) == [
        "kill switch (account_loss): HoldSbin is flat in SBIN NSE, OpenAlgo's position book "
        "still shows 10 units: close them by hand; its entries stay locked for the day"
    ]


def test_a_flat_report_waits_for_two_reads_in_a_row_with_one_figure():
    loop, kill = exited_then_flat(broker=10)
    step_for(loop, kill, 1)
    loop.worker.broker = 4
    step_for(loop, kill, 10)
    [report] = flat_reports(loop)
    assert "still shows 4 units: close them by hand" in report


def test_one_unreadable_read_after_the_exit_is_not_reported():
    loop, kill = exited_then_flat(broker=None)
    step_for(loop, kill, 1)
    loop.worker.broker = 0
    step_for(loop, kill, 10)
    assert flat_reports(loop) == [FLAT_TOO]


def test_a_position_book_that_stays_unreadable_after_the_exit_is_reported_once():
    loop, kill = exited_then_flat(broker=None)
    step_for(loop, kill, 60)
    [report] = flat_reports(loop)
    assert "its position book cannot be read: check the position in OpenAlgo" in report


def test_disagreeing_reads_about_two_different_positions_are_not_two_in_a_row():
    # One read lags the 10 units; an add whose cancel was refused then fills to 20, and
    # one read lags that fill too before the book catches up.
    loop, kill = fired(10, read_s=0.1, broker=0)
    step_for(loop, kill, 1)
    loop.strategy.units, loop.worker.broker = 20, 10
    step_for(loop, kill, 2.5)
    loop.worker.broker = 20
    step_for(loop, kill, 10)
    assert loop.alerts == []
    assert loop.strategy.exits == [ACCOUNT_LOSS]


def test_a_disagreement_is_alerted_again_once_the_engines_units_change():
    loop, kill = fired(10, read_s=0.1, broker=4)
    step_for(loop, kill, 10)
    loop.strategy.units, loop.worker.broker = 20, 14
    step_for(loop, kill, 60)
    assert len(loop.alerts) == 2
    assert "holds 20 units of SBIN NSE" in loop.alerts[1]
    assert "shows 14 units" in loop.alerts[1]
    assert "is sent for more than the broker holds" in loop.alerts[1]


def test_a_lasting_disagreement_is_read_less_often_once_alerted():
    loop, kill = fired(10, read_s=0.1, broker=4)
    step_for(loop, kill, 120)
    asked = loop.asked_ns
    # The disagreement is alerted at its second read; every read after waits the back-off.
    assert len(loop.alerts) == 1
    gaps = [later - earlier for earlier, later in zip(asked[1:], asked[2:], strict=False)]
    assert gaps and min(gaps) >= KILL_READ_BACKOFF_S * NS


def test_a_lasting_disagreement_is_written_to_the_events_once_per_broker_figure():
    loop, kill = fired(10, read_s=0.1, broker=4)
    step_for(loop, kill, 60)
    loop.worker.broker = 3
    step_for(loop, kill, 60)
    figures = [fields["broker_units"] for kind, fields in loop.events if kind == "kill_disagrees"]
    assert figures == [4, 3]
