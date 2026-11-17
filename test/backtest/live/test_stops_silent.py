"""A silent OpenAlgo while a stop or an exit waits, on the fake OpenAlgo
(`_live/_stops_silent.py`; 16-session-design 4.9, unit 10): after a dropped connection
the order path sends nothing until OpenAlgo answers a call again, so while a protective
request waits and no other call is due the session asks the order worker for a probe
every 2 s, and the request goes on the first answer; the silence is alerted once it has
lasted 10 s, at most once a minute; with nothing waiting no probe is asked.

`RestsAnEntry` rests a LIMIT buy of 10 SBIN at 499.00 with a stop 3 points below from
14:42:01 IST: the stream speaks for it, so the order path polls it every 30 s and calls
OpenAlgo for nothing else. From 14:43 the connection of the next poll drops, so OpenAlgo
is silent for the order path from that poll (about 14:43:02). At 14:43:10 a trade at
499.00 fills the entry, so the kernel places its stop, and that placement waits in the
order worker's queue for OpenAlgo to answer. `StopHold` instead holds 10 SBIN from
14:42:01 with its stop resting, and the poll that drops is the stop's: nothing waits.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, ist
from stops_samples import STOP_POINTS, StopHold, check_invariants, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._order_state import SAFETY_POLL_EVERY_NS
from openalgo.backtest._live._stops_silent import (
    PROBE_EVERY_S,
    SILENT_ALERT_AFTER_S,
    SILENT_ALERT_EVERY_S,
    SilentServer,
)
from openalgo.backtest._live._types import LOGGER_NAME, OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import PlaceRequest, Priority
from openalgo.backtest._live._worker_queue import Probe

#: When the poll whose connection drops is armed, and when the resting entry fills.
SILENCE_ARMED = ist(14, 43)
ENTRY_FILLED = ist(14, 43, 10)
#: The resting entry's limit, rupees.
ENTRY_LIMIT = 499.0
#: The session's loop waits at most this long between two steps in these tests
#: (`session_samples.WAIT_MAX_S`): a probe asked in one step is made in the next.
STEP_S = 1.0
#: What the silence alert begins with.
SILENT_WORDS = "OpenAlgo has not answered for"


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.script.at(SILENCE_ARMED, lambda: world.fake.faults.drop_connection("orderstatus"))
    world.stop_at(ist(14, 46))
    return world


def probes_fail(world, count: int) -> None:
    """The connection of the next `count` order-book reads drops (from 14:43)."""

    def arm():
        for _ in range(count):
            world.fake.faults.drop_connection("orderbook")

    world.script.at(SILENCE_ARMED, arm)


class RestsAnEntry(bt.Strategy):
    """Rests a LIMIT buy of 10 SBIN at `ENTRY_LIMIT` with a stop 3 points below at its
    first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, price=ENTRY_LIMIT, stop=bt.points(STOP_POINTS))


def the_entry_fills(world) -> None:
    """A trade at the entry's limit at `ENTRY_FILLED` fills the resting entry."""
    world.script.at(ENTRY_FILLED, lambda: world.fake.trade("SBIN", "NSE", ENTRY_LIMIT, volume=10.0))


def silent_from(world) -> int:
    """When the order path's poll whose connection dropped was made (UTC nanoseconds)."""
    return next(c.ts_ns for c in world.fake.log.calls("orderstatus") if c.ts_ns >= SILENCE_ARMED)


def probes(world) -> list[int]:
    """When each order-book read of the session was made from 14:43 alone: its probes (a
    reconciliation reads the position book with it, 16-session-design 5.2)."""
    books = {c.ts_ns for c in world.fake.log.calls("positionbook")}
    reads = world.fake.log.calls("orderbook")
    return [c.ts_ns for c in reads if c.ts_ns >= SILENCE_ARMED and c.ts_ns not in books]


def stop_placements(world) -> list[int]:
    """When each SL-M stop was placed at the fake."""
    return [o.placed_ns for o in world.orders() if o.price_type == "SL-M"]


def stop_statuses(world) -> list[str]:
    """The status of each SL-M stop at the fake."""
    return [o.status for o in world.orders() if o.price_type == "SL-M"]


def silence_alerts(world) -> list:
    """The silence alerts as the Telegram endpoint took them: text and time."""
    return [m for m in world.fake.telegram_sent if SILENT_WORDS in m.message]


# ----- a stop waits ----------------------------------------------------------------


def test_a_waiting_stop_has_openalgo_probed_every_2_seconds_and_goes_on_the_first_answer(world):
    probes_fail(world, 2)
    the_entry_fills(world)
    outcome = world.run(RestsAnEntry, on_stop="keep")

    made = probes(world)
    gaps = [b - a for a, b in zip(made, made[1:], strict=False)]
    assert len(made) == 3
    assert ENTRY_FILLED < made[0] <= ENTRY_FILLED + round((PROBE_EVERY_S + STEP_S) * NS)
    assert all(PROBE_EVERY_S * NS <= gap <= (PROBE_EVERY_S + STEP_S) * NS for gap in gaps)
    (placed,) = stop_placements(world)
    assert made[-1] < placed < made[-1] + round(PROBE_EVERY_S * NS)
    assert stop_statuses(world) == ["trigger pending"]
    # The leg's unprotected window ran while the stop waited (4.2).
    (held,) = [w for w in outcome.unprotected if w.reason == "stop held"]
    assert held.from_ns <= made[0] and held.to_ns >= placed
    check_invariants(world.timeline, strategy="RestsAnEntry")


def test_the_silence_is_alerted_once_at_10_seconds_with_the_positions_and_their_stops(world):
    probes_fail(world, 2)
    the_entry_fills(world)
    world.run(RestsAnEntry, on_stop="keep")

    (alert,) = silence_alerts(world)
    since = silent_from(world)
    assert since + SILENT_ALERT_AFTER_S * NS <= alert.ts_ns
    assert alert.ts_ns <= since + (SILENT_ALERT_AFTER_S + STEP_S) * NS
    assert alert.message.endswith(
        f"{SILENT_WORDS} 10 s: 1 position open, 1 without a confirmed stop at the broker"
    )


def test_a_long_silence_is_alerted_again_at_most_once_a_minute(world):
    # Every probe fails for about 80 s.
    probes_fail(world, 40)
    the_entry_fills(world)
    world.run(RestsAnEntry, on_stop="keep")

    first, second = silence_alerts(world)
    assert second.ts_ns - first.ts_ns >= SILENT_ALERT_EVERY_S * NS
    assert f"{SILENT_WORDS} 10 s" in first.message
    assert f"{SILENT_WORDS} 70 s" in second.message
    (placed,) = stop_placements(world)
    assert placed > probes(world)[-1]
    assert stop_statuses(world) == ["trigger pending"]


# ----- nothing waits -----------------------------------------------------------------


def test_a_silent_openalgo_with_nothing_waiting_is_not_probed(world):
    world.run(StopHold, on_stop="keep")

    since = silent_from(world)
    answered = [c.ts_ns for c in world.fake.log.calls("orderstatus") if c.ts_ns > since]
    assert probes(world) == []
    # The stop's next poll, 30 s on, is what OpenAlgo answers.
    assert answered and answered[0] >= since + SAFETY_POLL_EVERY_NS
    (alert,) = silence_alerts(world)
    assert alert.message.endswith("1 position open, 0 without a confirmed stop at the broker")


# ----- one probe at a time, on its own (no server) ------------------------------------


class _Worker:
    """The order worker's view of a silent OpenAlgo since 0, its last call at `asked_ns`."""

    def __init__(self) -> None:
        self.silent_since_ns: int | None = 0
        self.asked_ns: int | None = 0


class _Loop:
    """The session loop as the probe sees it: the requests handed over, at `clock`."""

    def __init__(self) -> None:
        self.parts = SimpleNamespace(worker=_Worker())
        self.traders = ()
        self.handed: list = []

    def hand_over(self, requests):
        self.handed.extend(requests)

    def alert(self, kind, text):
        pass


def held_stop(loop: _Loop) -> SilentServer:
    """A stop placement the worker put back, waiting on the silent OpenAlgo."""
    silence = SilentServer(loop, stops=None)
    intent = OrderIntent(
        intent_id="20261116-live-1-StopHold-o2",
        strategy="StopHold",
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        price_type="SL-M",
        product="MIS",
        purpose=Purpose.STOP,
        created_ns=0,
        trigger_price=497.0,
    )
    silence.held(PlaceRequest(1, 2, intent, Priority.PROTECT))
    return silence


def test_a_probe_not_made_yet_is_not_asked_for_again():
    # The worker is inside a long call: the probe asked at 2 s is not made by 10 s.
    loop = _Loop()
    silence = held_stop(loop)

    for second in range(2, 11):
        silence.due(second * NS)

    assert loop.handed == [Probe()]


def test_a_probe_is_asked_for_again_2_seconds_after_the_last_call():
    loop = _Loop()
    silence = held_stop(loop)
    silence.due(2 * NS)
    loop.parts.worker.asked_ns = 3 * NS  # the probe was made, and not answered

    silence.due(4 * NS)
    silence.due(5 * NS)

    assert loop.handed == [Probe(), Probe()]


def test_no_probe_is_asked_once_the_stop_was_sent():
    loop = _Loop()
    silence = held_stop(loop)
    silence.done(1)

    silence.due(5 * NS)

    assert loop.handed == []
