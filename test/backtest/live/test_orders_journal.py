"""The order path with the real journal file and mode guard: what a restart takes back,
an unbound intent settled in the file, a semi-auto queued intent across a restart, and
restore called before the guard has started."""

from dataclasses import dataclass

import pytest
from journal_samples import ACCOUNT, BROKER, DAY, journal_path
from orders_fakes import (
    SERVER_502,
    START_IST,
    Clock,
    FakeAlerts,
    FakeClient,
    book,
    book_row,
    make_intent,
    placed,
)

from openalgo.backtest._live._types import AlertKind, Mode, Purpose
from openalgo.backtest._live.guard import ModeGuard
from openalgo.backtest._live.journal import Journal
from openalgo.backtest._live.orders import OrderPath, RateBudget, Resolved
from openalgo.backtest.errors import LiveError

SEMI_AUTO = {
    "status": "success",
    "message": "Order queued for approval in Action Center",
    "mode": "semi_auto",
    "pending_order_id": 17,
}


class LiveClient(FakeClient):
    """A scripted client whose analyzer switch is off: OpenAlgo is in LIVE mode."""

    def analyzerstatus(self):
        return {"status": "success", "data": {"mode": "live", "analyze_mode": False}}


@dataclass
class Session:
    """One run of the engine on the journal file: its journal, guard, alerts and path."""

    journal: Journal
    guard: ModeGuard
    alerts: FakeAlerts
    path: OrderPath


@pytest.fixture
def start(tmp_path):
    """Start a session on one journal file, closing every journal at the end."""
    opened = []

    def start_session(client, clock, *, start_guard=True):
        journal = Journal.open(
            journal_path(tmp_path), account=ACCOUNT, broker=BROKER, mode=Mode.LIVE, now=clock.now
        )
        opened.append(journal)
        guard = ModeGuard(client, now=clock.now)
        if start_guard:
            guard.start()
        journal.start_session(DAY, Mode.LIVE, BROKER)
        alerts = FakeAlerts()
        path = OrderPath(
            client,
            journal=journal,
            guard=guard,
            alerts=alerts,
            now=clock.now,
            sleep=clock.sleep,
            rate=RateBudget(clock.now),
        )
        return Session(journal, guard, alerts, path)

    yield start_session
    for journal in opened:
        journal.close()


def test_a_restart_takes_back_an_order_from_the_journal_file_to_cancel_it(start):
    clock = Clock()
    start(LiveClient(placeorder=[placed("1")]), clock).path.send(make_intent("i-1"))
    client = LiveClient(cancelorder=[placed("1")])
    restarted = start(client, clock)

    restarted.path.restore()

    assert restarted.path.cancel("1").order_id == "1"
    assert client.calls_to("cancelorder") == [{"order_id": "1", "strategy": "ema_cross"}]


def test_an_unbound_intent_is_bound_in_the_journal_file_from_the_order_book(start):
    clock = Clock()
    start(LiveClient(placeorder=[SERVER_502]), clock).path.send(make_intent("i-1"))
    restarted = start(LiveClient(orderbook=[book(book_row("555", timestamp=START_IST))]), clock)
    restarted.path.restore()

    (resolution,) = restarted.path.resolve_unbound()

    assert (resolution.kind, resolution.order_id) == (Resolved.BOUND, "555")
    assert restarted.journal.order_id_of("i-1") == "555"


def test_an_order_another_intent_owns_in_the_file_is_never_bound_after_a_restart(start):
    clock = Clock()
    first = start(LiveClient(placeorder=[placed("111"), SERVER_502]), clock)
    first.path.send(make_intent("i-1"))
    first.path.send(make_intent("i-2", purpose=Purpose.EXIT))
    clock.advance(200)
    restarted = start(LiveClient(orderbook=[book(book_row("111", timestamp=START_IST))] * 2), clock)
    restarted.path.restore()

    resolutions = restarted.path.resolve_unbound()
    clock.advance(1)
    resolutions += restarted.path.resolve_unbound()

    assert [(r.intent.intent_id, r.kind) for r in resolutions] == [("i-2", Resolved.NOT_PLACED)]
    assert restarted.journal.order_id_of("i-1") == "111"


def test_a_semi_auto_intent_stays_queued_across_a_restart_and_is_never_marked_not_placed(start):
    clock = Clock()
    first = start(LiveClient(placeorder=[SEMI_AUTO]), clock)
    first.path.send(make_intent("i-1"))
    assert first.guard.locked
    clock.advance(300)
    client = LiveClient(orderbook=[book()] * 2)
    restarted = start(client, clock)

    restarted.path.restore()
    resolutions = restarted.path.resolve_unbound()
    clock.advance(1)
    resolutions += restarted.path.resolve_unbound()

    assert resolutions == []
    assert client.calls_to("orderbook") == []
    assert [queued.intent.intent_id for queued in restarted.journal.queued()] == ["i-1"]
    ((kind, text),) = restarted.alerts.sent
    assert kind is AlertKind.ERROR
    assert "i-1" in text and "reject pending order 17" in text


def test_restore_with_nothing_to_take_back_still_needs_the_guard_started(start):
    restarted = start(LiveClient(), Clock(), start_guard=False)

    with pytest.raises(LiveError, match=r"call guard\.start\(\)"):
        restarted.path.restore()


def test_restore_before_the_guard_has_started_raises_and_takes_back_nothing(start):
    clock = Clock()
    start(LiveClient(placeorder=[placed("1")]), clock).path.send(make_intent("i-1"))
    restarted = start(LiveClient(), clock, start_guard=False)

    with pytest.raises(LiveError, match=r"call guard\.start\(\), then journal\.start_session"):
        restarted.path.restore()

    with pytest.raises(LiveError, match="was not sent by this session"):
        restarted.path.cancel("1")
