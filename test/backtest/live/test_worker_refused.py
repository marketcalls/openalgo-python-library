"""The order worker's refusals and its end (`worker`, `_worker_send`; 16-session-design 2.4,
4.5; the unit 8 hand-offs in the merge notes): a protective request refused for good is
alerted in words, a cancel no read can ever settle is settled as refused once (its one
alert was the refusal's), and a step after `close` does nothing.

The worker runs with the fake's real SDK clients, a real mode guard and a real journal;
the test drives `step()` on the fake's manual clock (`worker_samples`).
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._types import AlertKind
from openalgo.backtest._live._venue_requests import OrderTerms, Priority
from openalgo.backtest._live.inbox import NotSent, NotSentReason, Settled

#: An order ID the session never sent.
FOREIGN_ORDER = "260101000000001"


@pytest.fixture
def rig(fake_openalgo, tmp_path):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    made = WorkerRig(fake_openalgo, tmp_path)
    made.messages()
    yield made
    made.worker.close()


def steps(rig: WorkerRig, seconds: float) -> list:
    """Step every half second for `seconds`; what was posted."""
    posted = []
    for _ in range(round(seconds / 0.5)):
        rig.clock.advance(0.5)
        rig.worker.step()
        posted += rig.messages()
    return posted


@pytest.mark.parametrize(
    ("make", "words"),
    [
        (lambda rig: rig.cancel(rig.stop(), FOREIGN_ORDER, Priority.PROTECT), "cancel"),
        (
            lambda rig: rig.modify(
                rig.stop(), FOREIGN_ORDER, OrderTerms(5, "SL-M", 0.0, 800.0), Priority.PROTECT
            ),
            "modify",
        ),
    ],
    ids=["cancel", "modify"],
)
def test_a_protective_refusal_names_what_was_refused_in_words(rig, make, words):
    rig.run(make(rig))

    ((kind, text),) = rig.alerts.sent
    assert kind is AlertKind.ERROR
    assert f"a protective {words} of ema_cross" in text
    assert "Request" not in text


def test_a_cancel_no_read_can_settle_is_settled_as_refused_once(rig):
    cancel = rig.cancel(rig.stop(), FOREIGN_ORDER, Priority.PROTECT)

    posted = rig.run(cancel) + steps(rig, 12.0)

    assert [(n.request, n.reason) for n in posted if isinstance(n, NotSent)] == [
        (cancel, NotSentReason.REFUSED)
    ]
    assert [(s.request, s.check) for s in posted if isinstance(s, Settled)] == [(cancel, None)]
    assert rig.fake.log.calls("orderstatus") == []
    assert rig.alerts.kinds() == [AlertKind.ERROR]


def test_a_cancel_its_reads_cannot_settle_is_alerted_once_after_the_third_read(rig):
    place = rig.place(price_type="LIMIT", price=700.0)
    (answer,) = [m for m in rig.run(place) if not isinstance(m, NotSent)][:1]
    order_id = answer.outcome.order_id
    rig.fake.faults.expire_session()
    cancel = rig.cancel(place, order_id, Priority.PROTECT)

    posted = rig.run(cancel) + steps(rig, 20.0)

    reads = rig.fake.log.calls("orderstatus")
    errors = [text for kind, text in rig.alerts.sent if kind is AlertKind.ERROR]
    assert len(reads) >= 4
    assert len(errors) == 1 and order_id in errors[0]
    assert [m for m in posted if isinstance(m, Settled)] == []


def test_a_step_after_close_posts_nothing_and_calls_nothing(rig):
    rig.fake.faults.bad_gateway_page("placeorder")
    rig.run(rig.place())
    rig.worker.close()
    rig.clock.advance(6)
    calls_before = len(rig.fake.log.calls())

    took = rig.worker.step()

    assert (took, rig.messages()) == (False, [])
    assert len(rig.fake.log.calls()) == calls_before
