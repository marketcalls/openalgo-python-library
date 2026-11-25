"""A finding of the GTT follower is applied to a strategy's kernel GTT only when it is about
the journal's GTT bound to that kernel GTT (review finding 1 of round 3 of unit N3b):
kernel GTT ids start again at 1 with each kernel session, and the follower follows the
journal's GTTs of every day, so a finding about an earlier day's GTT can name an id the
session's own GTT has too. It is matched by the journal's number (`seq`) as well, in the
GTT map (`_venue_gtts.GttMap.found`) and once the kernel's session has closed
(`_session_carry.CarriedLegs.after_close`).

The kernel session is played by hand, as `test_kernel_gtts` plays it: 10 SBIN bought in
CNC at 500.00 with a stop 3 points below, its GTT g1 confirmed as the journal's GTT 7,
trigger 261117000003. The earlier day's GTT is the journal's GTT 1, trigger 261116000001,
which that day's kernel session also numbered g1.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

from test_kernel_gtts import G1, STOP_3, at, holding

from openalgo.backtest._live._gtt_types import GttAnswer, GttAnswerKind, GttEvent, GttOutcome
from openalgo.backtest._live._session_carry import CarriedLegs
from openalgo.backtest._live._types import AlertKind
from openalgo.backtest._live._venue_gtts import GttApplied, GttMap

STRATEGY = "CncHold"
#: The journal's number and the broker's trigger ID of the session's own stop GTT.
OWN_SEQ, OWN_TRIGGER = 7, "261117000003"
#: The same of the earlier day's stop GTT, which that day's kernel also numbered g1.
EARLIER_SEQ, EARLIER_TRIGGER = 1, "261116000001"


def confirmed() -> tuple[Any, GttMap]:
    """The kernel session and its GTT map, g1 placed and confirmed as the journal's GTT
    `OWN_SEQ`."""
    engine = holding(STOP_3)
    (command,) = engine.take_commands()
    gtts = GttMap(
        engine,
        strategy=STRATEGY,
        markets=[("SBIN", "NSE")],
        product="CNC",
        request_ids=itertools.count(1),
        now=lambda: at(1, 5),
    )
    (request,) = gtts.requests(command)
    placed = GttAnswer(request, GttAnswerKind.PLACED, seq=OWN_SEQ, trigger_id=OWN_TRIGGER)
    assert gtts.answered(placed).applied is GttApplied.PLACED
    return engine, gtts


def ended(seq: int, trigger_id: str) -> GttOutcome:
    """The follower's finding that the journal's GTT `seq`, kernel g1, ended by hand."""
    return GttOutcome(
        seq=seq,
        strategy=STRATEGY,
        gtt_id=G1,
        trigger_id=trigger_id,
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
        event=GttEvent.ENDED_OUTSIDE,
        message="cancelled",
    )


def test_a_finding_about_an_earlier_days_gtt_of_the_same_kernel_id_applies_nothing():
    engine, gtts = confirmed()

    handled = gtts.found(ended(EARLIER_SEQ, EARLIER_TRIGGER))

    assert handled.applied is GttApplied.NOTHING
    assert engine.gtt(G1)["state"] == "Active"
    # The kernel places no second stop GTT beside the one resting.
    assert engine.take_commands() == []


def test_a_finding_about_the_sessions_own_gtt_is_applied():
    engine, gtts = confirmed()

    handled = gtts.found(ended(OWN_SEQ, OWN_TRIGGER))

    assert handled.applied is GttApplied.ENDED_OUTSIDE
    (again,) = engine.take_commands()
    assert (again["command"], again["gtt_id"]) == ("place_gtt", 2)


class Session:
    """The part of a `_session_loop.Session` the carried legs use after the close."""

    def __init__(self, trader: Any) -> None:
        self.traders = [trader]
        self.alerts: list[tuple[AlertKind, str]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []

    def alert(self, kind: AlertKind, text: str) -> None:
        self.alerts.append((kind, text))

    def event(self, kind: str, **fields: Any) -> None:
        self.events.append((kind, fields))


def closed_leg() -> tuple[Session, Any, GttMap]:
    """A session whose CNC leg of 10 SBIN is carried, its kernel session closed."""
    _, gtts = confirmed()
    trader = SimpleNamespace(
        name=STRATEGY,
        symbol="SBIN",
        exchange="NSE",
        product="CNC",
        gtts=gtts,
        finished=True,
        open_units=lambda: 10.0,
    )
    return Session(trader), trader, gtts


def test_after_the_close_a_finding_about_an_earlier_days_gtt_leaves_the_leg_confirmed():
    session, trader, gtts = closed_leg()

    CarriedLegs(session).after_close(trader, ended(EARLIER_SEQ, EARLIER_TRIGGER))

    assert gtts.gtt(G1).unconfirmed == ""
    assert [kind for kind, _ in session.alerts] == [AlertKind.REJECT]
    (_, text) = session.alerts[0]
    assert EARLIER_TRIGGER in text


def test_after_the_close_a_finding_about_the_sessions_own_stop_gtt_unconfirms_it():
    session, trader, gtts = closed_leg()

    CarriedLegs(session).after_close(trader, ended(OWN_SEQ, OWN_TRIGGER))

    assert gtts.gtt(G1).unconfirmed
    assert [kind for kind, _ in session.alerts] == [AlertKind.ERROR]
