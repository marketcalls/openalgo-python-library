"""A stop GTT that fires on a gap while its modify is cancelled to be placed again
(`_live/_gtt_send.py`, `_gtt_follow.py`; 20-positional-live 3.1 "Unanswered calls", 3.2;
the lead's ruling N3b-0 (a)).

The leg: another client of the account bought 10 SBIN at 800.00 (the session's start read
gives the follower its reference), and the stop GTT sells 10 at a trigger of 790.00, its
child's limit 780.00. The session's own exit sells 5; the stop is modified to the 5 left.
The broker makes the modify, but the GTT book still shows the old 10 units while the
confirmation reads it, so the modify cannot be told: the sender cancels the GTT to place
it again (REPLACE_CANCEL). Just as that cancel is sent SBIN gaps to 775.00, below the
child's limit: the stop fires, its child (sell 5 LIMIT 780.00, the modify's units) is
listed in the order book and rests unfilled, and the cancel gets a 502.

The child is the stop's, by the modify's terms: the cancel is refused for the replace (the
answer is UNRESOLVED and nothing is placed again), and the follower finds the child by
either set of terms (TRIGGERED). SBIN then recovers to 785.00 (the child fills: flat) and
falls to 789.00 (below the trigger: with a second stop placed beside the first, it would
sell 5 more, ending short 5). Expected values are worked from the scenario by hand.
"""

from __future__ import annotations

import copy

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from gtt_samples import FIRST_TRIGGER, GttRig, answers, bought, broker_with_sbin, outcomes

from openalgo.backtest._live._gtt_child import CHILD_SEARCH_S
from openalgo.backtest._live._gtt_types import GttAction, GttAnswerKind, GttEvent, ReadGtts

#: The gap below the child's 780.00 limit, the recovery above it, and the fall below the
#: 790.00 trigger, rupees.
GAP, RECOVERY, FALL = 775.0, 785.0, 789.0
#: The fake broker's order IDs on Monday 2026-11-16: the buy, the session's exit, the child.
CHILD = "261116000000003"

SHAPES = pytest.mark.parametrize("history", [True, False], ids=["history rows", "active only"])
PRODUCTS = pytest.mark.parametrize("product", ["CNC", "NRML"])


def stale_until_cancel(rig, fake, monkeypatch, *, answer=502):
    """The GTT book shows the stop's old 10 units until the replace's cancel is sent; the
    cancel's call meets the gap first (the stop fires) and is answered `answer`: a 502, or
    None for the broker's own refusal of a GTT no longer active."""
    client = rig.orders._client
    book, cancel = client.gttorderbook, client.cancelgttorder
    stale = {"on": True}

    def stale_rows(**fields):
        answer = book(**fields)
        if not stale["on"]:
            return answer
        rows = copy.deepcopy(answer["data"])
        for row in rows:
            for leg in row.get("legs", []):
                leg["quantity"] = 10
        return {**answer, "data": rows}

    def fired_then_cancel(**fields):
        stale["on"] = False
        fake.trade("SBIN", "NSE", GAP)
        if answer is not None:
            fake.faults.gateway_error("cancelgttorder", status=answer)
        return cancel(**fields)

    monkeypatch.setattr(client, "gttorderbook", stale_rows)
    monkeypatch.setattr(client, "cancelgttorder", fired_then_cancel)


def replaced_on_a_gap(make_fake_openalgo, tmp_path, monkeypatch, *, history, product, answer=502):
    """The scenario of the module docstring up to the stop modified to 5 (the replace's
    cancel answered `answer`, `stale_until_cancel`); the fake, the rig and the news of the
    modify and the 200 s after it."""
    fake = broker_with_sbin(make_fake_openalgo(gtt_history=history))
    rig = GttRig(fake, tmp_path)
    rig.messages()
    bought(fake, product=product)
    rig.start_read(10, product)
    (placed,) = answers(rig.send(rig.gtt(product=product)))
    rig.run(rig.own_exit(5, product=product))
    rig.step_for(40.0)
    rig.send(ReadGtts())
    stale_until_cancel(rig, fake, monkeypatch, answer=answer)
    news = rig.send(rig.gtt(GttAction.MODIFY, seq=placed.seq, units=5, product=product))
    news += rig.step_for(200.0, every_s=2.0)
    return fake, rig, news


@PRODUCTS
@SHAPES
def test_a_stop_fired_on_a_gap_during_its_replace_is_not_placed_again(
    make_fake_openalgo, tmp_path, monkeypatch, history, product
):
    fake, rig, news = replaced_on_a_gap(
        make_fake_openalgo, tmp_path, monkeypatch, history=history, product=product
    )

    assert [answer.kind for answer in answers(news)] == [GttAnswerKind.UNRESOLVED]
    assert [(g.trigger_id, g.status, g.quantity) for g in fake.gtts()] == [
        (FIRST_TRIGGER, "triggered", 5)
    ]
    fake.trade("SBIN", "NSE", RECOVERY)
    assert fake.position("SBIN", "NSE") == 0
    fake.trade("SBIN", "NSE", FALL)
    assert fake.position("SBIN", "NSE") == 0
    rig.worker.close()


@PRODUCTS
@SHAPES
def test_the_child_of_a_stop_fired_during_its_replace_is_found_by_the_modifys_units(
    make_fake_openalgo, tmp_path, monkeypatch, history, product
):
    _, rig, news = replaced_on_a_gap(
        make_fake_openalgo, tmp_path, monkeypatch, history=history, product=product
    )

    found = [(o.event, o.order_id) for o in outcomes(news)]
    assert found == [(GttEvent.IN_FLIGHT, ""), (GttEvent.TRIGGERED, CHILD)]
    rig.worker.close()


@PRODUCTS
@SHAPES
def test_a_stop_whose_replace_cancel_is_refused_as_fired_finds_its_child_by_the_modifys_units(
    make_fake_openalgo, tmp_path, monkeypatch, history, product
):
    # The broker refuses the cancel outright: the stop is no longer active, it fired.
    _, rig, news = replaced_on_a_gap(
        make_fake_openalgo, tmp_path, monkeypatch, history=history, product=product, answer=None
    )

    assert [answer.kind for answer in answers(news)] == [GttAnswerKind.UNRESOLVED]
    found = [(o.event, o.order_id) for o in outcomes(news)]
    assert found == [(GttEvent.IN_FLIGHT, ""), (GttEvent.TRIGGERED, CHILD)]
    rig.worker.close()


def test_a_history_row_that_says_triggered_is_never_ended_outside_the_session(
    make_fake_openalgo, tmp_path
):
    # The stop fires and its child sells 10 at 789.00, but the order book never lists the
    # child and the position book never shows the sale: the row says triggered.
    fake = broker_with_sbin(make_fake_openalgo(gtt_history=True))
    rig = GttRig(fake, tmp_path)
    rig.messages()
    bought(fake)
    rig.start_read(10)
    rig.send(rig.gtt())
    fake.lag_positionbook()
    fake.trade("SBIN", "NSE", FALL)
    fake.hide_order(fake.orders()[-1].order_id, reads=1000)

    news = rig.send(ReadGtts()) + rig.step_for(CHILD_SEARCH_S + 4.0, every_s=2.0)

    assert [o.event for o in outcomes(news)] == [GttEvent.IN_FLIGHT, GttEvent.MISMATCH]
    assert "triggered" in outcomes(news)[-1].message
    rig.worker.close()
