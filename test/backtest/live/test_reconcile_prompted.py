"""A comparison prompted by another client's fill on a traded symbol, on the fake OpenAlgo
(`_live/_worker_news.py`, `_live/reconcile.py`; 16-session-design 5.1 and 7.1 "As built
(unit 14", the lead's decision of 2026-11-18; unit 16).

`StopHold` holds 10 SBIN from 14:42:01 IST with an SL-M stop of 10 at 497.00. Another
client of the account sells 4 SBIN at market at `SOLD_AT`, just after one of the
session's scheduled reads, so the next one is most of `RECONCILE_EVERY_S` away. The order
stream reports that client's fill, and the session asks for a comparison
`RECONCILE_AFTER_FILL_S` later, as after its own fills: that one sees the sale, the next
(3 s after a comparison that differed) confirms it, and the stop is cut to the 6 left. A
fill on a symbol the session does not trade (INFY) prompts nothing. Expected values come
from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, WAIT_MAX_S, ist
from stops_samples import StopHold, events, sold_by_hand, stop_world

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.reconcile import RECONCILE_AFTER_FILL_S, RECONCILE_AGAIN_AFTER_S

#: When the other client sells: 14:43:10, five seconds after a scheduled read was asked
#: for, so its books were read before the sale (the session's reads come every 30 s from
#: the one asked for 3 s after the entry's fill, at 14:42:05).
SOLD_AT = ist(14, 43, 10)
#: The units sold by hand, of the 10 the session holds.
SOLD = 4
#: The longest a fit may take after the sale: the prompted comparison, the one that
#: confirms it, each asked for within one loop wait of its due time, and the cut sent and
#: answered within one more.
FIT_WITHIN_NS = round((RECONCILE_AFTER_FILL_S + RECONCILE_AGAIN_AFTER_S + 3 * WAIT_MAX_S) * NS)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = stop_world(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    made.stop_at(ist(14, 46))
    return made


def read_times(outcome) -> list[int]:
    """When the session asked the order worker for each book read."""
    return [e["ts_ns"] for e in events(outcome, "request") if e["type"] == "ReadBooks"]


def stop_cut_at(world) -> int:
    """When the fake's stop of SBIN was cut to the units the sale left."""
    return next(
        change.ts_ns
        for change in world.timeline.changes
        if change.price_type == "SL-M" and change.quantity == StopHold.qty - SOLD
    )


def other_client_buys_infy(world, at_ns: int) -> None:
    other = world.fake.client()
    world.script.at(
        at_ns,
        lambda: other.placeorder(
            strategy="manual", symbol="INFY", action="BUY", exchange="NSE",
            price_type="MARKET", product="MIS", quantity=5,
        ),
    )  # fmt: skip


def test_another_clients_fill_on_the_traded_symbol_prompts_a_read_after_the_fill_delay(world):
    sold_by_hand(world, SOLD, SOLD_AT)
    outcome = world.run(StopHold)
    after = [at for at in read_times(outcome) if at > SOLD_AT]
    assert after[0] >= SOLD_AT + RECONCILE_AFTER_FILL_S * NS
    assert after[0] <= SOLD_AT + round((RECONCILE_AFTER_FILL_S + WAIT_MAX_S) * NS)


def test_a_sale_by_hand_of_4_of_10_is_fitted_within_the_two_prompted_comparisons(world):
    sold_by_hand(world, SOLD, SOLD_AT)
    world.run(StopHold)
    assert 0 < stop_cut_at(world) - SOLD_AT <= FIT_WITHIN_NS


def test_a_fill_on_a_symbol_the_session_does_not_trade_prompts_nothing(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The same session twice, the second with another client's INFY fill at `SOLD_AT`:
    # the reads come at the same instants.
    times = []
    for bought in (False, True):
        made = stop_world(
            make_fake_openalgo, monkeypatch, tmp_path / str(bought), symbols=("SBIN", "INFY")
        )
        made.stop_at(ist(14, 46))
        if bought:
            other_client_buys_infy(made, SOLD_AT)
        times.append(read_times(made.run(StopHold)))
        assert made.fake.position("INFY", "NSE") == (5 if bought else 0)
    assert times[0] == times[1]
