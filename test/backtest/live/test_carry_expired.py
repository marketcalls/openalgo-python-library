"""A position carried on a contract whose last trading day has passed, at the start of a later
day, on the fake OpenAlgo (20-positional-live 9 and 4.2 step 4; the lead's rulings N5a-1 and
N5a-3, review rounds 1 and 2 of unit N5a): an index future is settled by the exchange in
cash, so the start ends its stop GTT in the journal, names the settlement (a SESSION alert
and a warning of the result, which say that no result counts its profit or loss after its
last close) and starts the strategy flat on the contract it trades now, never looking the
expired one up; the close writes its flat day-close record, so the settlement is named
once. A contract settled by delivery is named so, asking the operator to check the
account. A strategy moved to another contract while its carried one still trades is refused
before anything is sent, saying what to do.

On Monday 2026-11-16 `NovemberKept` buys one lot of 65 units of November's NIFTY future
(`NIFTY23NOV26FUT`, its last trading day Monday 2026-11-23) at 120.00 in NRML with a stop
GTT 10 points below, carried at NSE's close. No session runs until Wednesday 2026-11-25
(Tuesday 2026-11-24 is a holiday); the exchange settles the contract after its last day,
which ends its GTT at the broker. On Wednesday the same strategy trades December's future
(`NIFTY29DEC26FUT`); its `self.state` keeps its entry, so it buys nothing more. The
session starts at 14:40:05 IST and is stopped at 14:46.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest
from carry_samples import (
    AFTER_NOVEMBERS_LAST,
    DAY1,
    DAY2,
    DAY3,
    DAY4,
    FUTURE,
    NOVEMBER_FUTURE,
    TO_NOVEMBERS_LAST,
    FutKept,
    NovemberKept,
    carried_world,
    journal_of,
    list_future,
    missed_day,
    on,
    to_day,
)
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls
from session_samples import alerts

from openalgo.backtest._live import _carry_contracts as carry_contracts
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import PreflightError

#: November's contract's last trading day.
LAST_DAY = TO_NOVEMBERS_LAST[-1]
#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class OnDecember(FutKept):
    """The strategy `NovemberKept` moved to December's future (its class name is the
    strategy's name)."""


OnDecember.__name__ = OnDecember.__qualname__ = "NovemberKept"


def wednesday(world, strategy=OnDecember, *, settled=True):
    """No session from Tuesday to November's last trading day, the contract settled by the
    exchange after it when `settled`, then Wednesday's session of `strategy`, stopped at
    14:46."""
    for day in (DAY2, DAY3, DAY4, *TO_NOVEMBERS_LAST):
        missed_day(world, day, ())
    if settled:
        world.fake.settle_contract(NOVEMBER_FUTURE, "NFO")
    to_day(world, AFTER_NOVEMBERS_LAST)
    world.stop_at(on(AFTER_NOVEMBERS_LAST, 14, 46))
    return world.run(strategy)


def november_carried(make_fake_openalgo, monkeypatch, tmp_path, **options):
    """Monday's session of `NovemberKept` carried at the close; from then on the fake also
    lists and trades December's future."""
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, NovemberKept, future=NOVEMBER_FUTURE, **options
    )
    world.trace.waive("protected", "the exchange settles the contract and ends its stop GTT")
    list_future(world, FUTURE)
    return world


def settled_words(how):
    """The words of the settlement, worked by hand from the scenario, settled `how`."""
    return (
        f"NovemberKept: 65 {NOVEMBER_FUTURE} NFO (NRML) carried from {DAY1} expired on "
        f"{LAST_DAY}: the exchange settled it {how} (its last close in the journal 120.00, of "
        f"{DAY1}; the contract note holds the settlement price, and no result counts its "
        f"profit or loss after that close), so it starts flat on {FUTURE} NFO"
    )


#: The words of the index future's settlement in cash.
SETTLED = settled_words("in cash")


@MODES
@BOOKS
def test_a_carried_index_future_past_its_last_day_is_named_settled_and_the_start_goes_on(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = november_carried(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    sent_before = len(calls(world, *SENDS))

    outcome = wednesday(world)

    assert outcome.stopped_by == "ctrl_c"
    assert f"{world.fake.mode.name} (NovemberKept): {SETTLED}" in alerts(caplog, "session")
    assert SETTLED in outcome.warnings
    assert calls(world, *SENDS)[sent_before:] == []
    wednesday_start = on(AFTER_NOVEMBERS_LAST, 14, 40, 5)
    looked_up = [c for c in calls(world, "symbol") if c.ts_ns >= wednesday_start]
    assert [c.detail["request"]["symbol"] for c in looked_up] == [FUTURE]
    with journal_of(world) as journal:
        assert journal.gtt_by_trigger(trigger_id).state.terminal
    assert not alerts(caplog, "error")


class DeliveryTables:
    """The package's reference tables, with November's future settled by delivery (as a
    stock future is): the scenario's contract stands in for one."""

    def __init__(self, tables):
        self._tables = tables

    def instrument(self, symbol, exchange, day):
        found = self._tables.instrument(symbol, exchange, day)
        if symbol != NOVEMBER_FUTURE:
            return found
        return dataclasses.replace(found, settlement="physical")


def test_a_carried_contract_settled_by_delivery_asks_to_check_the_account(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = november_carried(make_fake_openalgo, monkeypatch, tmp_path)
    tables = DeliveryTables(carry_contracts.default_tables())
    monkeypatch.setattr(carry_contracts, "default_tables", lambda: tables)

    outcome = wednesday(world)

    delivered = settled_words(
        "by delivery: check the account for the shares or the obligation it left"
    )
    assert delivered in outcome.warnings
    assert f"{world.fake.mode.name} (NovemberKept): {delivered}" in alerts(caplog, "session")


def test_the_close_writes_a_flat_record_so_the_settled_contract_is_not_carried_again(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = november_carried(make_fake_openalgo, monkeypatch, tmp_path)

    wednesday(world)

    with journal_of(world) as journal:
        record = journal.day_closes()["NovemberKept"]
    assert record.day == AFTER_NOVEMBERS_LAST
    assert not [p for p in record.carried["positions"] if p["quantity_units"]]


def test_a_strategy_moved_off_a_carried_contract_that_still_trades_is_refused(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Monday's November future still trades on Wednesday 2026-11-18: what it carries
    # continues only on its own contract.
    world = november_carried(make_fake_openalgo, monkeypatch, tmp_path)
    missed_day(world, DAY2, ())
    to_day(world, DAY3)
    sent_before = len(calls(world, *SENDS))

    with pytest.raises(PreflightError) as refused:
        world.run(OnDecember)

    assert refused.value.failures == (
        f"NovemberKept: what it carries from {DAY1} is 65 {NOVEMBER_FUTURE} NFO (NRML), and "
        f"this start trades {FUTURE} NFO: a carried position continues only on its own "
        f"contract, so start it on {NOVEMBER_FUTURE} NFO; or close the position and cancel "
        "its GTTs in OpenAlgo by hand, then start it under another account name (account=)",
    )
    assert calls(world, *SENDS)[sent_before:] == []
