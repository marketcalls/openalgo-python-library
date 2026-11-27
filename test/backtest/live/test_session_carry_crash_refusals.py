"""The refusals that remain where a session that crashed before its day-close record cannot be
continued safely on a later trading day, on the fake OpenAlgo (20-positional-live 4.2 step 2;
the lead's ruling N5c-2; review round 1 of unit N5c, finding 2): each says what was wrong and
what to do, with the words written out, and nothing is sent.

The days and the strategies are `carry_samples`' and `crash_samples`': on Monday `CncKept`
buys 10 SBIN CNC at 500.00 at 14:42:01 IST with its stop GTT at 497.00 (child limit 492.00),
and the session crashes at 14:46, its last save at 14:45:01. Between the crash and
Tuesday's start the journal is changed as each test says. The broker still holds the 10
units, so each refusal also names them as possibly the user's own.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest
from carry_samples import DAY1, DAY2, CncKept, journal_of, on, to_day
from crash_samples import SENDS, crashed_world
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls

from openalgo.backtest._live._carry_unrecorded import CHECK_BY_HAND
from openalgo.backtest._live._journal_gtt_rows import GttState, GttTerms
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME
from openalgo.backtest.errors import PreflightError

#: What each refusal adds: the broker holds the leg's 10 units, which may be the user's own.
OWN_UNITS = (
    ". The books show 10 SBIN NSE (CNC) held: if any of them are units of your own, the "
    "start cannot tell them from the strategy's and never finds the leg closed, so once the "
    "strategy's units are sold and its GTTs cancelled, start it under another account name "
    "(account=)"
)


def crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer, history=True):
    """Monday's session of `CncKept`, crashed at 14:46 (module docstring)."""
    return crashed_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, analyzer=analyzer, history=history
    )


def change_the_save(world, change):
    """Rewrite `CncKept`'s save of Monday in the journal's file as `change` (a function of
    the saved state's dict) leaves it."""
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    key = (world.fake.mode.value, "CncKept", DAY1.isoformat())
    where = "WHERE ledger = ? AND strategy = ? AND day = ?"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        (state,) = connection.execute(f"SELECT state FROM risk_states {where}", key).fetchone()
        changed = json.loads(state)
        change(changed)
        connection.execute(f"UPDATE risk_states SET state = ? {where}", (json.dumps(changed), *key))


def refused_on_tuesday(world):
    """Tuesday's start, refused: its failures; nothing is sent."""
    to_day(world, DAY2)
    sent = len(calls(world, *SENDS))
    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)
    assert calls(world, *SENDS)[sent:] == []
    return refused.value.failures


@MODES
def test_a_save_the_engine_refuses_is_refused_naming_why(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # The save names a layout of the kernel's state this version does not read.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    change_the_save(world, lambda state: state["live_state"].update(layout=999))

    assert refused_on_tuesday(world) == (
        f"CncKept: the engine refuses its last save of its session of {DAY1} (the saved live "
        "state has layout 999, and this kernel reads layout 6: restore it with the version "
        "of openalgo that saved it), so what it holds cannot be rebuilt: "
        f"{CHECK_BY_HAND}{OWN_UNITS}",
    )


@MODES
def test_a_save_that_names_no_instruments_is_refused(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # An earlier version of openalgo saved no kernel instruments.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    change_the_save(world, lambda state: state.pop("kernel_instruments"))

    assert refused_on_tuesday(world) == (
        f"CncKept: its save of {DAY1} names no instruments (an earlier version of openalgo "
        f"wrote it), so what it holds cannot be rebuilt: {CHECK_BY_HAND}{OWN_UNITS}",
    )


@MODES
def test_a_save_of_another_instrument_than_the_strategy_trades_is_refused(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # The save's kernel session traded INFY: the strategy, on SBIN now, cannot rebuild it.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    change_the_save(world, lambda state: state["kernel_instruments"][0].update(symbol="INFY"))

    assert refused_on_tuesday(world) == (
        f"CncKept: its save of {DAY1} is of NSE:INFY, and it trades NSE:SBIN now, so what "
        f"it holds cannot be rebuilt: {CHECK_BY_HAND}{OWN_UNITS}",
    )


@MODES
@BOOKS
def test_an_update_after_the_save_that_cannot_be_accounted_for_is_refused(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    # The journal shows the stop GTT being modified after the save, which the save, made
    # before each batch of commands, never asked for.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer, history)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    with journal_of(world) as journal:
        journal.set_gtt_state(1, GttState.MODIFYING)

    assert refused_on_tuesday(world) == (
        f"CncKept: its session of {DAY1} stopped without its day-close record, and what the "
        "journal holds after its last save cannot be accounted for: its stop GTT "
        f"{trigger_id}: the engine's was Active and the journal's says modifying: "
        f"{CHECK_BY_HAND}{OWN_UNITS}",
    )


@MODES
@BOOKS
def test_a_gtt_neither_the_save_nor_its_updates_bind_refuses_the_rebuilt_record(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    # The journal holds a second GTT of CncKept of Monday, a GTT order resting at the
    # broker, that the save does not know: the rebuilt record binds only the stop GTT, so a
    # start would trade beside a GTT no engine follows.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer, history)
    placed = world.fake.client().placegttorder(
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        product="CNC",
        quantity=5,
        price=490.0,
        triggerprice_sl=490.0,
    )
    with journal_of(world) as journal:
        journal.start_session(DAY1, world.fake.mode, BROKER)
        terms = GttTerms("BUY", 490.0, 490.0, 5)
        seq = journal.record_gtt(
            "CncKept", 7, 7, "SBIN", "NSE", "CNC", terms, purpose=Purpose.ENTRY
        )
        journal.bind_gtt(seq, placed["trigger_id"])

    assert refused_on_tuesday(world) == (
        f"CncKept: its GTT order {placed['trigger_id']} of {DAY1} on NSE:SBIN (CNC) has not "
        f"ended and no day-close record carries it: {CHECK_BY_HAND}{OWN_UNITS}",
    )


def test_the_refused_start_of_a_crashed_session_starts_once_the_cause_is_gone(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The refusal changes nothing: once the save is readable again, the next start
    # continues the leg from it, its stop GTT taken back.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, False)
    layouts = []

    def unreadable(state):
        layouts.append(state["live_state"]["layout"])
        state["live_state"]["layout"] = 999

    change_the_save(world, unreadable)
    refused_on_tuesday(world)
    change_the_save(world, lambda state: state["live_state"].update(layout=layouts[0]))
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    (position,) = outcome.open_positions
    assert (position.units, position.stop_confirmed) == (10, True)
    assert len(calls(world, "placegttorder")) == 1
