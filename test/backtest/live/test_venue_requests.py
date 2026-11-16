"""The order requests built from the kernel's command dicts (`_venue_requests`,
16-session-design 2.7 and 2.4): each "place" key becomes an intent field, an intent ID is
unique across modes and restarts, each request has its priority, and a targeted read of
an order gives its verdict on a modify or cancel whose answer was lost or refused.
"""

from __future__ import annotations

import pytest
from venue_samples import DAY, OPEN_NS, STRATEGY, journal_session

from openalgo.backtest._live._order_check import OrderCheck
from openalgo.backtest._live._types import Mode, OrderIntent, OrderStatus, Purpose
from openalgo.backtest._live._venue_requests import (
    ModifyRequest,
    OrderTerms,
    Priority,
    Verdict,
    cancel_priority,
    cancel_verdict,
    intent_fields,
    intent_id,
    modify_priority,
    modify_verdict,
    place_priority,
    terms_of,
)
from openalgo.backtest._live.journal import Journal
from openalgo.backtest.errors import JournalError, LiveError

MARKETS = (("SBIN", "NSE"), ("RELIANCE", "NSE"))


def place(**changes):
    """A "place" command dict as the binding hands it (venue.rs)."""
    command = {
        "command": "place",
        "order_id": 7,
        "instrument": 0,
        "product": "MIS",
        "side": "BUY",
        "role": "entry",
        "reason": None,
        "quantity_units": 100.0,
        "order_type": "MARKET",
        "price": None,
        "trigger_price": None,
    }
    return command | changes


def intent_of(command, mode=Mode.LIVE, starts=1):
    return intent_fields(
        command,
        strategy=STRATEGY,
        markets=MARKETS,
        session=journal_session(mode, starts),
        created_ns=OPEN_NS,
    )


@pytest.mark.parametrize(
    ("changes", "fields"),
    [
        ({}, {"action": "BUY", "price_type": "MARKET", "price": 0.0, "trigger_price": 0.0}),
        ({"side": "SELL"}, {"action": "SELL"}),
        ({"order_type": "LIMIT", "price": 499.5}, {"price_type": "LIMIT", "price": 499.5}),
        (
            {"order_type": "SL", "price": 114.25, "trigger_price": 120.3},
            {"price_type": "SL", "price": 114.25, "trigger_price": 120.3},
        ),
        (
            {"order_type": "SL-M", "trigger_price": 490.0},
            {"price_type": "SL-M", "price": 0.0, "trigger_price": 490.0},
        ),
        ({"instrument": 1}, {"symbol": "RELIANCE", "exchange": "NSE"}),
        ({"quantity_units": 25.0}, {"quantity": 25}),
        ({"role": "add"}, {"purpose": Purpose.ADD}),
        ({"role": "leg_stop"}, {"purpose": Purpose.STOP}),
        ({"role": "exit"}, {"purpose": Purpose.EXIT}),
        ({"role": "risk_exit", "reason": "daily_loss"}, {"purpose": Purpose.RISK_EXIT}),
        ({"role": "square_off"}, {"purpose": Purpose.SQUARE_OFF}),
    ],
    ids=[
        "market buy",
        "side",
        "limit",
        "stop-limit",
        "stop-market",
        "instrument",
        "units",
        "add",
        "leg stop",
        "exit",
        "risk exit",
        "square-off",
    ],
)
def test_each_place_key_becomes_its_intent_field(changes, fields):
    intent = intent_of(place(**changes))

    expected = {"strategy": STRATEGY, "symbol": "SBIN", "exchange": "NSE", "quantity": 100}
    expected |= {"product": "MIS", "purpose": Purpose.ENTRY, "created_ns": OPEN_NS} | fields
    assert {name: getattr(intent, name) for name in expected} == expected


def test_an_intent_id_names_the_day_the_mode_the_start_the_strategy_and_the_order():
    assert intent_of(place()).intent_id == "20250701-live-1-ema_cross-o7"


def test_intent_ids_of_one_kernel_order_differ_across_modes_and_restarts():
    ids = {
        intent_id(journal_session(mode, starts), STRATEGY, 1)
        for mode in (Mode.SANDBOX, Mode.LIVE)
        for starts in (1, 2)
    }
    assert len(ids) == 4


def test_the_journal_takes_a_sandbox_rehearsal_and_a_live_session_of_one_day(tmp_path):
    def record(mode):
        with Journal.open(
            tmp_path / "journal.sqlite", account="a", broker="zerodha", mode=mode, now=lambda: 1
        ) as journal:
            journal.start_session(DAY, mode, "zerodha")
            journal.record_intent(intent_of(place(), mode=mode))

    record(Mode.SANDBOX)
    record(Mode.LIVE)
    with pytest.raises(JournalError, match="already in the journal"):
        record(Mode.LIVE)


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"quantity_units": 10.5}, "whole units"),
        ({"quantity_units": 0.0}, "whole units"),
        ({"role": "hedge"}, "role 'hedge'"),
        ({"side": "SHORT"}, "side 'SHORT'"),
        ({"product": "CNC"}, "product 'CNC'"),
        ({"instrument": 2}, "unregistered"),
    ],
    ids=["fraction", "no units", "role", "side", "product", "instrument"],
)
def test_a_command_the_session_cannot_send_is_refused(changes, words):
    with pytest.raises(LiveError, match=words):
        intent_of(place(**changes))


def an_intent(role):
    return intent_of(place(role=role))


@pytest.mark.parametrize(
    ("role", "priority"),
    [
        ("entry", Priority.ENTER),
        ("add", Priority.ENTER),
        ("leg_stop", Priority.PROTECT),
        ("exit", Priority.PROTECT),
        ("risk_exit", Priority.PROTECT),
        ("square_off", Priority.PROTECT),
    ],
)
def test_a_placement_is_protective_unless_it_enters(role, priority):
    assert place_priority(an_intent(role)) is priority


STOP_TERMS = OrderTerms(100, "SL-M", 0.0, 490.0)


@pytest.mark.parametrize(
    ("role", "new", "priority"),
    [
        ("leg_stop", OrderTerms(100, "SL-M", 0.0, 492.0), Priority.ADJUST),
        ("leg_stop", OrderTerms(160, "SL-M", 0.0, 490.0), Priority.PROTECT),
        ("leg_stop", OrderTerms(60, "SL-M", 0.0, 492.0), Priority.PROTECT),
        ("leg_stop", OrderTerms(100, "LIMIT", 465.5, 0.0), Priority.PROTECT),
        ("risk_exit", OrderTerms(100, "LIMIT", 465.0, 0.0), Priority.PROTECT),
        ("entry", OrderTerms(100, "LIMIT", 499.0, 0.0), Priority.ADJUST),
    ],
    ids=["trailing", "units after an add", "units for a partial exit", "reprice", "exit", "entry"],
)
def test_a_modify_is_protective_unless_it_trails_a_trigger_or_changes_an_entry(role, new, priority):
    assert modify_priority(an_intent(role), STOP_TERMS, new) is priority


@pytest.mark.parametrize(
    ("role", "priority"),
    [
        ("entry", Priority.ADJUST),
        ("add", Priority.ADJUST),
        ("leg_stop", Priority.PROTECT),
        ("exit", Priority.PROTECT),
    ],
)
def test_the_cancel_of_a_closing_order_is_protective_and_of_an_entry_is_not(role, priority):
    assert cancel_priority(an_intent(role)) is priority


def test_modify_terms_carry_the_orders_total_units():
    command = {"command": "modify", "order_id": 7, "quantity_units": 160.0}
    command |= {"order_type": "SL-M", "price": None, "trigger_price": 490.0}
    assert terms_of(command) == OrderTerms(160, "SL-M", 0.0, 490.0)


def a_modify(new: OrderTerms, old: OrderTerms = STOP_TERMS) -> ModifyRequest:
    intent: OrderIntent = an_intent("leg_stop")
    return ModifyRequest(1, 7, intent, "B9", new, old, Priority.ADJUST)


def a_check(status=OrderStatus.TRIGGER_PENDING, **row):
    values = {"quantity": 100, "price": 0.0, "trigger_price": 490.0, "price_type": "SL-M"}
    return OrderCheck("B9", status=status, **(values | row))


@pytest.mark.parametrize(
    ("check", "verdict"),
    [
        (a_check(trigger_price=492.0), Verdict.CONFIRMED),
        (a_check(), Verdict.REFUSED),
        (a_check(OrderStatus.COMPLETE), Verdict.ENDED),
        (a_check(OrderStatus.CANCELLED, trigger_price=492.0), Verdict.ENDED),
        (a_check(trigger_price=491.0), Verdict.UNSETTLED),
        (OrderCheck("B9", reason="orderstatus failed: timeout"), Verdict.UNSETTLED),
    ],
    ids=["new terms", "old terms", "complete", "cancelled", "neither", "unread"],
)
def test_a_read_row_gives_its_verdict_on_a_modify(check, verdict):
    assert modify_verdict(a_modify(OrderTerms(100, "SL-M", 0.0, 492.0)), check) is verdict


def test_a_limit_price_counts_in_a_reprice_and_float_noise_does_not():
    reprice = a_modify(OrderTerms(100, "LIMIT", 465.5, 0.0))
    row = {"price_type": "LIMIT", "trigger_price": 0.0}
    assert modify_verdict(reprice, a_check(price=465.5000000001, **row)) is Verdict.CONFIRMED
    assert modify_verdict(reprice, a_check(price=466.0, **row)) is Verdict.UNSETTLED


@pytest.mark.parametrize(
    ("check", "verdict"),
    [
        (a_check(OrderStatus.CANCELLED), Verdict.ENDED),
        (a_check(OrderStatus.COMPLETE), Verdict.ENDED),
        (a_check(OrderStatus.OPEN), Verdict.REFUSED),
        (a_check(), Verdict.REFUSED),
        (OrderCheck("B9", reason="orderstatus failed: timeout"), Verdict.UNSETTLED),
    ],
    ids=["cancelled", "complete", "open", "trigger pending", "unread"],
)
def test_a_read_row_gives_its_verdict_on_a_cancel(check, verdict):
    assert cancel_verdict(check) is verdict
