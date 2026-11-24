"""The rules of reconciliation (`_live/_reconcile_rules.py`; 16-session-design 5.3 and 5.5,
decided 2026-11-16): how the broker's position of a leg is judged against the engine's,
what each judgement allows, how a restart judges the journal's position, and how the
books' answers are read. Pure: no session, no server.

Positions are signed units (long above 0). Expected values come from the design's tables.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._reconcile_rules import (
    GUARDED,
    RULES,
    Kind,
    RestartKind,
    Trade,
    adoption_price,
    exit_room,
    judge,
    leg_books,
    restart_kind,
    unexplained_by,
)
from openalgo.backtest._live._types import OrderStatus

#: The table of 5.3: (the engine's units, the broker's, other orders' trades, adopted, kind).
#: The group rules of a group of one are pinned against it (`test_reconcile_groups`).
TABLE_5_3 = [
    (10, 10, 0, 0, Kind.AGREE),
    (10, 0, -10, 0, Kind.EXPLAINED_FLAT),
    (10, 4, -6, 0, Kind.EXPLAINED_FEWER),
    (-10, -4, 6, 0, Kind.EXPLAINED_FEWER),
    (10, 0, 0, 0, Kind.UNEXPLAINED),
    (10, 4, 0, 0, Kind.UNEXPLAINED),
    (10, 4, -4, 0, Kind.UNEXPLAINED),
    (4, 0, -10, -6, Kind.EXPLAINED_FLAT),
    (4, 0, -6, -6, Kind.UNEXPLAINED),
    (10, 20, 10, 0, Kind.MORE),
    (0, 10, 10, 0, Kind.MORE),
    (0, -10, 0, 0, Kind.MORE),
    (10, -10, -20, 0, Kind.OTHER_SIDE),
    (10, None, 0, 0, Kind.UNREADABLE),
]


@pytest.mark.parametrize(("engine", "broker", "external", "adopted", "kind"), TABLE_5_3)
def test_a_difference_is_judged_by_the_table_of_5_3(engine, broker, external, adopted, kind):
    unexplained = 0 if broker is None else unexplained_by(engine, broker, external, adopted)
    assert judge(engine, broker, unexplained) is kind


def test_only_an_explained_difference_is_adopted():
    adopted = {kind for kind, rule in RULES.items() if rule.adopt}
    assert adopted == {Kind.EXPLAINED_FLAT, Kind.EXPLAINED_FEWER}


def test_only_the_other_side_cancels_the_legs_stop():
    assert {kind for kind, rule in RULES.items() if rule.cancel_stop} == {Kind.OTHER_SIDE}


def test_every_kind_that_shows_fewer_units_at_the_broker_guards_the_exits():
    assert {
        Kind.EXPLAINED_FLAT,
        Kind.EXPLAINED_FEWER,
        Kind.UNEXPLAINED,
        Kind.OTHER_SIDE,
        Kind.EXPLAINED_SHARED,
    } == GUARDED


@pytest.mark.parametrize(
    ("engine", "broker", "room"),
    [(10, 4, 4), (10, 10, 10), (10, 15, 10), (10, 0, 0), (10, -5, 0), (-10, -4, 4)],
)
def test_a_closing_order_may_be_for_what_the_broker_holds_on_the_engines_side(engine, broker, room):
    assert exit_room(engine, broker) == room


#: The table of 5.5: (the journal's units, the broker's, unexplained, what the restart does).
TABLE_5_5 = [
    (10, 10, 0, RestartKind.RESUME),
    (0, 0, 0, RestartKind.RESUME),
    (0, 10, 10, RestartKind.UNKNOWN_POSITION),
    (10, 0, -10, RestartKind.UNEXPLAINED_FLAT),
    (10, 0, 0, RestartKind.DIFFERS),
    (10, 4, 0, RestartKind.DIFFERS),
    (10, 4, -6, RestartKind.DIFFERS),
]


@pytest.mark.parametrize(("journal", "broker", "unexplained", "kind"), TABLE_5_5)
def test_a_restart_judges_the_journal_position_by_the_table_of_5_5(
    journal, broker, unexplained, kind
):
    assert restart_kind(journal, broker, unexplained) is kind


def test_units_are_adopted_at_the_newest_trades_of_other_orders_on_their_side():
    trades = [
        Trade("A", "SELL", 5, 498.0),
        Trade("B", "BUY", 3, 501.0),
        Trade("C", "SELL", 4, 496.0),
    ]
    # The newest sale (4 at 496.00), then 2 of the older one at 498.00.
    assert adoption_price(trades, 6, "SELL") == pytest.approx((4 * 496.0 + 2 * 498.0) / 6)


def test_no_trade_on_the_side_to_adopt_at_is_refused():
    with pytest.raises(ValueError, match="no trade"):
        adoption_price([Trade("A", "BUY", 3, 501.0)], 3, "SELL")


def answers(positions, trades, orders):
    return (
        {"status": "success", "data": positions},
        {"status": "success", "data": trades},
        {"status": "success", "data": {"orders": orders}},
    )


def test_the_books_of_one_leg_are_read_for_its_instrument_in_mis_only():
    positionbook, tradebook, orderbook = answers(
        [
            {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": 10},
            {"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "quantity": 7},
            {"symbol": "INFY", "exchange": "NSE", "product": "MIS", "quantity": 3},
        ],
        [
            {
                "symbol": "SBIN",
                "exchange": "NSE",
                "product": "MIS",
                "action": "BUY",
                "quantity": 10,
                "average_price": 500.0,
                "orderid": "1",
            },
            {
                "symbol": "INFY",
                "exchange": "NSE",
                "product": "MIS",
                "action": "BUY",
                "quantity": 3,
                "average_price": 1500.0,
                "orderid": "2",
            },
        ],
        [
            {
                "symbol": "SBIN",
                "exchange": "NSE",
                "product": "MIS",
                "orderid": "3",
                "order_status": "trigger pending",
                "quantity": 10,
                "pricetype": "SL-M",
                "trigger_price": 497.0,
            }
        ],
    )
    books = leg_books(positionbook, tradebook, orderbook, "SBIN", "NSE", product="MIS")
    assert books.units == 10
    assert books.trades == (Trade("1", "BUY", 10, 500.0),)
    [row] = books.rows
    assert (row.order_id, row.status, row.quantity, row.trigger_price) == (
        "3",
        OrderStatus.TRIGGER_PENDING,
        10,
        497.0,
    )
    assert books.fills("1") == (10, 500.0)
    assert books.external(frozenset({"1"})) == ()


def test_a_book_with_a_row_that_cannot_be_read_is_unreadable_whole():
    positionbook, tradebook, orderbook = answers(
        [{"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": "ten"}],
        [
            {
                "symbol": "SBIN",
                "exchange": "NSE",
                "product": "MIS",
                "action": "BUY",
                "quantity": 10.5,
                "average_price": 500.0,
                "orderid": "1",
            }
        ],
        [{"symbol": "SBIN", "exchange": "NSE", "orderid": "3", "order_status": "parked"}],
    )
    books = leg_books(positionbook, tradebook, orderbook, "SBIN", "NSE", product="MIS")
    assert (books.units, books.trades, books.rows) == (None, None, None)


def test_an_error_answer_reads_as_no_book():
    error = {"status": "error", "message": "broker down"}
    books = leg_books(error, error, error, "SBIN", "NSE", product="MIS")
    assert (books.units, books.trades, books.rows) == (None, None, None)


def a_row(product, orderid, **fields):
    """A trade-book or order-book row of SBIN on NSE in `product`."""
    return {"symbol": "SBIN", "exchange": "NSE", "product": product, "orderid": orderid} | fields


#: Books of an account that trades SBIN in MIS and CNC: 10 MIS units, 7 CNC bought today
#: and 40 held from earlier days.
MIS_AND_CNC = answers(
    [
        {"symbol": "SBIN", "exchange": "NSE", "product": "MIS", "quantity": 10},
        {"symbol": "SBIN", "exchange": "NSE", "product": "CNC", "quantity": 7},
    ],
    [
        a_row("MIS", "1", action="BUY", quantity=10, average_price=500.0),
        a_row("CNC", "4", action="BUY", quantity=7, average_price=501.0),
    ],
    [
        a_row("MIS", "3", order_status="trigger pending", quantity=10, pricetype="SL-M"),
        a_row("CNC", "5", order_status="open", quantity=7, pricetype="LIMIT"),
    ],
)
HELD_40 = {
    "status": "success",
    "data": {"holdings": [{"symbol": "SBIN", "exchange": "NSE", "quantity": 40}]},
}


@pytest.mark.parametrize(
    ("product", "units", "trade", "row"),
    [
        ("MIS", 10, Trade("1", "BUY", 10, 500.0), "3"),
        ("CNC", 47, Trade("4", "BUY", 7, 501.0), "5"),
    ],
    ids=["mis", "cnc"],
)
def test_the_books_of_a_leg_are_read_for_its_product(product, units, trade, row):
    positionbook, tradebook, orderbook = MIS_AND_CNC

    books = leg_books(
        positionbook, tradebook, orderbook, "SBIN", "NSE", product=product, holdings=HELD_40
    )

    assert (books.units, books.trades, [r.order_id for r in books.rows]) == (units, (trade,), [row])


def test_a_cnc_leg_whose_holdings_were_not_read_has_no_units_known():
    books = leg_books(*MIS_AND_CNC, "SBIN", "NSE", product="CNC")

    assert (books.units, books.trades) == (None, (Trade("4", "BUY", 7, 501.0),))
