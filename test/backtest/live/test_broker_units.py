"""The one reader of the broker's units of a leg, by product (`_live/_broker_units.py`;
20-positional-live 7.1, the lead's ruling N2-1).

MIS and NRML units are the position book's line of the symbol, exchange and product; CNC
units are the holdings row's quantity plus the CNC line of the position book. A read that
failed (or was not made) is unknown, never zero; a read that succeeded with no row of the
symbol is zero. Expected values are worked by hand from the rows of each case.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._broker_units import CNC, MIS, NRML, broker_units
from openalgo.backtest.errors import LiveError


def positions(*rows: tuple[str, str, str, object]) -> dict:
    """A position-book answer as the SDK returns it: (symbol, exchange, product, quantity)."""
    data = [
        {"symbol": symbol, "exchange": exchange, "product": product, "quantity": quantity}
        for symbol, exchange, product, quantity in rows
    ]
    return {"status": "success", "data": data}


def holdings(*rows: tuple[str, str, object]) -> dict:
    """A holdings answer as the SDK returns it (services/holdings_service.py): rows of
    (symbol, exchange, quantity), each CNC, under `data.holdings`."""
    data = [
        {"symbol": symbol, "exchange": exchange, "product": "CNC", "quantity": quantity}
        for symbol, exchange, quantity in rows
    ]
    statistics = {"totalholdingvalue": 0.0, "totalinvvalue": 0.0}
    return {"status": "success", "data": {"holdings": data, "statistics": statistics}}


#: The SDK's answer to a book read the broker failed (an expired login: HTTP 500).
FAILED = {
    "status": "error",
    "message": 'HTTP 500: {"message":"broker refused","status":"error"}',
    "code": 500,
    "error_type": "http_error",
}
#: A day's position book of the account: SBIN in three products, INFY in MIS.
BOOK = positions(
    ("SBIN", "NSE", MIS, 25),
    ("SBIN", "NSE", NRML, -50),
    ("SBIN", "NSE", CNC, 7),
    ("INFY", "NSE", MIS, 3),
)
EMPTY_BOOK = positions()
NO_HOLDINGS = holdings()


@pytest.mark.parametrize(
    ("product", "book", "held", "expected"),
    [
        (MIS, BOOK, None, 25),
        (NRML, BOOK, None, -50),
        (MIS, EMPTY_BOOK, None, 0),
        (MIS, BOOK, FAILED, 25),
        (MIS, FAILED, NO_HOLDINGS, None),
        (NRML, FAILED, None, None),
        # A buy on the day of the entry: the CNC line only.
        (CNC, BOOK, NO_HOLDINGS, 7),
        # A later day: the holdings only (the position book shows no CNC line).
        (CNC, positions(("SBIN", "NSE", MIS, 25)), holdings(("SBIN", "NSE", 40)), 40),
        # Both while a buy settles: 40 held and 7 bought today.
        (CNC, BOOK, holdings(("SBIN", "NSE", 40)), 47),
        # A part sold from the holdings today: 40 held, the CNC line shows -15.
        (CNC, positions(("SBIN", "NSE", CNC, -15)), holdings(("SBIN", "NSE", 40)), 25),
        # Two holdings rows of the symbol add up; another exchange's row is not its.
        (
            CNC,
            EMPTY_BOOK,
            holdings(("SBIN", "NSE", 10), ("SBIN", "NSE", 5), ("SBIN", "BSE", 9)),
            15,
        ),
        (CNC, EMPTY_BOOK, NO_HOLDINGS, 0),
        (CNC, BOOK, FAILED, None),
        (CNC, BOOK, None, None),
        (CNC, FAILED, holdings(("SBIN", "NSE", 40)), None),
    ],
    ids=[
        "mis line",
        "nrml line",
        "mis no row",
        "mis needs no holdings",
        "mis failed position read",
        "nrml failed position read",
        "cnc position only",
        "cnc holdings only",
        "cnc both",
        "cnc part sold",
        "cnc rows of the symbol",
        "cnc no row",
        "cnc failed holdings read",
        "cnc holdings not read",
        "cnc failed position read",
    ],
)
def test_the_brokers_units_of_a_leg_by_its_product(product, book, held, expected):
    assert broker_units(book, held, "SBIN", "NSE", product) == expected


@pytest.mark.parametrize(
    ("book", "held"),
    [
        (positions(("SBIN", "NSE", MIS, "ten")), None),
        ({"status": "success", "data": ["SBIN"]}, None),
        ({"status": "success", "data": {"positions": []}}, None),
        (EMPTY_BOOK, holdings(("SBIN", "NSE", "forty"))),
        (EMPTY_BOOK, {"status": "success", "data": {"statistics": {}}}),
        (EMPTY_BOOK, {"status": "success", "data": []}),
    ],
    ids=[
        "position quantity not a number",
        "position row not a row",
        "position book not a list",
        "holding quantity not a number",
        "holdings missing",
        "holdings data not a dict",
    ],
)
def test_a_book_that_cannot_be_read_is_unknown_never_zero(book, held):
    product = MIS if held is None else CNC
    assert broker_units(book, held, "SBIN", "NSE", product) is None


def test_a_product_the_live_layer_does_not_know_is_refused():
    with pytest.raises(LiveError, match="product 'BO'.*MIS, CNC or NRML"):
        broker_units(BOOK, None, "SBIN", "NSE", "BO")
