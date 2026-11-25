"""The GTT book's rows as the live layer reads them (`_live/_gtt_book.py`; 20-positional-live
3.1, 7.1): a row whose terms cannot be read is unreadable on its own, so the other rows of
the read stand; a row that names no GTT makes the whole read unreadable, since it could be
any GTT. Rows are written as OpenAlgo's GTT book answers them (a SINGLE GTT: one trigger,
one leg); expected values are the ones written.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._gtt_book import Seen, gtt_rows

#: The stop of the tests as a broker's row: sell 10 SBIN CNC at a trigger of 790.00.
STOP_ROW = {
    "trigger_id": "261116000001",
    "status": "active",
    "symbol": "SBIN",
    "exchange": "NSE",
    "trigger_prices": [790.0],
    "legs": [{"action": "SELL", "quantity": 10, "price": 780.0, "product": "CNC"}],
}


def book(*rows):
    return {"status": "success", "data": list(rows)}


@pytest.mark.parametrize(
    "change",
    [
        {"legs": []},
        {"legs": [{"action": "SELL", "quantity": None, "price": 780.0, "product": "CNC"}]},
        {"trigger_prices": []},
    ],
    ids=["no leg", "a leg with no units", "no trigger"],
)
def test_a_row_whose_terms_cannot_be_read_is_unreadable_and_the_others_stand(change):
    other = STOP_ROW | {"trigger_id": "261116000002"}

    rows = gtt_rows(book(STOP_ROW | change, other))

    assert rows is not None
    assert (rows["261116000001"].seen, rows["261116000002"].seen) == (
        Seen.UNREADABLE,
        Seen.RESTING,
    )


def test_a_row_that_names_no_gtt_makes_the_read_unreadable():
    assert gtt_rows(book(STOP_ROW, {"status": "active", "legs": []})) is None


def test_an_unreadable_row_shows_no_terms_and_has_no_shape():
    (row,) = gtt_rows(book(STOP_ROW | {"legs": []})).values()

    assert (row.seen.rests, row.quantity, row.triggers) == (False, 0, ())
