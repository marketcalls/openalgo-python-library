"""The order worker's holdings read (20-positional-live 7.1; the lead's ruling N2-1): a
session with a CNC strategy reads `holdings()` with every book read, under the same pacing
as the other books, after the position book and before the trade book, and posts it with
them; a session without one reads none, so an MIS session's reads are as they were.

The worker runs with the fake's real SDK clients (`worker_samples`); the fake's call log
shows what was read and in which order.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._worker_queue import ReadBooks, ReadPositions
from openalgo.backtest._live.inbox import Books, PositionRows

#: The account's delivery holding of SBIN the tests set.
HELD_UNITS = 40
#: What a reconciliation reads, in order, when the session has a CNC strategy: the
#: holdings after the position book and before the trade book, so a fill that lands
#: during the read shows in the trade book (16-session-design 5.4, round 9).
CNC_BOOKS = ["positionbook", "holdings", "orderbook", "tradebook", "funds"]


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    fake_openalgo.set_holding("SBIN", "NSE", HELD_UNITS, average_price=800.0)
    return fake_openalgo


def rig_of(broker, folder, *, read_holdings):
    rig = WorkerRig(broker, folder, read_holdings=read_holdings)
    rig.messages()
    return rig


def book_calls(rig) -> list[str]:
    books = {"positionbook", "holdings", "orderbook", "tradebook", "funds"}
    return [entry.what for entry in rig.fake.log.calls() if entry.what in books]


def held(answer) -> list[tuple[str, int]]:
    return [(row["symbol"], row["quantity"]) for row in answer["data"]["holdings"]]


def test_a_session_with_a_cnc_strategy_reads_the_holdings_before_the_trade_book(broker, tmp_path):
    rig = rig_of(broker, tmp_path, read_holdings=True)

    (books,) = rig.of(Books, rig.run(ReadBooks(3)))

    assert book_calls(rig) == CNC_BOOKS
    assert held(books.holdings) == [("SBIN", HELD_UNITS)]
    rig.worker.close()


def test_a_session_with_a_cnc_strategy_reads_the_holdings_with_a_position_read(broker, tmp_path):
    rig = rig_of(broker, tmp_path, read_holdings=True)

    (rows,) = rig.of(PositionRows, rig.run(ReadPositions(orders=True)))

    assert book_calls(rig) == ["positionbook", "holdings", "orderbook"]
    assert held(rows.holdings) == [("SBIN", HELD_UNITS)]
    rig.worker.close()


@pytest.mark.parametrize(
    ("request_", "message", "calls"),
    [
        (ReadBooks(3), Books, ["positionbook", "orderbook", "tradebook", "funds"]),
        (ReadPositions(), PositionRows, ["positionbook"]),
    ],
    ids=["books", "position read"],
)
def test_a_session_without_a_cnc_strategy_reads_no_holdings(
    broker, tmp_path, request_, message, calls
):
    rig = rig_of(broker, tmp_path, read_holdings=False)

    (posted,) = rig.of(message, rig.run(request_))

    assert (book_calls(rig), posted.holdings) == (calls, None)
    rig.worker.close()


def test_a_failed_holdings_read_is_posted_as_the_answer_it_was(broker, tmp_path):
    rig = rig_of(broker, tmp_path, read_holdings=True)
    broker.faults.gateway_error("holdings", status=503)

    (books,) = rig.of(Books, rig.run(ReadBooks(3)))

    assert (books.holdings["status"], books.positionbook["status"]) == ("error", "success")
    rig.worker.close()
