"""The order worker's holdings read (20-positional-live 7.1; the lead's ruling N2-1): a
session with a CNC strategy (`SessionSpec.reads_holdings`) starts its order worker reading
`holdings()` with every book read, under the same pacing as the other books, after the
position book and before the trade book, and posted with them; a session without one reads
none, so an MIS session's reads are as they were.

The worker runs with the fake's real SDK clients (`worker_samples`), and the session on the
fake (`session_samples`); the fake's call log shows what was read and in which order.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import ist
from session_samples import BuyAndHold, rig
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._worker_queue import ReadBooks, ReadPositions
from openalgo.backtest._live.inbox import Books, PositionRows
from openalgo.backtest._live.plan import SessionSpec

#: The account's delivery holding of SBIN the tests set.
HELD_UNITS = 40
#: What a reconciliation reads, in order, when the worker reads holdings: the holdings
#: after the position book and before the trade book, so a fill that lands during the read
#: shows in the trade book (16-session-design 5.4, round 9).
CNC_BOOKS = ["positionbook", "holdings", "orderbook", "tradebook", "funds"]


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    fake_openalgo.set_holding("SBIN", "NSE", HELD_UNITS, average_price=800.0)
    return fake_openalgo


def rig_of(broker, folder, *, read_holdings):
    worker_rig = WorkerRig(broker, folder, read_holdings=read_holdings)
    worker_rig.messages()
    return worker_rig


def book_calls(worker_rig) -> list[str]:
    books = {"positionbook", "holdings", "orderbook", "tradebook", "funds"}
    return [entry.what for entry in worker_rig.fake.log.calls() if entry.what in books]


def held(answer) -> list[tuple[str, int]]:
    return [(row["symbol"], row["quantity"]) for row in answer["data"]["holdings"]]


# ----- which sessions read holdings ------------------------------------------------------------


@pytest.mark.parametrize(
    ("products", "reads"),
    [
        (("CNC",), True),
        (("MIS", "CNC"), True),
        (("MIS",), False),
        (("NRML",), False),
        (("MIS", "NRML"), False),
    ],
    ids=["cnc", "mis and cnc", "mis", "nrml", "mis and nrml"],
)
def test_a_session_reads_holdings_when_one_of_its_strategies_trades_cnc(products, reads):
    strategies = tuple(SimpleNamespace(product=product) for product in products)

    spec = SessionSpec(SimpleNamespace(), dt.date(2026, 11, 16), strategies)

    assert spec.reads_holdings is reads


@pytest.mark.parametrize("reads", [True, False], ids=["reads holdings", "reads none"])
def test_a_sessions_order_worker_reads_holdings_as_its_spec_says(
    make_fake_openalgo, monkeypatch, tmp_path, reads
):
    # Until unit N3 no CNC strategy runs live, so an MIS session's spec is made to say it.
    monkeypatch.setattr(SessionSpec, "reads_holdings", property(lambda spec: reads))
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 41, 30))

    world.run(BuyAndHold)

    assert bool(world.fake.log.calls("holdings")) is reads


# ----- the worker's reads ----------------------------------------------------------------------


def test_a_worker_reading_holdings_reads_them_before_the_trade_book(broker, tmp_path):
    worker_rig = rig_of(broker, tmp_path, read_holdings=True)

    (books,) = worker_rig.of(Books, worker_rig.run(ReadBooks(3)))

    assert book_calls(worker_rig) == CNC_BOOKS
    assert held(books.holdings) == [("SBIN", HELD_UNITS)]
    worker_rig.worker.close()


def test_a_worker_reading_holdings_reads_them_with_a_position_read(broker, tmp_path):
    worker_rig = rig_of(broker, tmp_path, read_holdings=True)

    (rows,) = worker_rig.of(PositionRows, worker_rig.run(ReadPositions(orders=True)))

    assert book_calls(worker_rig) == ["positionbook", "holdings", "orderbook"]
    assert held(rows.holdings) == [("SBIN", HELD_UNITS)]
    worker_rig.worker.close()


@pytest.mark.parametrize(
    ("request_", "message", "calls"),
    [
        (ReadBooks(3), Books, ["positionbook", "orderbook", "tradebook", "funds"]),
        (ReadPositions(), PositionRows, ["positionbook"]),
    ],
    ids=["books", "position read"],
)
def test_a_worker_reading_no_holdings_reads_the_books_it_read_before(
    broker, tmp_path, request_, message, calls
):
    worker_rig = rig_of(broker, tmp_path, read_holdings=False)

    (posted,) = worker_rig.of(message, worker_rig.run(request_))

    assert (book_calls(worker_rig), posted.holdings) == (calls, None)
    worker_rig.worker.close()


def test_a_failed_holdings_read_is_posted_as_the_answer_it_was(broker, tmp_path):
    worker_rig = rig_of(broker, tmp_path, read_holdings=True)
    broker.faults.gateway_error("holdings", status=503)

    (books,) = worker_rig.of(Books, worker_rig.run(ReadBooks(3)))

    assert (books.holdings["status"], books.positionbook["status"]) == ("error", "success")
    worker_rig.worker.close()
