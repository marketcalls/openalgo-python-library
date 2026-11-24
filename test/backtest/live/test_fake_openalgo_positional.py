"""The fake OpenAlgo's holdings and GTT book (20-positional-live 5 and 7.1; the lead's
rulings N2-1 and N2-3): `holdings` answers the account's delivery holdings in the shape
the server writes them, and `gttorderbook` answers as a broker with GTT orders, as one
without them (HTTP 501), and as the sandbox, which answers every GTT call before it looks
at the broker. The expected bodies are written out from the server's code
(services/holdings_service.py and the reference broker's transform_holdings_data,
sandbox/holdings_manager.py; services/gtt_orderbook_service.py and restx_api/gtt_orderbook.py,
sandbox/gtt_manager.py), as the SDK returns them. Nothing leaves the process.
"""

from __future__ import annotations

from fake_openalgo import BROKER
from fake_openalgo.answers import BOOK_PATHS, broker_http_error
from fake_openalgo.fixtures import *  # noqa: F403

from openalgo.backtest._live._types import Mode

NOT_SUPPORTED = f"GTT orders are not supported for broker '{BROKER}' yet"


# ----- holdings ------------------------------------------------------------------------


def test_live_holdings_are_the_brokers_cnc_rows_and_their_totals(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 820.0)
    fake_openalgo.set_holding("SBIN", "NSE", 40, average_price=800.0)

    answer = fake_openalgo.client().holdings()

    # pnl (820 - 800) * 40 = 800.00; pnlpercent 20 / 800 * 100 = 2.5.
    row = {
        "symbol": "SBIN",
        "exchange": "NSE",
        "quantity": 40,
        "product": "CNC",
        "average_price": 800.0,
        "ltp": 820.0,
        "pnl": 800.0,
        "pnlpercent": 2.5,
    }
    statistics = {
        "totalholdingvalue": 32800.0,
        "totalinvvalue": 32000.0,
        "totalprofitandloss": 800.0,
        "totalpnlpercentage": 2.5,
    }
    assert answer == {"status": "success", "data": {"holdings": [row], "statistics": statistics}}


def test_an_account_holding_nothing_answers_an_empty_list(fake_openalgo):
    data = fake_openalgo.client().holdings()["data"]

    assert data["holdings"] == []


def test_the_sandbox_answers_its_own_holdings_marked_analyze(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.set_holding("SBIN", "NSE", 10, average_price=800.0)
    fake.set_holding("SBIN", "NSE", 99, average_price=800.0, mode=Mode.LIVE)

    answer = fake.client().holdings()

    assert answer["mode"] == "analyze"
    assert [(row["symbol"], row["quantity"]) for row in answer["data"]["holdings"]] == [
        ("SBIN", 10)
    ]


def test_a_holding_set_to_zero_units_is_no_longer_listed(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.set_holding("SBIN", "NSE", 40, average_price=800.0)
    fake_openalgo.set_holding("SBIN", "NSE", 0)

    assert fake_openalgo.client().holdings()["data"]["holdings"] == []


def test_live_holdings_while_the_login_has_expired_are_the_servers_500(fake_openalgo):
    fake_openalgo.faults.expire_session()

    answer = fake_openalgo.client().holdings()

    assert (answer["status"], answer["code"]) == ("error", 500)
    assert broker_http_error(403, BOOK_PATHS["holdings"]) in answer["message"]


# ----- the GTT book ----------------------------------------------------------------------


def test_the_gtt_book_of_a_broker_with_gtt_orders_is_a_success(fake_openalgo):
    answer = fake_openalgo.client().gttorderbook()

    assert answer == {"status": "success", "data": []}


def test_the_gtt_book_of_a_broker_without_gtt_orders_is_the_servers_501(make_fake_openalgo):
    fake = make_fake_openalgo(gtt=False)

    answer = fake.client().gttorderbook()

    assert answer == {"status": "error", "message": NOT_SUPPORTED, "code": 501}


def test_the_sandbox_answers_the_gtt_book_for_a_broker_without_gtt_orders(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True, gtt=False)

    answer = fake.client().gttorderbook(status="all")

    assert answer == {"status": "success", "mode": "analyze", "data": []}


def test_a_gtt_book_status_other_than_active_or_all_is_the_servers_400(fake_openalgo):
    answer = fake_openalgo.client().gttorderbook(status="fired")

    refused = {"status": ["Must be one of: active, all."]}
    assert (answer["code"], answer["message"]) == (400, refused)
