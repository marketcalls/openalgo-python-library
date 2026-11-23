"""`api.charges` (openalgo/account.py): the request it sends to OpenAlgo's Charges API and
the answers it returns. The client's transport answers in-process, so nothing reaches a
server.

The server's answers are those of the API page (api-documentation/v1/accounts-api/
charges.md) and of restx_api/charges.py with services/charges_service.py.
"""

from __future__ import annotations

import httpx
import pytest
from rest_fakes import (
    API_KEY,
    TIMED_OUT,
    UNREACHABLE,
    answering,
    client_answering,
    failing,
    recording,
)

# The API page's sample: a delivery buy of 50 RELIANCE at 1350 on NSE.
RELIANCE_BUY = {
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "action": "BUY",
    "product": "CNC",
    "quantity": 50,
    "price": 1350,
}

# The API page's sample answer to RELIANCE_BUY.
RELIANCE_BUY_PRICED = {
    "status": "success",
    "data": {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "product": "CNC",
        "action": "BUY",
        "quantity": 50,
        "price": 1350.0,
        "segment": "Equity Delivery",
        "trade_value": 67500.0,
        "charges": {
            "total": 104.13,
            "brokerage": 20.0,
            "taxes": {"gst": 3.99, "stt": 68.0, "stamp_duty": 10.0},
            "other_charges": {
                "transaction": 2.0,
                "clearing": 0.0,
                "ipft": 0.07,
                "sebi_turnover": 0.07,
            },
        },
    },
}


def test_charges_posts_the_order_and_the_api_key_to_the_charges_endpoint():
    seen = []

    with client_answering(recording(seen, RELIANCE_BUY_PRICED)) as client:
        client.charges(
            symbol="RELIANCE", exchange="NSE", action="BUY", product="CNC", quantity=50, price=1350
        )

    assert seen == [
        (
            "POST",
            "http://openalgo.invalid/api/v1/charges",
            {
                "apikey": API_KEY,
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "action": "BUY",
                "product": "CNC",
                "quantity": 50,
                "price": 1350,
            },
        )
    ]


def test_charges_sends_an_option_lot_in_units_and_its_premium_as_given():
    # One NIFTY lot of 65 is sent as 65, and an option is priced on its premium.
    seen = []

    with client_answering(recording(seen, {"status": "success", "data": {}})) as client:
        client.charges(
            symbol="NIFTY29DEC2626000CE",
            exchange="NFO",
            action="SELL",
            product="NRML",
            quantity=65,
            price=135.5,
        )

    sent = seen[0][2]
    assert (sent["quantity"], sent["price"]) == (65, 135.5)


def test_charges_returns_the_servers_priced_order_as_it_is():
    with client_answering(answering(200, RELIANCE_BUY_PRICED)) as client:
        answer = client.charges(**RELIANCE_BUY)

    assert answer == RELIANCE_BUY_PRICED


@pytest.mark.parametrize(
    "message",
    [
        "{'price': ['Missing data for required field.']}",
        "{'price': ['Price must be a positive number.']}",
        "Symbol NOSUCHSYMBOL not found on NSE",
        "Orders cannot be placed on NSE_INDEX, so no charges apply to it",
    ],
)
def test_a_refused_order_comes_back_with_the_servers_message_and_code_400(message):
    refused = {"status": "error", "message": message}

    with client_answering(answering(400, refused)) as client:
        answer = client.charges(**RELIANCE_BUY)

    assert answer == {"status": "error", "message": message, "code": 400}


def test_an_invalid_api_key_comes_back_with_the_servers_message_and_code_403():
    refused = {"status": "error", "message": "Invalid openalgo apikey"}

    with client_answering(answering(403, refused)) as client:
        answer = client.charges(**RELIANCE_BUY)

    assert answer == {"status": "error", "message": "Invalid openalgo apikey", "code": 403}


def test_a_server_that_does_not_answer_in_time_gives_the_sdks_timeout_error():
    with client_answering(failing(httpx.ReadTimeout("no answer"))) as client:
        answer = client.charges(**RELIANCE_BUY)

    assert answer == TIMED_OUT


def test_a_server_that_cannot_be_reached_gives_the_sdks_connection_error():
    with client_answering(failing(httpx.ConnectError("refused"))) as client:
        answer = client.charges(**RELIANCE_BUY)

    assert answer == UNREACHABLE


def test_charges_takes_the_order_by_keyword_only():
    with (
        client_answering(answering(200, RELIANCE_BUY_PRICED)) as client,
        pytest.raises(TypeError),
    ):
        client.charges("RELIANCE", "NSE", "BUY", "CNC", 50, 1350)
