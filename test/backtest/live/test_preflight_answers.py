"""How the preflight words an error answer (`_live/_preflight_answers.py`): one wording for
every shape the SDK returns, so no check prints an HTTP status twice or drops it.

The answers are written as the SDK returns them: its older calls (openalgo/account.py)
flatten a non-200 reply into "HTTP <code>: <text>", its newer ones (`BaseAPI._post`, the GTT
calls) keep the server's JSON body and add `code`.
"""

from __future__ import annotations

import pytest
from preflight_fakes import sdk_http_error

from openalgo.backtest._live._preflight_answers import MAX_QUOTED_CHARS, answer_reason

PAGE = "<html><body>Bad Gateway</body></html>"


@pytest.mark.parametrize(
    ("answer", "words"),
    [
        (
            sdk_http_error(403, {"status": "error", "message": "Invalid openalgo apikey"}),
            "HTTP 403: Invalid openalgo apikey",
        ),
        (
            {
                "status": "error",
                "message": f"HTTP 502: {PAGE}",
                "code": 502,
                "error_type": "http_error",
            },
            f"HTTP 502: {PAGE}",
        ),
        (
            {
                "status": "error",
                "message": "GTT orders are not supported for broker 'x' yet",
                "code": 501,
            },
            "HTTP 501: GTT orders are not supported for broker 'x' yet",
        ),
        (
            {"status": "error", "message": "Broker busy", "code": 200, "error_type": "api_error"},
            "Broker busy",
        ),
        (
            {"status": "error", "message": "Failed to connect", "error_type": "connection_error"},
            "Failed to connect",
        ),
        ({"status": "error", "code": 500}, "HTTP 500: an error without a message"),
    ],
    ids=[
        "server json inside the sdk's wrapper",
        "a page that is not json",
        "the server's body kept with its status",
        "an error answered with 200",
        "no server reached",
        "no message",
    ],
)
def test_an_error_answer_is_worded_with_its_http_status_once(answer, words):
    assert answer_reason(answer) == words


def test_a_long_message_is_clipped():
    answer = {"status": "error", "message": "x" * (MAX_QUOTED_CHARS + 50), "code": 500}

    assert answer_reason(answer) == "HTTP 500: " + "x" * MAX_QUOTED_CHARS + "..."
