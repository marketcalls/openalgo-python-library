"""Fixtures of the opt-in live suite: checks against a running OpenAlgo server.

Every test here is skipped unless the environment variable OPENALGO_API_KEY holds an
API key. The server is http://127.0.0.1:5000 unless OPENALGO_HOST says otherwise; the
feed URL is derived from it unless OPENALGO_WS_URL is set.

    OPENALGO_API_KEY=... pytest test/live -v

Only test_live_orders.py places orders. It asserts that the server is in analyzer
(sandbox) mode immediately before every order, places one lot of an MCX future,
cancels or offsets everything it placed, and never calls an account-wide action.
The feed and order tests need the MCX session to be open; they skip otherwise.
"""

import pytest
from live_support import LIVE, new_client

from openalgo.backtest.data import OpenAlgoProvider


@pytest.fixture(scope="session")
def client():
    """A REST client for the server, closed at the end of the session."""
    if not LIVE:
        pytest.skip("set OPENALGO_API_KEY to run the live suite")
    rest = new_client()
    yield rest
    rest.close()


@pytest.fixture(scope="session")
def broker(client) -> str:
    """The broker the server is connected to, as the server names it."""
    name = OpenAlgoProvider(client).broker_name()
    assert name, "the server did not name its broker"
    return name
