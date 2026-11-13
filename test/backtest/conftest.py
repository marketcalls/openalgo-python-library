"""Shared setup for the backtest test suite.

These tests never contact an OpenAlgo server or a broker. Anything that needs a
client uses an in-process fake.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def fresh_unverified_rate_announcements():
    """A run announces unverified charge rates once per process (`_market.announce_once`);
    each test starts with none announced, so what it sees does not depend on the tests
    that ran before it."""
    market = sys.modules.get("openalgo.backtest._market")
    if market is not None:
        market.announce_once.cache_clear()
