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


#: The marker of a test of the robustness reruns' worker processes themselves.
WORKER_POOL = "worker_pool"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", f"{WORKER_POOL}: a test that starts the robustness reruns' worker processes"
    )


def _refuse_pool(self):
    raise AssertionError(
        "a robustness test started the reruns' worker processes: pass workers=1 (the "
        "default starts one process a CPU, more memory than a test machine has), or mark "
        f"a test of the pool itself with @pytest.mark.{WORKER_POOL}"
    )


@pytest.fixture(autouse=True, scope="module")
def robustness_pool_guard(request):
    """In the robustness tests, starting the reruns' worker processes fails the test, so
    no test (nor a module fixture) starts one process a CPU by leaving `workers=` out.
    Yields the pool's own starter, or None outside those tests."""
    if not request.module.__name__.startswith("test_robust"):
        yield None
        return
    from openalgo.backtest._robust._reruns import Reruns

    starter = Reruns._executor
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Reruns, "_executor", _refuse_pool)
        yield starter


@pytest.fixture(autouse=True)
def worker_pool_allowed(request, robustness_pool_guard, monkeypatch):
    """A test marked `worker_pool` may start the reruns' worker processes."""
    if robustness_pool_guard is not None and request.node.get_closest_marker(WORKER_POOL):
        from openalgo.backtest._robust._reruns import Reruns

        monkeypatch.setattr(Reruns, "_executor", robustness_pool_guard)
