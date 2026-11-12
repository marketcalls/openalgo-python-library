"""Shared setup for the pytest suites under test/.

Makes the repository root importable so the suites run against the source tree. These
suites never contact an OpenAlgo server or a broker. The older scripts in this folder
that do need a live server are run by hand, not collected by CI.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openalgo.indicators import _backend  # noqa: E402


@pytest.fixture(params=["rust", "numpy"])
def backend(request):
    """Run a test on the compiled indicator core, then on the pure NumPy fallback.

    Yields the backend name. The "rust" case is skipped when the extension is not
    installed.
    """
    had_rust = _backend.HAVE_RUST
    if request.param == "rust" and not had_rust:
        pytest.skip("the _oaindicators extension is not installed")
    _backend.HAVE_RUST = request.param == "rust"
    try:
        yield request.param
    finally:
        _backend.HAVE_RUST = had_rust


def _market(bars, seed, bar_range=0.5):
    """Deterministic OHLCV random walk: (open, high, low, close, volume) float arrays.

    `bar_range` scales how far the high and the low reach beyond the bar's body.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, bars))
    open_ = close + rng.normal(0.0, 0.4, bars)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, bar_range, bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, bar_range, bars))
    volume = rng.integers(1000, 5000, bars).astype(float)
    return open_, high, low, close, volume


@pytest.fixture(scope="session")
def market():
    """The `market(bars, seed, bar_range=0.5)` helper, handed out as a fixture."""
    return _market
